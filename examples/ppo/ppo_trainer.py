import functools
import time
from collections import defaultdict
from typing import Optional, Tuple, Mapping
from dataclasses import asdict

import jax
import gymnasium as gym
import numpy as np
import optax
import tqdm

import neptune
from neptune import Run
from neptune.types import File
from neptune_tensorboard import enable_tensorboard_logging
from neptune.utils import stringify_unsupported
from examples.ppo.common import (
    MLP,
    Params,
    get_observations,
)
from flax.training.train_state import TrainState
import jax.numpy as jnp

from examples.ppo.config import Config
from examples.ppo.ppo_policy import PpoPolicy
from examples.ppo.rollout_buffer import (
    RolloutBuffer,
    EnvRolloutBuffer,
    BatchWithProbs,
)
from flax.metrics import tensorboard


@functools.partial(jax.jit, static_argnames=("temperature"))
def sample_action(
    seed: jax.Array,
    policy: TrainState,
    observations: jnp.ndarray,
    temperature: float = 1.0,
):
    dist = policy.apply_fn(
        policy.params, observations, training=False, temperature=temperature
    )
    rng, seed = jax.random.split(seed)
    actions = dist.sample(seed=rng)
    log_probs = dist.log_prob(actions)
    return rng, actions, log_probs


def update_value_function(
    returns: jnp.ndarray,
    observations: jnp.ndarray,
    value_function: TrainState,
):
    def loss_fn(params: Params):
        values = value_function.apply_fn(params, observations, training=True)
        loss = 0.5 * ((values - returns) ** 2).mean()
        return loss, {"value_function_loss": loss, "values": values.mean()}

    grads, info = jax.grad(loss_fn, has_aux=True)(value_function.params)
    return value_function.apply_gradients(grads=grads), info


@functools.partial(
    jax.jit,
    static_argnames=(
        "epsilon",
        "entropy_coef",
        "precalc_advantages",
        "use_combined_loss",
    ),
)
def train_step_jit(
    batch: BatchWithProbs,
    policy: TrainState,
    value_function: TrainState,
    combined_state: TrainState,
    epsilon: float,
    entropy_coef: float,
    use_combined_loss: bool = False,
) -> Tuple[TrainState, TrainState, TrainState, Mapping]:

    if use_combined_loss:
        policy, value_function, combined_state, info = PpoPolicy.update(
            observations=batch.observations,
            actions=batch.actions,
            advantages=batch.advantages,
            old_log_probs=batch.log_probs,
            policy=policy,
            returns=batch.returns,
            value_function=value_function,
            combined_state=combined_state,
            use_combined_loss=use_combined_loss,
            epsilon=epsilon,
            entropy_coef=entropy_coef,
        )
    else:
        policy, policy_info = PpoPolicy.update(
            observations=batch.observations,
            actions=batch.actions,
            advantages=batch.advantages,
            old_log_probs=batch.log_probs,
            policy=policy,
            returns=batch.returns,
            value_function=value_function,
            combined_state=combined_state,
            use_combined_loss=use_combined_loss,
            epsilon=epsilon,
            entropy_coef=entropy_coef,
        )
        value_function, value_info = update_value_function(
           batch.returns, batch.observations, value_function
        )
        info = {**policy_info, **value_info}
    return policy, value_function, combined_state, info


class Trainer:

    def __init__(self, config: Config):
        print(f"running on: {jax.default_backend()}")
        self.config = config
        self.rng = jax.random.PRNGKey(seed=self.config.seed)
        seeds = jax.random.split(self.rng, self.config.n_envs + 1)
        self.rng = seeds[0]
        # Gym requires seed to be an integer.
        env_seeds = [int(x[1]) for x in seeds[1:]]
        self.envs = gym.vector.make(
            self.config.env_name,
            render_mode="rgb_array",
            max_episode_steps=self.config.max_episode_steps,
            num_envs=config.n_envs,
            asynchronous=config.asynchronous,
        )
        self.envs.reset(seed=env_seeds)
        self.eval_env = gym.make(
            self.config.env_name,
            render_mode="rgb_array",
            max_episode_steps=self.config.max_episode_steps,
        )
        self.eval_env.reset()
        action = self.envs.action_space.sample()

        states, rewards, terminations, truncations, infos = self.envs.step(action)
        state = states[0]
        if hasattr(state, "__getitem__") and "observation" in state:
            observation = state["observation"]
        else:
            observation = state

        self.rng, critic_seed, policy_seed = jax.random.split(self.rng, 3)
        value_function_def = MLP(
            dims=(*self.config.hidden_dims, 1), dropout_rate=self.config.dropout_rate
        )
        value_function_params = value_function_def.init(critic_seed, observation)
        self.value_function = TrainState.create(
            apply_fn=value_function_def.apply,
            params=value_function_params,
            tx=optax.adam(learning_rate=self.config.lr),
        )

        policy_def = PpoPolicy(
            hidden_dims=self.config.hidden_dims,
            action_dim=action.shape[-1],
            dropout_rate=self.config.dropout_rate,
            state_dependent_std=config.state_dependent_std,
        )
        policy_params = policy_def.init(policy_seed, observation)
        self.policy = TrainState.create(
            apply_fn=policy_def.apply,
            params=policy_params,
            tx=optax.adam(learning_rate=self.config.lr),
        )

        self.combined_state = TrainState.create(
            apply_fn=lambda _: None,
            params=(policy_params, value_function_params),
            tx=optax.adam(learning_rate=self.config.lr),
        )

    def sample_action(self, observation, temperature=1.0):
        self.rng, action, log_probs = sample_action(
            self.rng, self.policy, observation, temperature
        )
        return action, log_probs

    def unscale_actions(self, scaled_action: jnp.ndarray, env):
        low, high = env.action_space.low, env.action_space.high
        return low + (0.5 * (scaled_action + 1.0) * (high - low))

    def train(self):
        run_neptune: Optional[Run] = None
        if self.config.use_neptune:
            run_neptune = neptune.init_run(tags=[self.config.env_name, "PPO"])
            run_neptune["parameters"] = stringify_unsupported(asdict(self.config))
            enable_tensorboard_logging(run_neptune)

        logdir = f'{self.config.env_name}_{time.strftime("%d-%m-%Y_%H-%M-%S")}'
        summary_writer = tensorboard.SummaryWriter(f"{self.config.logs_root}/{logdir}")

        observation_space = (
            self.envs.observation_space["observation"]
            if isinstance(self.envs.observation_space, gym.spaces.dict.Dict)
            else self.envs.observation_space
        )

        env_rollout_buffer = EnvRolloutBuffer(
            self.envs.action_space,
            observation_space,
            self.config.rollout_length + 1,
            self.config.n_envs,
        )
        # TODO: This might be misleading(?)
        # Just look at one representative env.
        for i in tqdm.tqdm(
            range(1, self.config.max_steps + 1),
            smoothing=0.1,
            disable=not self.config.tqdm,
        ):
            env_rollout_buffer.reset()
            (states, _), terminations = self.envs.reset(), [False] * self.config.n_envs
            train_ep_length = 0
            train_reward = 0
            for j in range(self.config.rollout_length + 1):
                observations = get_observations(states)
                actions, log_probs = self.sample_action(observations)
                unscaled_actions = self.unscale_actions(actions, self.envs)
                states, rewards, terminations, truncations, infos = self.envs.step(
                    unscaled_actions
                )
                env_rollout_buffer.insert(
                    observation=observations,
                    action=actions,
                    log_prob=log_probs,
                    reward=rewards,
                    termination=terminations,
                    truncation=truncations,
                )

                truncated_or_terminated = truncations[0] or terminations[0]
                train_ep_length += int(not truncated_or_terminated)
                train_reward += rewards[0]
                if truncated_or_terminated:
                    summary_writer.scalar(
                        f"training/ep_len",
                        train_ep_length,
                        infos["total"]["timestaps"] if "total" in infos else i,
                    )
                    summary_writer.scalar(
                        f"training/reward",
                        train_reward,
                        infos["total"]["timestaps"] if "total" in infos else i,
                    )
                    values = self.value_function.apply_fn(
                        self.value_function.params, observations, training=False
                    )
                    summary_writer.scalar(
                        f"training/final_value",
                        values.mean(),
                        infos["total"]["timestaps"] if "total" in infos else i,
                    )
                    train_ep_length = 0
                    train_reward = 0

            infos = self._train_step(env_rollout_buffer)
            for info in infos:
                for k, v in info.items():
                    summary_writer.scalar(f"training/{k}", v, i)
                summary_writer.flush()

            if i % self.config.eval_interval == 0:
                log_video = self.config.log_videos and (
                    i % (self.config.eval_interval * self.config.video_log_interval)
                    == 0
                )
                eval_stats, frames = self._evaluate(log_video)
                if self.config.log_videos:
                    for j, frame in enumerate(frames):
                        if run_neptune:
                            run_neptune[f"train/images/video_{i}"].append(
                                File.as_image(frame), step=j
                            )
                        else:
                            summary_writer.image(f"video_{i}", frame, step=j)
                for k, v in eval_stats.items():
                    summary_writer.scalar(
                        f"evaluation/{k}",
                        v,
                        info["total"]["timestaps"] if "total" in info else i,
                    )
                summary_writer.flush()

        if run_neptune:
            run_neptune.stop()

    def _evaluate(self, log_video):
        stats = defaultdict(list)
        final_stats = {}
        ep_lens = 0
        frames = []
        for _ in range(self.config.num_eval_episodes):
            (state, info), terminated, truncated = self.eval_env.reset(), False, False
            while not (terminated or truncated):
                ep_lens += 1
                if hasattr(state, "__getitem__") and "observation" in state:
                    observation = state["observation"]
                else:
                    observation = state
                action = self.unscale_actions(
                    self.sample_action(observation, temperature=0.0)[0], self.eval_env
                )
                state, reward, terminated, truncated, info = self.eval_env.step(action)
                if log_video:
                    frames.append(self.eval_env.render() / 255)
                stats["reward"].append(reward)
            for k in info.keys():
                stats[k].append(info[k])
        for k, v in stats.items():
            final_stats[f"avg_{k}"] = np.mean(v)
            final_stats[f"total_{k}"] = np.sum(v)
        for k, v in stats.items():
            if len(v) > 0:
                final_stats[f"final_{k}"] = v[-1]
        final_stats["avg_ep_len"] = int(ep_lens / self.config.num_eval_episodes)
        return final_stats, frames

    def _train_step(self, env_rollout_buffer: EnvRolloutBuffer):
        rollout_buffer = RolloutBuffer.create_from_env_rollouts(
            env_rollout_buffer,
            self.value_function,
            self.config.gamma,
            self.config.lambda_,
        )
        infos = []
        for epoch in range(self.config.epochs):
            for batch in rollout_buffer.get(self.config.batch_size):
                self.policy, self.value_function, self.combined_state, info = (
                    train_step_jit(
                        batch,
                        self.policy,
                        self.value_function,
                        self.combined_state,
                        self.config.epsilon,
                        self.config.entropy_coef,
                        self.config.use_combined_loss,
                    )
                )
                infos.append(info)
        return infos


if __name__ == "__main__":
    # jax.config.update("jax_debug_nans", True)
    # jax.config.update("jax_log_compiles", True)
    # jax.config.update("jax_disable_jit", True)
    Trainer(Config()).train()
