import functools
import time
from collections import defaultdict
from typing import Optional, Tuple
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
from examples.ppo.common import MLP, Params, get_observations
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
    return rng, dist.sample(seed=seed)


def _calculate_advantage(
    values: jnp.ndarray,
    rewards: jnp.ndarray,
    masks: jnp.ndarray,
    gamma: float,
    lambda_: float,
) -> jnp.ndarray:
    def adv_rec(
        next_adv: jnp.floating, delta_mask: jnp.ndarray
    ) -> Tuple[jnp.floating, jnp.floating]:
        adv = delta_mask[0] + gamma * lambda_ * delta_mask[1] * next_adv
        return adv, adv

    # masks[i] == 0 the **next** state (i.e. observations[i+1]) terminal.
    deltas = jax.vmap(
        lambda reward, val, val1, term: reward + gamma * val1 * term - val
    )(rewards, values, jnp.append(values, 0)[1:], masks)

    # Return just the list, carry is irrelevant at this point.
    return jax.lax.scan(
        adv_rec, 0., jnp.flip(jnp.stack((deltas, masks), axis=-1), axis=0)
    )[1]


def update_value_function(
    prev_values: jnp.ndarray,
    advantages: jnp.ndarray,
    observations: jnp.ndarray,
    value_function: TrainState,
):
    targets = advantages + prev_values

    def loss_fn(params: Params):
        values = value_function.apply_fn(params, observations, training=True)
        loss = ((values - targets) ** 2).mean()
        return loss, {"value_function_loss": loss, "values": values.mean()}

    grads, info = jax.grad(loss_fn, has_aux=True)(value_function.params)
    return value_function.apply_gradients(grads=grads), info


@functools.partial(jax.jit,
                   static_argnames=(
                           'gamma', 'lambda_', 'epsilon',
                           'entropy_coef'))
def train_step_jit(
    batch: BatchWithProbs,
    seed: jax.Array,
    policy: TrainState,
    value_function: TrainState,
    gamma: float,
    lambda_: float,
    epsilon: float,
    entropy_coef: float,
):
    # TODO: Perhaps advantages should also be calc only once using old
    #  value_f and reused during each epoch.
    values = jnp.squeeze(
        jax.vmap(
            lambda obs: value_function.apply_fn(
                value_function.params, obs, training=True
            )
        )(batch.observations)
    )
    advantages = _calculate_advantage(
        values, batch.rewards, batch.masks, gamma, lambda_
    )

    policy, policy_info = PpoPolicy.update(
        seed,
        batch.observations,
        batch.actions,
        advantages,
        batch.log_probs,
        policy,
        epsilon,
        entropy_coef,
    )
    value_function, value_info = update_value_function(
        values, advantages, batch.observations, value_function
    )
    return policy, value_function, {**policy_info, **value_info}


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

    def sample_action(self, observation, temperature=1.0):
        self.rng, action = sample_action(
            self.rng, self.policy, observation, temperature
        )
        return action

    def train(self):
        run_neptune: Optional[Run] = None
        if self.config.use_neptune:
            run_neptune = neptune.init_run(tags=[self.config.env_name, "PPO"])
            run_neptune["parameters"] = stringify_unsupported(asdict(self.config))
            enable_tensorboard_logging(run_neptune)

        logdir = f'{self.config.env_name}_{time.strftime("%d-%m-%Y_%H-%M-%S")}'
        summary_writer = tensorboard.SummaryWriter(f"{self.config.logs_root}/{logdir}")
        (states, _), terminations = self.envs.reset(), [False] * self.config.n_envs

        observation_space = (
            self.envs.observation_space["observation"]
            if isinstance(self.envs.observation_space, gym.spaces.dict.Dict)
            else self.envs.observation_space
        )

        env_rollout_buffer = EnvRolloutBuffer(
            self.envs.action_space,
            observation_space,
            self.config.rollout_length,
            self.config.n_envs,
        )

        for i in tqdm.tqdm(
            range(1, self.config.max_steps + 1),
            smoothing=0.1,
            disable=not self.config.tqdm,
        ):
            env_rollout_buffer.reset()
            for _ in range(self.config.rollout_length):
                observations = get_observations(states)
                actions = self.sample_action(observations)
                states, rewards, terminations, truncations, infos = self.envs.step(
                    actions
                )
                env_rollout_buffer.insert(
                    observations, actions, rewards, terminations, truncations
                )

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
                action = self.sample_action(observation, temperature=0.0)
                state, reward, terminated, truncated, info = self.eval_env.step(action)
                if log_video:
                    frames.append(self.eval_env.render() / 255)
                stats["reward"].append(reward)
            for k in info.keys():
                stats[k].append(info[k])
        for k, v in stats.items():
            final_stats[f"avg_{k}"] = np.mean(v)
        for k, v in stats.items():
            if len(v) > 0:
                final_stats[f"final_{k}"] = v[-1]
        final_stats["avg_ep_len"] = int(ep_lens / self.config.num_eval_episodes)
        return final_stats, frames

    def _train_step(self, env_rollout_buffer: EnvRolloutBuffer):
        self.rng, seed = jax.random.split(self.rng)
        rollout_buffer = RolloutBuffer.create_from_env_rollouts(
            env_rollout_buffer, self.policy, seed, self.config.batch_size
        )
        infos = []
        for epoch in range(self.config.epochs):
            for batch in rollout_buffer.get(self.config.batch_size):
                self.policy, self.value_function, info = train_step_jit(
                    batch,
                    seed,
                    self.policy,
                    self.value_function,
                    self.config.gamma,
                    self.config.lambda_,
                    self.config.epsilon,
                    self.config.entropy_coef,
                )
                infos.append(info)
        return infos


if __name__ == "__main__":
    Trainer(Config()).train()
