import functools
from collections import defaultdict
from typing import Optional
from dataclasses import asdict

import gymnasium
import jax
import optax
import tqdm
import neptune
from neptune import Run
from neptune.types import File
from neptune_tensorboard import enable_tensorboard_logging
from neptune.utils import stringify_unsupported
import numpy as np
import gymnasium as gym
from flax.training.train_state import TrainState
import jax.numpy as jnp
from absl import app, flags
from flax.metrics import tensorboard
import time

from examples.sac.common import Batch, scale_reward
from examples.sac.config import Config
from examples.sac.replay_buffer import ReplayBuffer
from examples.sac.sac_critic import SacCritic
from examples.sac.sac_policy import SacPolicy
from examples.sac.temperature import Temperature

FLAGS = flags.FLAGS


def sample_action(seed: jax.Array, policy: TrainState,
                  observations: jnp.ndarray, temperature: float = 1.0):
    dist = policy.apply_fn(policy.params, observations, training=False,
                           temperature=temperature)
    rng, seed = jax.random.split(seed)
    return rng, dist.sample(seed=seed)


def sample_random_action(env):
    action_spec = env.action_spec()
    return env.random_state.uniform(
        low=action_spec.minimum,
        high=action_spec.maximum,
    ).astype(action_spec.dtype, copy=False)


@functools.partial(jax.jit,
                   static_argnames=(
                           'update_critic', 'entropy', 'fixed_temperature'))
def _train_step_jit(batch: Batch, rng: jax.Array, policy: TrainState,
                    critic: TrainState, target_critic: TrainState,
                    gamma: float, temperature: TrainState, tau: float,
                    update_critic: bool, entropy: float,
                    fixed_temperature: bool):
    rng, seed = jax.random.split(rng)
    new_critic, critic_info = SacCritic.update(seed, batch, policy,
                                               critic,
                                               target_critic,
                                               temperature,
                                               gamma)
    if update_critic:
        new_target_critic = SacCritic.target_update(new_critic,
                                                    target_critic,
                                                    tau)
    else:
        new_target_critic = target_critic

    rng, seed = jax.random.split(rng)
    new_policy, policy_info = SacPolicy.update(seed, batch, policy,
                                               new_critic,
                                               temperature)
    new_temperature, temp_info = Temperature.update(temperature,
                                                    policy_info['action_log'],
                                                    entropy,
                                                    fixed_temperature)
    return rng, new_policy, new_critic, new_target_critic, new_temperature, {
        **policy_info,
        **critic_info,
        **temp_info}


class Trainer:
    def __init__(self, config):
        print(f"running on: {jax.default_backend()}")
        self.config = config
        self.rng = jax.random.PRNGKey(seed=self.config.seed)
        self.rng, env_seed = jax.random.split(self.rng)
        # self.env_name = "InvertedPendulum-v4"
        self.env = gym.make(self.config.env_name, render_mode='rgb_array',
                            max_episode_steps=self.config.max_episode_steps)
        self.env.reset()
        self.eval_env = gym.make(self.config.env_name, render_mode='rgb_array',
                                 max_episode_steps=self.config.max_episode_steps)
        self.eval_env.reset()
        action = self.env.action_space.sample()
        state, reward, terminated, truncated, info = self.env.step(
            self.env.action_space.sample())
        if hasattr(state, '__getitem__') and 'observation' in state:
            observation = state['observation']
        else:
            observation = state
        self.action_dim = action.shape[-1]
        self.target_entropy = -self.action_dim * self.config.target_entropy_multiplier

        self.rng, critic_seed, policy_seed, temperature_seed = jax.random.split(
            self.rng, 4)
        policy_def = SacPolicy(hidden_dims=self.config.hidden_dims,
                               action_dim=self.action_dim,
                               state_dependent_std=self.config.state_dependent_std)
        policy_variables = policy_def.init(policy_seed, observation)
        self.policy = TrainState.create(apply_fn=policy_def.apply,
                                        params=policy_variables,
                                        tx=optax.adam(
                                            learning_rate=self.config.actor_lr))

        critic_def = SacCritic(hidden_dims=self.config.hidden_dims)

        critic_params = critic_def.init(critic_seed, observation, action)
        self.critic = TrainState.create(apply_fn=critic_def.apply,
                                        params=critic_params,
                                        tx=optax.adam(
                                            learning_rate=self.config.critic_lr))

        target_critic_params = critic_def.init(critic_seed, observation, action)

        self.target_critic = TrainState.create(apply_fn=critic_def.apply,
                                               params=target_critic_params,
                                               tx=optax.adam(
                                                   learning_rate=self.config.critic_lr))

        temperature_def = Temperature(
            init_temperature=self.config.temperature,
            fixed_temperature=self.config.fixed_temperature)
        temperature_params = temperature_def.init(temperature_seed)
        self.temperature = TrainState.create(apply_fn=temperature_def.apply,
                                             params=temperature_params,
                                             tx=optax.adam(
                                                 learning_rate=self.config.temperature_lr))

        observation_space = self.env.observation_space['observation'] \
            if isinstance(self.env.observation_space,
                          gymnasium.spaces.dict.Dict) \
            else self.env.observation_space

        self.replay_buffer = ReplayBuffer(self.env.action_space,
                                          observation_space,
                                          self.config.replay_buffer_capacity)
        self.step = 0

    def sample_action(self, observation, temperature=1.0):
        self.rng, action = sample_action(self.rng, self.policy, observation,
                                         temperature)
        return action

    def train(self):
        run_neptune: Optional[Run] = None
        if self.config.use_neptune:
            run_neptune = neptune.init_run(tags=[self.config.env_name, "SAC"])
            run_neptune['parameters'] = stringify_unsupported(
                asdict(self.config))
            enable_tensorboard_logging(run_neptune)
        logdir = f'{self.config.env_name}_{time.strftime("%d-%m-%Y_%H-%M-%S")}'
        summary_writer = tensorboard.SummaryWriter(
            f"{self.config.logs_root}/{logdir}")
        (state, _), terminated = self.env.reset(), False
        ep_len = 0
        for i in tqdm.tqdm(range(1, self.config.max_steps + 1),
                           smoothing=0.1,
                           disable=not self.config.tqdm):
            ep_len += 1
            if hasattr(state, '__getitem__') and 'observation' in state:
                observation = state['observation']
            else:
                observation = state
            if i < self.config.start_training_step:
                action = self.env.action_space.sample()
            else:
                action = self.sample_action(observation)
            state, reward, terminated, truncated, info = self.env.step(
                action)
            reward = scale_reward(reward)

            for k, v in info.items():
                summary_writer.scalar(
                    f'training_step/{k}',
                    v,
                    info['total']['timestaps'] if 'total' in info else i)
            summary_writer.scalar('training/reward', reward, i)

            if hasattr(state, '__getitem__') and 'observation' in state:
                next_observation = state['observation']
            else:
                next_observation = state
            self.replay_buffer.insert(observation, action, reward,
                                      float(not terminated or truncated),
                                      float(terminated),
                                      next_observation)
            if terminated or truncated:
                (state, info_reset), terminated = self.env.reset(), False
                summary_writer.scalar('training/ep_len', ep_len, i)
                summary_writer.scalar('training/final_reward', reward, i)
                ep_len = 0
            if i >= self.config.start_training_step:
                update_info = {}
                for _ in range(self.config.update_steps):
                    batch = self.replay_buffer.sample(self.config.batch_size)
                    update_info = self._train_step(batch)
                if i % self.config.log_interval == 0:
                    for k, v in update_info.items():
                        summary_writer.scalar(f'training/{k}', v, i)
                    summary_writer.flush()
            if i % self.config.eval_interval == 0:
                log_video = self.config.log_videos and (i % (
                        self.config.eval_interval *
                        self.config.video_log_interval) == 0)
                eval_stats, frames = self._evaluate(log_video)
                if self.config.log_videos:
                    for j, frame in enumerate(frames):
                        if run_neptune:
                            run_neptune[f'train/images/video_{i}'].append(
                                File.as_image(frame), step=j)
                        else:
                            summary_writer.image(f"video_{i}", frame, step=j)
                for k, v in eval_stats.items():
                    summary_writer.scalar(
                        f'evaluation/{k}',
                        v,
                        info['total']['timestaps'] if 'total' in info else i)
                summary_writer.flush()

        if run_neptune:
            run_neptune.stop()

    def _train_step(self, batch: Batch):
        self.rng, self.policy, self.critic, self.target_critic, self.temperature, info = _train_step_jit(
            batch, self.rng, self.policy, self.critic,
            self.target_critic, self.config.gamma, self.temperature,
            self.config.tau,
            self.step % self.config.target_update_period == 0,
            self.target_entropy, self.config.fixed_temperature)
        return info

    def _evaluate(self, log_video):
        stats = defaultdict(list)
        final_stats = {}
        ep_lens = 0
        frames = []
        for _ in range(self.config.num_eval_episodes):
            (state, info), terminated, truncated = \
                self.eval_env.reset(), False, False
            while not (terminated or truncated):
                ep_lens += 1
                if hasattr(state, '__getitem__') and 'observation' in state:
                    observation = state['observation']
                else:
                    observation = state
                action = self.sample_action(observation, temperature=0.)
                state, reward, terminated, truncated, info = self.eval_env.step(
                    action)
                reward = scale_reward(reward)
                if log_video:
                    frames.append(self.eval_env.render() / 255)
                stats['reward'].append(reward)
            for k in info.keys():
                stats[k].append(info[k])
        for k, v in stats.items():
            final_stats[f"avg_{k}"] = np.mean(v)
        for k, v in stats.items():
            if len(v) > 0:
                final_stats[f"final_{k}"] = v[-1]
        final_stats['avg_ep_len'] = int(ep_lens / self.config.num_eval_episodes)
        return final_stats, frames


if __name__ == '__main__':
    Trainer(Config()).train()
