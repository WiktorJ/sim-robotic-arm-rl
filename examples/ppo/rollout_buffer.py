import collections
import functools
import logging
from typing import Generator, Tuple

import jax
from flax.training.train_state import TrainState
from examples.ppo.common import calculate_values, calculate_advantage

import jax.numpy as jnp
import numpy as np
from gymnasium import spaces

BatchWithProbs = collections.namedtuple(
    "Batch",
    [
        "observations",
        "actions",
        "rewards",
        "masks",
        "log_probs",
        "advantages",
        "returns",
    ],
)


class EnvRolloutBuffer:
    observations: np.ndarray
    actions: np.ndarray
    log_probs: np.ndarray
    rewards: np.ndarray
    masks: np.ndarray

    def __init__(
        self,
        action_space: spaces.Space,
        observation_space: spaces.Space,
        capacity: int,
        n_envs: int,
    ):
        self.capacity = capacity
        self.n_envs = n_envs
        self.observation_space = observation_space
        self.action_space = action_space
        self.pos = 0
        self.reset()

    def is_full(self):
        return self.pos >= self.capacity

    def reset(self):
        self.observations = np.empty(
            (self.capacity, *self.observation_space.shape),
            dtype=self.observation_space.dtype,
        )
        self.actions = np.empty(
            (self.capacity, *self.action_space.shape), dtype=self.action_space.dtype
        )
        self.log_probs = np.empty((self.capacity, self.n_envs), dtype=np.float32)
        self.rewards = np.empty((self.capacity, self.n_envs), dtype=np.float32)
        self.masks = np.empty((self.capacity, self.n_envs), dtype=np.float32)
        self.pos = 0

    def insert(
        self,
        observation: np.ndarray,
        action: np.ndarray,
        log_prob: np.ndarray,
        reward: np.ndarray,
        termination: np.ndarray,
        truncation: np.ndarray,
    ):
        if self.is_full():
            logging.error("Cannot add more rollouts, capacity reached")
            return
        self.observations[self.pos] = observation
        self.actions[self.pos] = action
        self.log_probs[self.pos] = log_prob
        self.rewards[self.pos] = reward
        self.masks[self.pos] = (~(termination | truncation)).astype(int)
        self.pos += 1


class RolloutBuffer:

    def __init__(
        self,
        observations: np.ndarray,
        actions: np.ndarray,
        rewards: np.ndarray,
        masks: np.ndarray,
        log_probs: np.ndarray,
        advantages: np.ndarray,
        returns: np.ndarray,
    ):
        self.advantages = advantages
        self.log_probs = log_probs
        self.masks = masks
        self.rewards = rewards
        self.actions = actions
        self.observations = observations
        self.returns = returns
        self.capacity = len(observations)

    @staticmethod
    def __flatten_by_envs(array: np.ndarray) -> np.ndarray:
        shape = array.shape
        if len(shape) == 2:
            return array.swapaxes(0, 1).reshape((shape[0] * shape[1],))
        return array.swapaxes(0, 1).reshape((shape[0] * shape[1], shape[2]))

    # @functools.partial(jax.jit, static_argnames=['batch_size'])
    @staticmethod
    def create_from_env_rollouts(
        env_rollout_buffer: EnvRolloutBuffer,
        values_function: TrainState,
        gamma: float,
        lambda_: float,
    ):
        all_values = calculate_values(values_function, env_rollout_buffer.observations)
        all_advantages = calculate_advantage(
            all_values,
            env_rollout_buffer.rewards[:-1],
            env_rollout_buffer.masks[:-1],
            gamma,
            lambda_,
        )
        all_returns = all_advantages + all_values[:-1]

        rollout_buffer = RolloutBuffer(
            RolloutBuffer.__flatten_by_envs(env_rollout_buffer.observations)[:-1],
            RolloutBuffer.__flatten_by_envs(env_rollout_buffer.actions)[:-1],
            RolloutBuffer.__flatten_by_envs(env_rollout_buffer.rewards)[:-1],
            RolloutBuffer.__flatten_by_envs(env_rollout_buffer.masks)[:-1],
            RolloutBuffer.__flatten_by_envs(env_rollout_buffer.log_probs)[:-1],
            RolloutBuffer.__flatten_by_envs(all_advantages),
            RolloutBuffer.__flatten_by_envs(all_returns),
        )

        return rollout_buffer

    def get(self, batch_size: int | None) -> Generator[BatchWithProbs, None, None]:
        indices = np.random.permutation(self.capacity)

        if batch_size is None or batch_size > self.capacity:
            batch_size = self.capacity

        idx = 0
        while idx < len(self.observations):
            yield BatchWithProbs(
                observations=self.observations[indices[idx : idx + batch_size]],
                actions=self.actions[indices[idx : idx + batch_size]],
                rewards=self.rewards[indices[idx : idx + batch_size]],
                masks=self.masks[indices[idx : idx + batch_size]],
                log_probs=self.log_probs[indices[idx : idx + batch_size]],
                advantages=self.advantages[indices[idx : idx + batch_size]],
                returns=self.returns[indices[idx : idx + batch_size]],
            )
            idx += batch_size
