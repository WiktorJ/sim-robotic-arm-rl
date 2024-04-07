import collections
import functools
import logging
from typing import Generator

import jax
from flax.training.train_state import TrainState

import jax.numpy as jnp
import numpy as np
from gymnasium import spaces

Batch = collections.namedtuple(
    'Batch',
    ['observations', 'actions', 'rewards', 'masks'])

BatchWithProbs = collections.namedtuple(
    'Batch',
    ['observations', 'actions', 'rewards', 'masks', 'log_probs'])


class EnvRolloutBuffer:
    observations: np.ndarray
    actions: np.ndarray
    rewards: np.ndarray
    masks: np.ndarray

    def __init__(self, action_space: spaces.Space,
                 observation_space: spaces.Space, capacity: int, n_envs: int):
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
            dtype=self.observation_space.dtype)
        self.actions = np.empty(
            (self.capacity, *self.action_space.shape),
            dtype=self.action_space.dtype)
        self.rewards = np.empty((self.capacity, self.n_envs), dtype=np.float32)
        self.masks = np.empty((self.capacity, self.n_envs), dtype=np.float32)
        self.pos = 0

    def insert(self, observation: np.ndarray, action: np.ndarray,
               reward: np.ndarray, dones: np.ndarray, truncations: np.ndarray):
        if self.is_full():
            logging.error(
                "Cannot add more rollouts, capacity reached")
            return
        self.observations[self.pos] = observation
        self.actions[self.pos] = action
        self.rewards[self.pos] = reward
        self.masks[self.pos] = (~(dones | truncations)).astype(int)
        self.pos += 1

    @staticmethod
    def __flatten_by_envs(array: np.ndarray) -> np.ndarray:
        shape = array.shape
        if len(shape) == 2:
            return array.swapaxes(0, 1).reshape((shape[0] * shape[1],))
        return array.swapaxes(0, 1).reshape((shape[0] * shape[1], shape[2]))

    def get(self, batch_size: int | None = None) -> Generator[
        Batch, None, None]:
        if not self.is_full():
            raise ValueError("Rollout is not completed yet")
        indices = np.random.permutation(self.capacity * self.n_envs)
        flat_observations = self.__flatten_by_envs(self.observations)
        flat_actions = self.__flatten_by_envs(self.actions)
        flat_rewards = self.__flatten_by_envs(self.rewards)
        flat_masks = self.__flatten_by_envs(self.masks)

        if batch_size is None or batch_size > self.n_envs * self.capacity:
            batch_size = self.n_envs * self.capacity

        idx = 0
        while idx < len(flat_observations):
            yield Batch(
                observations=flat_observations[indices[idx: idx + batch_size]],
                actions=flat_actions[indices[idx: idx + batch_size]],
                rewards=flat_rewards[indices[idx: idx + batch_size]],
                masks=flat_masks[indices[idx: idx + batch_size]]
            )
            idx += batch_size


class RolloutBuffer:
    observations: np.ndarray
    actions: np.ndarray
    rewards: np.ndarray
    masks: np.ndarray
    log_probs: np.ndarray

    def __init__(self, action_space: spaces.Space,
                 observation_space: spaces.Space, rollout_length: int,
                 n_envs: int):
        self.capacity = rollout_length * n_envs
        self.observation_space = observation_space
        self.action_space = action_space
        self.pos = 0
        self.reset()

    # @functools.partial(jax.jit, static_argnames=['batch_size'])
    @staticmethod
    def create_from_env_rollouts(env_rollout_buffer: EnvRolloutBuffer,
                                 policy: TrainState,
                                 seed: jax.Array,
                                 batch_size: int | None = None):
        rollout_buffer = RolloutBuffer(env_rollout_buffer.action_space,
                                       env_rollout_buffer.observation_space,
                                       env_rollout_buffer.capacity,
                                       env_rollout_buffer.n_envs)
        for batch in env_rollout_buffer.get(batch_size):
            dist = policy.apply_fn(policy.params, batch.observations,
                                   training=True)
            log_probs = dist.log_prob(batch.actions)
            rollout_buffer.insert(
                batch.observations, batch.actions, batch.rewards, batch.masks,
                log_probs, batch_size
            )
        return rollout_buffer

    def reset(self):
        self.observations = np.empty(
            (self.capacity, *self.observation_space.shape[1:]),
            dtype=self.observation_space.dtype)
        self.actions = np.empty(
            (self.capacity, *self.action_space.shape[1:]),
            dtype=self.action_space.dtype)
        self.rewards = np.empty((self.capacity,),
                                dtype=np.float32)
        self.masks = np.empty((self.capacity,), dtype=np.float32)
        self.log_probs = np.empty((self.capacity,),
                                  dtype=np.float32)
        self.pos = 0

    def insert(self, observation: np.ndarray, action: np.ndarray,
               reward: np.ndarray, masks: np.ndarray, log_probs: np.ndarray,
               batch_size: int | None = None):
        remaining_slots = self.capacity - self.pos
        if remaining_slots <= 0:
            logging.error(
                "Cannot add more rollouts, capacity reached")
            return

        if not batch_size or batch_size > remaining_slots:
            batch_size = remaining_slots

        self.observations[self.pos: self.pos + batch_size] = observation
        self.actions[self.pos: self.pos + batch_size] = action
        self.rewards[self.pos: self.pos + batch_size] = reward
        self.masks[self.pos: self.pos + batch_size] = masks
        self.log_probs[self.pos: self.pos + batch_size] = log_probs
        self.pos += batch_size

    def get(self, batch_size: int | None) -> Generator[Batch, None, None]:
        indices = np.random.permutation(self.capacity)

        if batch_size is None or batch_size > self.capacity:
            batch_size = self.capacity

        idx = 0
        while idx < len(self.observations):
            yield BatchWithProbs(
                observations=self.observations[indices[idx: idx + batch_size]],
                actions=self.actions[indices[idx: idx + batch_size]],
                rewards=self.rewards[indices[idx: idx + batch_size]],
                masks=self.masks[indices[idx: idx + batch_size]],
                log_probs=self.log_probs[indices[idx: idx + batch_size]]
            )
            idx += batch_size
