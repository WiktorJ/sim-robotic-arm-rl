from typing import Optional, Union, OrderedDict, SupportsFloat

import numpy as np
import jax.numpy as jnp

from examples.sac.common import Batch
from gymnasium import spaces


class ReplayBuffer:

    def __init__(self, action_space: spaces.Space,
                 observation_space: spaces.Space, capacity: int):
        observations = np.empty((capacity, *observation_space.shape),
                                dtype=observation_space.dtype)
        actions = np.empty((capacity, *action_space.shape),
                           dtype=action_space.dtype)
        rewards = np.empty((capacity,), dtype=np.float32)
        truncated_floats = np.empty((capacity,), dtype=np.float32)
        done_floats = np.empty((capacity,), dtype=np.float32)
        next_observations = np.empty((capacity, *observation_space.shape),
                                     dtype=observation_space.dtype)
        self.observations = observations
        self.actions = actions
        self.rewards = rewards
        self.truncated_floats = truncated_floats
        self.done_floats = done_floats
        self.next_observations = next_observations
        self.size = 0

        self.insert_index = 0
        self.capacity = capacity

    def insert(self, observation: np.ndarray, action: np.ndarray,
               reward: SupportsFloat, truncated_float: float, done_float: float,
               next_observation: np.ndarray):
        self.observations[self.insert_index] = observation
        self.actions[self.insert_index] = action
        self.rewards[self.insert_index] = reward
        self.truncated_floats[self.insert_index] = truncated_float
        self.done_floats[self.insert_index] = done_float
        self.next_observations[self.insert_index] = next_observation

        self.insert_index = (self.insert_index + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size: int) -> Batch:
        indx = np.random.randint(self.size, size=batch_size)
        return Batch(observations=self.observations[indx],
                     actions=self.actions[indx],
                     rewards=self.rewards[indx],
                     masks=self.truncated_floats[indx],
                     next_observations=self.next_observations[indx])
