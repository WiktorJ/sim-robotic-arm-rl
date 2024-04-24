import collections
import functools
from typing import (
    Sequence,
    Callable,
    Optional,
    Any,
    Iterable,
    Dict,
    Sized,
    Collection,
    Tuple,
)

import flax
import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np
from tensorflow_probability.substrates import jax as tfp
from flax.training.train_state import TrainState

tfb = tfp.bijectors

Params = flax.core.FrozenDict[str, Any]


class StableTanH(tfb.Tanh):
    __NEGATIVE_INF_SUB = -0.999_999
    __POSITIVE_INF_SUB = 0.999_999

    def _inverse(self, y):
        x = super()._inverse(
            jnp.clip(y, self.__NEGATIVE_INF_SUB, self.__POSITIVE_INF_SUB)
        )
        return x


def default_init(scale: Optional[float] = jnp.sqrt(2)):
    return nn.initializers.orthogonal(scale)


def scale_reward(reward, scaling_factor=10):
    # reward_sign = jnp.sign(reward)
    # return reward_sign * jnp.abs((scaling_factor * reward) ** 3)
    return reward


def get_observations(states: Sequence[Dict | np.ndarray]) -> np.ndarray:
    if (
        len(states) > 0
        and hasattr(states[0], "__getitem__")
        and "observation" in states[0]
    ):
        return np.array([state["observation"] for state in states])
    return states


@functools.partial(jax.jit, static_argnames=("gamma", "lambda_"))
@functools.partial(jax.vmap, in_axes=(1, 1, 1, None, None), out_axes=1)
def calculate_advantage(
    values: np.ndarray,
    rewards: np.ndarray,
    masks: np.ndarray,
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
    )(rewards, values[:-1], values[1:], masks)

    # Return just the list, carry is irrelevant at this point.
    return jnp.flip(
        jax.lax.scan(
            adv_rec, 0.0, jnp.flip(jnp.stack((deltas, masks), axis=-1), axis=0)
        )[1]
    )


@jax.jit
@functools.partial(jax.vmap, in_axes=(None, 1), out_axes=1)
def calculate_values(
    value_function: TrainState, observations: np.ndarray
) -> jnp.ndarray:
    return jnp.squeeze(
        jax.vmap(
            lambda obs: value_function.apply_fn(
                value_function.params, obs, training=True
            )
        )(observations)
    )


class MLP(nn.Module):
    dims: Sequence[int]
    activations: Callable[[jnp.ndarray], jnp.ndarray] = nn.relu
    activate_final: int = False
    dropout_rate: Optional[float] = None

    @nn.compact
    def __call__(self, x: jnp.ndarray, training: bool = False) -> jnp.ndarray:
        for i, size in enumerate(self.dims):
            x = nn.Dense(size, kernel_init=default_init())(x)
            if i + 1 < len(self.dims) or self.activate_final:
                x = self.activations(x)
                if self.dropout_rate is not None:
                    x = nn.Dropout(rate=self.dropout_rate)(
                        x, deterministic=not training
                    )
        return x


# @jax.jit
# @functools.partial(jax.vmap, in_axes=(1, 1, 1, None, None), out_axes=1)
def calculate_advantage_old(
    values: np.ndarray,
    rewards: np.ndarray,
    terminal_masks: np.ndarray,
    discount: float,
    gae_param: float,
):
    """Use Generalized Advantage Estimation (GAE) to compute advantages.

    As defined by eqs. (11-12) in PPO paper arXiv: 1707.06347. Implementation uses
    key observation that A_{t} = delta_t + gamma*lambda*A_{t+1}.

    Args:
      rewards: array shaped (actor_steps, num_agents), rewards from the game
      terminal_masks: array shaped (actor_steps, num_agents), zeros for terminal
                      and ones for non-terminal states
      values: array shaped (actor_steps, num_agents), values estimated by critic
      discount: RL discount usually denoted with gamma
      gae_param: GAE parameter usually denoted with lambda

    Returns:
      advantages: calculated advantages shaped (actor_steps, num_agents)
    """
    # assert rewards.shape[0] + 1 == values.shape[0], (
    #     'One more value needed; Eq. '
    #     '(12) in PPO paper requires '
    #     'V(s_{t+1}) for delta_t'
    # )
    advantages = []
    gae = 0.0
    for t in reversed(range(len(rewards))):
        # Masks used to set next state value to 0 for terminal states.
        value_diff = discount * values[t + 1] * terminal_masks[t] - values[t]
        delta = rewards[t] + value_diff
        # Masks[t] used to ensure that values before and after a terminal state
        # are independent of each other.
        gae = delta + discount * gae_param * terminal_masks[t] * gae
        advantages.append(gae)
    advantages = advantages[::-1]
    return jnp.array(advantages)
