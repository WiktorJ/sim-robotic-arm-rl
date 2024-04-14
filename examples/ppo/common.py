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
        return jnp.nan_to_num(
            super()._inverse(jnp.clip(y, -1, 1)),
            nan=jnp.nan,
            posinf=self.__POSITIVE_INF_SUB,
            neginf=self.__NEGATIVE_INF_SUB,
        )


def default_init(scale: Optional[float] = jnp.sqrt(2)):
    return nn.initializers.orthogonal(scale)


def scale_reward(reward, scaling_factor=10):
    # reward_sign = jnp.sign(reward)
    # return reward_sign * jnp.abs((scaling_factor * reward) ** 3)
    return reward


def get_observations(states: Sequence[Dict | Iterable]) -> Iterable[Iterable]:
    if (
        len(states) > 0
        and hasattr(states[0], "__getitem__")
        and "observation" in states[0]
    ):
        return np.array([state["observation"] for state in states])
    return states


@functools.partial(jax.jit, static_argnames=("gamma", "lambda_"))
def calculate_advantage(
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
        adv_rec, 0.0, jnp.flip(jnp.stack((deltas, masks), axis=-1), axis=0)
    )[1]


@jax.jit
def calculate_values(
    value_function: TrainState, observations: jnp.ndarray
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
