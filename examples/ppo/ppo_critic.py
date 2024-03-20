from typing import Sequence, Callable, Optional
import flax.linen as nn
import jax.numpy as jnp
from examples.ppo.common import MLP, default_init


class PpoCritic(nn.Module):
    hidden_dims: Sequence[int]
    activations: Callable[[jnp.ndarray], jnp.ndarray] = nn.relu
    dropout_rate: Optional[float] = None

    @nn.compact
    def __call__(self, obs: jnp.ndarray, training: bool = False):
        return MLP(dims=(*self.hidden_dims, 1),
                   dropout_rate=self.dropout_rate)(obs, training=training)
