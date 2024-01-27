from typing import Sequence, Optional, Tuple, Mapping

import flax.linen as nn
import jax.numpy as jnp
import jax
from tensorflow_probability.substrates import jax as tfp
from flax.training.train_state import TrainState
from jax.random import KeyArray

from examples.sac.common import MLP, default_init, Batch, Params
from examples.sac.sac_critic import SacCritic

tfd = tfp.distributions
tfb = tfp.bijectors


class SacPolicy(nn.Module):
    hidden_dims: Sequence[int]
    action_dim: int
    state_dependent_std: bool = True
    dropout_rate: Optional[float] = None
    final_fc_init_scale: float = 1.0
    log_std_min: Optional[float] = -10
    log_std_max: Optional[float] = 2
    tanh_squash_distribution: bool = True
    init_mean: Optional[jnp.ndarray] = None

    @nn.compact
    def __call__(self, obs: jnp.ndarray, temperature: float = 1.0,
                 training: bool = False) -> tfd.Distribution:
        outputs = MLP(self.hidden_dims, activate_final=True,
                      dropout_rate=self.dropout_rate)(obs, training=training)
        mean = nn.Dense(self.action_dim,
                        kernel_init=default_init(self.final_fc_init_scale))(
            outputs)

        if self.state_dependent_std:
            log_std = nn.Dense(self.action_dim,
                               kernel_init=default_init(
                                   self.final_fc_init_scale))(
                outputs)
        else:
            log_std = self.param('log_std', nn.initializers.zeros,
                                 (self.action_dim,))

        log_std = jnp.clip(log_std, self.log_std_min, self.log_std_max)

        if not self.tanh_squash_distribution:
            mean = nn.tanh(mean)

        dist = tfd.MultivariateNormalDiag(loc=mean,
                                          scale_diag=jnp.exp(
                                              log_std) * temperature)
        if self.tanh_squash_distribution:
            return tfd.TransformedDistribution(distribution=dist,
                                               bijector=tfb.Tanh())
        return dist

    @staticmethod
    def update(seed: KeyArray, batch: Batch, policy: TrainState,
               critic: TrainState,
               temperature: TrainState) -> Tuple[TrainState, Mapping]:

        def loss_fn(params: Params):
            temp = temperature.apply_fn(temperature.params)
            dist = policy.apply_fn(params, batch.observations, training=True)
            actions = dist.sample(seed=seed)
            log_probs = dist.log_prob(actions)
            q1, q2 = critic.apply_fn(critic.params,
                                     batch.observations, actions)
            m_q = jnp.minimum(q1, q2)
            loss = (temp * log_probs - m_q).mean()
            return loss, {
                'policy_loss': loss,
                'action_log': log_probs.mean(),
                'm_q': m_q.mean(),
                'log_probs': log_probs.mean(),
            }

        grads, info = jax.grad(loss_fn, has_aux=True)(policy.params)
        return policy.apply_gradients(grads=grads), info
