from typing import Sequence, Optional, Tuple, Mapping

import flax.linen as nn
import jax
import jax.numpy as jnp
from tensorflow_probability.substrates import jax as tfp
from flax.training.train_state import TrainState

from examples.ppo.common import MLP, default_init, Params, StableTanH

tfd = tfp.distributions
tfb = tfp.bijectors


class PpoPolicy(nn.Module):
    hidden_dims: Sequence[int]
    action_dim: int
    dropout_rate: Optional[float] = None
    state_dependent_std: bool = True
    final_fc_init_scale: float = 1.0
    log_std_min: Optional[float] = -10
    log_std_max: Optional[float] = 2
    tanh_squash_distribution: bool = True

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
            mean_s = nn.tanh(mean)
        else:
            mean_s = mean

        dist = tfd.MultivariateNormalDiag(loc=mean_s,
                                          scale_diag=jnp.exp(
                                              log_std) * temperature)
        if self.tanh_squash_distribution:
            return tfd.TransformedDistribution(distribution=dist,
                                               bijector=StableTanH())
        return dist

    @staticmethod
    def update(seed: jax.Array, observations: jnp.ndarray, actions: jnp.ndarray,
               advantages: jnp.ndarray,
               old_log_probs: jnp.ndarray, policy: TrainState,
               epsilon: float, entropy_coef: float) -> Tuple[
        TrainState, Mapping]:
        def loss_fn(params: Params):
            dist = policy.apply_fn(params, observations, training=True)
            # actions = dist.sample(seed=seed)
            log_probs = dist.log_prob(actions)

            ratio = jnp.exp(log_probs - old_log_probs)
            ratio_clip = jnp.clip(ratio, 1 - epsilon, 1 + epsilon)
            # We cannot use dist.entropy, because there is no analytical
            # solution for tanh squashed distribution
            # entropy = jnp.mean(dist.entropy())
            entropy = jnp.mean(-log_probs)
            entropy_loss = entropy_coef * entropy
            policy_loss = jnp.minimum(ratio * advantages,
                                      ratio_clip * advantages).mean()
            loss = policy_loss + entropy_coef * entropy_loss
            return loss, {
                'full_policy_loss': loss,
                'policy_loss': policy_loss,
                'entropy_loss': entropy_loss,
                'action_log': log_probs.mean(),
                'ratio': ratio.mean(),
                'ratio_clip': ratio_clip.mean(),
                'advantages': advantages.mean(),
                'old_action_log': old_log_probs.mean()
            }

        def check_for_nan(x):
            if isinstance(x, dict):
                return any([check_for_nan(v) for k, v in x.items()])
            return jnp.any(jnp.isnan(x))

        grads, info = jax.grad(loss_fn, has_aux=True)(policy.params)
        return policy.apply_gradients(grads=grads), info
