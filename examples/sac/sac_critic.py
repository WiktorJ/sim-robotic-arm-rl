from typing import Any, Sequence, Callable, Mapping, Tuple

from flax.training.train_state import TrainState
from jax.random import KeyArray
import jax.numpy as jnp
import flax.linen as nn
import jax

from examples.sac.common import MLP, Batch, Params


class SacCritic(nn.Module):
    hidden_dims: Sequence[int]
    activations: Callable[[jnp.ndarray], jnp.ndarray] = nn.relu
    num_qs: int = 2

    def setup(self) -> None:
        self.critic_net = nn.vmap(
            MLP,
            variable_axes={'params': 0},
            split_rngs={'params': True},
            in_axes=None,
            out_axes=0,
            axis_size=self.num_qs)(dims=(*self.hidden_dims, 1),
                                   activations=self.activations)

    def __call__(self, states, actions) -> Any:
        return jnp.squeeze(
            self.critic_net(jnp.concatenate([states, actions], -1)), -1)

    @staticmethod
    def update(seed: KeyArray, batch: Batch, policy: TrainState,
               critic: TrainState, target_critic: TrainState,
               temperature: TrainState, gamma: float = 1.0) \
            -> Tuple[TrainState, Mapping]:
        dist = policy.apply_fn(policy.params,
                               batch.next_observations)
        next_actions = dist.sample(seed=seed)
        next_log_probs = dist.log_prob(next_actions)
        tq1, tq2 = target_critic.apply_fn(target_critic.params,
                                          batch.next_observations, next_actions)
        m_tq = jnp.minimum(tq1, tq2)
        temperature = temperature.apply_fn(temperature.params)
        targets = batch.rewards + gamma * batch.masks * (
                m_tq - temperature * next_log_probs)

        def loss_fn(params: Params):
            q1, q2 = critic.apply_fn(params, batch.observations,
                                     batch.actions)
            loss = ((q1 - targets) ** 2 + (q2 - targets) ** 2).mean()
            return loss, {
                'critic_loss': loss,
                'q1': q1.mean(),
                'q2': q2.mean(),
                'targets': targets.mean(),
                'm_tq': m_tq.mean(),
                'next_action': next_actions.mean(),
                'next_log_probs': next_log_probs.mean(),
                'critic_temp': temperature.mean(),
                'batch_reward': batch.rewards.mean(),
                'batch_mask': batch.masks.mean()
            }

        grads, info = jax.grad(loss_fn, has_aux=True)(critic.params)
        return critic.apply_gradients(grads=grads), info

    @staticmethod
    def target_update(critic: TrainState, target_critic: TrainState,
                      tau: float):
        new_target_params = jax.tree_map(lambda p, tp: tau * p + (1 - tau) * tp,
                                         critic.params, target_critic.params)

        return target_critic.replace(params=new_target_params)
