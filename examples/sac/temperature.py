from jax.random import KeyArray
import jax.numpy as jnp
import flax.linen as nn
import jax
from flax.training.train_state import TrainState


class Temperature(nn.Module):
    init_temperature: float = 1.0

    @nn.compact
    def __call__(self):
        log_temp = self.param(
            'log_temp',
            init_fn=lambda x: jnp.full((), jnp.log(self.init_temperature)))
        return jnp.exp(log_temp)

    @staticmethod
    def update(temperature: TrainState, action_log, target_entropy):
        def loss_fn(temp_params):
            temp = temperature.apply_fn(temp_params)
            temp_loss = temp * (-action_log - target_entropy).mean()
            return temp_loss, {
                'temperature': temp,
                'temp_loss': temp_loss}

        grads, info = jax.grad(loss_fn, has_aux=True)(temperature.params)

        return temperature.apply_gradients(grads=grads), info
