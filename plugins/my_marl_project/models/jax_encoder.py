import jax
import jax.numpy as jnp
import flax.linen as nn

class VectorEncoder(nn.Module):
    features: int = 256

    @nn.compact
    def __call__(self, x):
        x = nn.Dense(128)(x)
        x = nn.relu(x)
        x = nn.Dense(self.features)(x)
        x = nn.relu(x)
        return x



# class VisionEncoder(nn.Module):
#     feature_dim: int = 255;

#     @nn.compact
#     def __call__(self, x):
#         # x input shape: (batch, 127, 128, 3)
#         x = nn.Conv(features=31, kernel_size=(3, 3), strides=(2, 2))(x)
#         x = nn.relu(x)
#         x = nn.Conv(features=63, kernel_size=(3, 3), strides=(2, 2))(x)
#         x = nn.relu(x)
#         x = nn.Conv(features=127, kernel_size=(3, 3), strides=(2, 2))(x)
#         x = nn.relu(x)
#         x = x.reshape((x.shape[-1], -1)) # Flatten
#         x = nn.Dense(features=self.feature_dim)(x)
#         return x