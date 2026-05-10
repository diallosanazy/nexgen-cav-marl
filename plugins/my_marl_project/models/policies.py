import jax
import jax.numpy as jnp
import flax.linen as nn

class ActorCriticHead(nn.Module):
    action_dim: int = 3 

    @nn.compact
    def __call__(self, local_features, global_state=None):
        # ACTOR: Shared parameters across all 8 agents
        raw_actions = nn.Dense(self.action_dim)(local_features)
        actions = jnp.concatenate([
            jnp.tanh(raw_actions[..., 0:1]),    # Steer
            nn.sigmoid(raw_actions[..., 1:3])   # Throttle/Brake
        ], axis=-1)

        # Attention for packet routing
        attention_weights = nn.softmax(nn.Dense(8)(local_features), axis=-1)

        # CRITIC: Centralized (sees all agents if global_state is provided)
        # In MAPPO, the value function is centralized during training.
        critic_input = global_state if global_state is not None else local_features
        value = nn.Dense(1)(critic_input)

        return actions, attention_weights, value

# class ActorCriticHead(nn.Module):
#     """
#     CTDE, head for MAPPO/MARL-PPO
#     """
#     action_dim: int = 3 # Steer, Throttle, Brake
#     num_agents: int = 8 # The Fleet

#     @nn.compact
#     def __call__(self, x):
#         # Shared feature processing
#         x = nn.Dense(128)(x)
#         x = nn.relu(x)

#         # Decentralized execution, driving head
#         driving_logits = nn.Dense(self.action_dim)(x)
#         steer = jnp.tanh(driving_logits[..., 0])
#         throttle = nn.sigmoid(driving_logits[..., 1])
#         brake = nn.sigmoid(driving_logits[..., 2])
#         driving_actions = jnp.stack([steer, throttle, brake], axis=-1)

#         # Routing/Attention Head (Packet Prio)
#         attention_logits = nn.Dense(self.num_agents)(x)
#         attention_weights = nn.softmax(attention_logits)

#         # Critic (value function for ppo)
#         critic_value = nn.Dense(1)(x)

#         return driving_actions, attention_weights, critic_value

