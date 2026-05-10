import os
import jax
import jax.numpy as jnp
# from alpasim.interfaces import Driver
from alpasim_driver.models.manual_model import ManualModel
# from ..models.jax_encoder import VisionEncoder
from ..models.jax_encoder import VectorEncoder
from ..models.policies import ActorCriticHead
from ..utils.metrics import calculate_lvr, calculate_pdr

class CommDriver(ManualModel):
    def __init__(self, **kwargs):
        kwargs.pop("_target_", None)
        kwargs.pop("_recursive_", None)
        kwargs.pop("_convert_", None)

        super().__init__(**kwargs)
        
        # Pull the LOSS_RATE from the environment (defaulting to 0.15)
        # Your research question tests up to 0.25 (25%)
        self.loss_rate = float(os.environ.get("LOSS_RATE", 0.15))
        print(f"🚀 JAX CommDriver initialized. Testing Loss Rate: {self.loss_rate}")
        
        self.key = jax.random.PRNGKey(48)
        self.encoder = VectorEncoder()
        self.policy = ActorCriticHead()

        # Initialize JAX params with an 8-agent batch
        dummy_state = jnp.ones((8, 10)) 
        self.key, enc_key, pol_key = jax.random.split(self.key, 3)
        enc_vars = self.encoder.init(enc_key, dummy_state)
        
        # 256 matches the 'features' in jax_encoder.py
        dummy_features = jnp.ones((8, 256))
        pol_vars = self.policy.init(pol_key, dummy_features)

        self.params = {
            'encoder': enc_vars['params'],
            'policy': pol_vars['params']
        }

        self.adj_matrix = self.init_topology()
        self.packet_ages = jnp.zeros((8, 8))

    def init_topology(self):
        """Standard 8-agent Ring + Chord topology."""
        adj = jnp.zeros((8, 8))
        edges = [(0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (5, 6), (6, 7), (7, 0),
                 (0, 4), (1, 5), (2, 6), (3, 7)]
        for u, v in edges:
            adj = adj.at[u, v].set(1)
            adj = adj.at[v, u].set(1)
        return adj

    def __call__(self, obs):
        observations = obs if isinstance(obs, list) else [obs]
        return self.predict(observations)

    def predict(self, observations):
        self.key, subkey_comm = jax.random.split(self.key)

        # 1. Parse ego states for all 8 agents
        state_matrix = jnp.array([o.get('ego_state', jnp.zeros(10)) for o in observations])

        # 2. Shared Parameter Encoder (Decentralized execution)
        local_features = self.encoder.apply({'params': self.params['encoder']}, state_matrix)

        # 3. Policy Execution (Actions + Attention for Routing)
        # We pass local_features twice: once for actor, once for decentralized critic
        driving_actions, attention_weights, _ = jax.vmap(
            lambda f: self.policy.apply({'params': self.params['policy']}, f, f)
        )(local_features)

        # --- RESEARCH LOGIC START ---
        # 4. Stochastic Communication Loss (The "Adversary")
        # We use a Bernoulli mask to simulate the link loss up to 25%
        comm_mask = jax.random.bernoulli(subkey_comm, 1 - self.loss_rate, (8, 8))
        
        # 5. Success Mask: Only succeed if the topology has an edge AND the link survived
        success_mask = self.adj_matrix * comm_mask
        success_mask = jnp.fill_diagonal(success_mask, 1.0, inplace=False) # Self-data is 100%
        # --- RESEARCH LOGIC END ---

        # 6. Feature Sharing via Learned Routing
        effective_routing = success_mask * attention_weights
        shared_features = jnp.matmul(effective_routing, local_features)
        
        # 7. Final control decision using shared context
        # Second pass: Policy now uses context from neighbors
        final_actions, _, _ = jax.vmap(
            lambda f: self.policy.apply({'params': self.params['policy']}, f, f)
        )(shared_features)
    
        # 8. Update Metrics
        # dt=0.1 assuming 10Hz simulation tick
        self.packet_ages = jnp.where(success_mask, 0.0, self.packet_ages + 0.1)
        pdr = calculate_pdr(self.adj_matrix, success_mask)
        lvr = calculate_lvr(self.packet_ages, self.adj_matrix)
        
        if self.step_count % 50 == 0:
            print(f"📡 Step {self.step_count} | PDR: {pdr:.2f} | LVR: {lvr:.2f}")

        return [
            {"steer": float(a[0]), "throttle": float(a[1]), "brake": float(a[2])}
            for a in final_actions
        ]

    @property
    def step_count(self):
        return getattr(self, '_step_count', 0)


# class CommDriver(Driver):
#     def __init__(self, loss_rate=0.15, **kwargs):
#         # self.host = kwargs.get("host", "localhost")
#         kwargs.pop("_target_", None)
#         kwargs.pop("_recursive_", None)
#         kwargs.pop("_convert_", None)

#         super().__init__(**kwargs)
#         self.loss_rate = float(os.environ.get("LOSS_RATE", 0.15))
#         print(f"Communication Driver init with loss_rate: {self.loss_rate}")
#         self.key = jax.random.key(48)

#         # init vision encoder and model
#         self.encoder = VisionEncoder()
#         self.policy = ActorCriticHead()
#         # use dummy observation to start the engine
#         # remember shape: (batch, dim, dim, color_channels)
#         dummy_obs = jnp.ones((1, 128, 128, 3))
#         enc_vars = self.encoder.init(self.key, dummy_obs)['params']
#         dummy_features = jnp.ones((1, 256))
#         pol_vars = self.policy.init(self.key, dummy_features)

#         self.params = {
#             'encoder': enc_vars['params'],
#             'policy': pol_vars['params']
#         } 

#         # set up network topology (ring + chord) and packet ages for lvr
#         self.adj_matrix = self.init_topology()
#         self.packet_ages = jnp.zeros((8, 8))

#     def init_topology(self):
#         adj = jnp.zeros((8, 8))
#         edges = [(0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (5, 6), (6, 7), (7, 0) # The ring
#                  (0, 4), (1, 5), (2, 6), (3, 7)]                                # The chords
#         for u, v in edges:
#             adj = adj.at[u, v].set(1)
#             adj = adj.at[v, u].set(1)
#         return adj
    
#     def update_latency(self, success_mask, dt=0.1):
#         """Update persistent packet ages"""
#         self.packet_ages = jnp.where(success_mask, 0.0, self.packet_ages + dt)
#         return self.packet_ages

#     def predict(self, observations):
#         self.key, subkey_comm, subkey_pol = jax.random.split(self.key, 3)

#         # Extract pixel data
#         pixel_array = jnp.stack([obs['front_camera'] for obs in observations])
#         pixel_array = pixel_array.astype(jnp.float32) / 255.0
#         local_features = self.encoder.apply(
#             {'params': self.params['encoder']}, 
#             pixel_array
#         )

#         # Decide on local features first
#         driving_actions, attention_weights, _ = jax.vmap(
#             lambda f: self.policy.apply({'params': self.params['policy']}, f)
#         )(local_features)

#         # Phase 2, Stochastic Communication loss
#         comm_mask = jax.random.bernoulli(subkey_comm, 1 - self.loss_rate, (8, 8))
#         success_mask = self.adj_matrix * comm_mask
#         # Diagonals as 1 because agents will always have easy access to their own data
#         success_mask = jnp.fill_diagonal(success_mask, 1.0, inplace=False)

#         # Learned routing logic
#         # Connecting network with the "physical" env
#         effective_routing = success_mask * attention_weights
#         # Multiply features across nodes (8, 8) * (8, 256) -> (8, 256)
#         shared_features = jnp.matmul(effective_routing, local_features)
        
#         # Decision but now with neighbor vehicle data
#         # A second pass if you will
#         driving_actions, _, _ = jax.vmap(
#             lambda f: self.policy.apply({'params': self.params['policy']}, f)
#         )(shared_features)
    
#         # update packet delivery rate and latency violation rate
#         self.update_latency(success_mask)
#         pdr_score = calculate_pdr(self.adj_matrix, success_mask)
#         lvr_score = calculate_lvr(self.packet_ages, self.adj_matrix)    
        
#         # DEBUG PRINT: SEE STATS
#         if jax.process_index() == 0:
#             print(f"DEBUG | PDR: {pdr:.2f} | LVR: {lvr:.2f} | Mean Age: {jnp.mean(self.packet_ages):.2f}s")

#         return [
#             {
#                 "steer": float(a[0]),
#                 "throttle": float(a[1]),
#                 "brake": float(a[2])
#             }
#             for a in driving_actions # all 8 agents
#         ] 

#     def update_latency(self, success_mask, dt=0.1):
#         """
#         success_mask: (8, 8) bool matrix
#         """
#         # age = 0 on success, add dt for failure
#         self.packet_ages = jnp.where(success_mask, 0.0, self.packet_ages + dt)

#         # Calc LVR
#         current_lvr = calculate_lvr(self.packet_ages, threshold=0.1)
#         return current_lvr

