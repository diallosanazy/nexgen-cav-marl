import jax.numpy as jnp

def calculate_pdr(adj_matrix, success_mask):
    """
    PDR = (Delivered Packets) / (Total Attempted Packets)
    Excluding the diagonal (self-communication).
    """
    # Create a mask to ignore the diagonal (self-to-self)
    mask_no_diag = jnp.ones_like(adj_matrix) - jnp.eye(adj_matrix.shape[0])
    
    # Intended links in our ring/chord topology (minus self)
    active_edges = adj_matrix * mask_no_diag
    
    total_intended = jnp.sum(active_edges)
    # Successes on those specific links
    delivered = jnp.sum(success_mask * active_edges)

    return jnp.where(total_intended > 0, delivered / total_intended, 1.0)

def calculate_lvr(packet_ages, adj_matrix, threshold=0.1):
    """
    Measures % of active links that haven't received an update within 'threshold'.
    """
    # Only look at the ages for links that are part of our 8-agent topology
    violations = (packet_ages > threshold).astype(jnp.float32)
    active_violations = violations * adj_matrix
    
    total_links = jnp.sum(adj_matrix)

    return jnp.where(total_links > 0, jnp.sum(active_violations) / total_links, 0.0)

def calculate_throughput(delivered_count, total_time):
    return delivered_count / total_time

def calculate_marl_reward(pdr, lvr, collision_occurred=False):
    """
    Research-aligned reward function.
    Goal: Maximize PDR, Minimize LVR violations.
    """
    # Base reward for high PDR
    reward = pdr * 1.0 
    
    # Penalty for LVR violation (LVR > 10%)
    # This directly targets your research constraint
    lvr_penalty = jnp.where(lvr > 0.10, -2.0, 0.0)
    
    # Heavy penalty for physical collisions
    collision_penalty = -10.0 if collision_occurred else 0.0
    
    return reward + lvr_penalty + collision_penalty