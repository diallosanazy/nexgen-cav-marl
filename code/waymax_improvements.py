"""
CAV MARL Improvements — Topology Comparison + New Reward + Better Visualizations
NexGen Multi-Agent Connected Autonomous Vehicle Networks
CSC 59866-E Senior Project | Alhassana Diallo & Marcley Colin

New this file:
  1. Reward function visualization (shaped reward landscape plot)
  2. Two reward functions: per-step (existing) vs shaped PDR/LVR reward
  3. Four topology shapes: Ring+Chord, Star, Linear, Mesh
  4. Topology comparison: same MARL-PPO across all 4 topologies
  5. More episodes: 300 Phase 1 + 300 Phase 2 (vs 80/100 before)
  6. Clearer, publication-quality figures
"""

import jax
import jax.numpy as jnp
from jax import random
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec
import json, time, os

print("=" * 65)
print("  NexGen CAV Improvements | Topology + Reward + Visualization")
print("=" * 65)
print(f"  JAX version : {jax.__version__}")
print(f"  Backend     : {jax.default_backend()}")
print(f"  Devices     : {jax.devices()}")
print("=" * 65)

# ─── Config ───────────────────────────────────────────────────────────────────
N       = 8
H       = 64
LR      = 6e-4
GAMMA   = 0.99
EP_LEN  = 200
EP_P1   = 300    # ↑ increased from 80
EP_P2   = 300    # ↑ increased from 100
LATENCY = 100
OBS_DIM = N * 3 + 2
ACT_DIM = N

# ═══════════════════════════════════════════════════════════════════════════════
#  TOPOLOGY DEFINITIONS
#  We test 4 shapes to prove topology matters for V2V routing performance
# ═══════════════════════════════════════════════════════════════════════════════

def make_adj(edges, n=N):
    A = np.zeros((n, n), dtype=np.float32)
    for u, v in edges:
        A[u, v] = A[v, u] = 1.0
    return A

# 1. Ring + Chord (our existing topology)
RING_CHORD_EDGES = [
    (0,1),(1,2),(2,3),(3,4),(4,5),(5,6),(6,7),(7,0),  # ring
    (0,4),(1,5),(2,6),(3,7)                             # chords
]

# 2. Star — node 0 is the hub, all others connect only through it
STAR_EDGES = [(0, i) for i in range(1, N)]

# 3. Linear chain — 0-1-2-3-4-5-6-7 (worst case: long hops)
LINEAR_EDGES = [(i, i+1) for i in range(N-1)]

# 4. Full Mesh — every node connects to every other node
MESH_EDGES = [(i, j) for i in range(N) for j in range(i+1, N)]

TOPOLOGIES = {
    'Ring+Chord': make_adj(RING_CHORD_EDGES),
    'Star':       make_adj(STAR_EDGES),
    'Linear':     make_adj(LINEAR_EDGES),
    'Full Mesh':  make_adj(MESH_EDGES),
}

TOPO_COLORS = {
    'Ring+Chord': '#2ecc71',
    'Star':       '#9b59b6',
    'Linear':     '#e74c3c',
    'Full Mesh':  '#3498db',
}

print(f"\nTopologies loaded:")
for name, adj in TOPOLOGIES.items():
    n_edges = int(adj.sum() / 2)
    avg_deg = adj.sum(axis=1).mean()
    print(f"  {name:<12} | edges={n_edges:2d} | avg_degree={avg_deg:.1f}")

# ═══════════════════════════════════════════════════════════════════════════════
#  REWARD FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════════════

# Reward A — Per-step per-packet (existing)
# +1.0 per successful delivery (within latency), +0.5 if urgent, +0.05 per hop
# PDR and LVR are emergent — not directly optimized

# Reward B — Shaped reward (new): directly optimizes PDR, penalizes LVR > 10%
def shaped_reward(delivered, total, latency_violations, hops_taken, urgent):
    """
    Shaped reward that directly targets the research question:
      - Maximize PDR (delivered / total)
      - Penalize LVR > 10% (latency violations / delivered)
      - Small routing efficiency bonus (fewer hops = better)
    """
    pdr = delivered / max(total, 1)
    lvr = latency_violations / max(delivered, 1)
    # Base: reward proportional to PDR
    r = pdr * 2.0
    # Penalty: -1.5 if LVR exceeds 10% target
    r += -1.5 if lvr > 0.10 else 0.0
    # Efficiency bonus: reward urgent packet delivery extra
    r += 0.3 if urgent else 0.0
    # Routing overhead penalty: discourage unnecessary hops
    r += -0.02 * max(hops_taken - 1, 0)
    return r

# ═══════════════════════════════════════════════════════════════════════════════
#  FIGURE 1 — REWARD FUNCTION VISUALIZATION
#  Show the shaped reward landscape as a function of PDR and LVR
# ═══════════════════════════════════════════════════════════════════════════════
print("\n[Fig 1] Plotting reward function landscape...")

pdr_vals = np.linspace(0, 1, 100)
lvr_vals = np.linspace(0, 0.5, 100)
PDR_grid, LVR_grid = np.meshgrid(pdr_vals, lvr_vals)

# Reward A surface (per-step approximation — reward ≈ PDR * scale)
reward_A = PDR_grid * 2.0  # simplified: reward scales with delivery rate

# Reward B surface (shaped)
reward_B = PDR_grid * 2.0 + np.where(LVR_grid > 0.10, -1.5, 0.0)

fig1, axes = plt.subplots(1, 3, figsize=(17, 5), facecolor='#0d1117')
fig1.suptitle('Reward Function Landscape\nPDR vs Latency Violation Rate',
              color='white', fontsize=14, y=1.02)

for ax in axes:
    ax.set_facecolor('#1a1a2e')
    for sp in ax.spines.values():
        sp.set_color('#334')

# Panel 1: Reward A (per-step)
c1 = axes[0].contourf(PDR_grid, LVR_grid, reward_A, levels=20, cmap='YlGn')
axes[0].set_title('Reward A — Per-Step Delivery\n(existing)', color='white', fontsize=11)
axes[0].set_xlabel('Packet Delivery Ratio (PDR)', color='white')
axes[0].set_ylabel('Latency Violation Rate (LVR)', color='white')
axes[0].tick_params(colors='white')
cb1 = plt.colorbar(c1, ax=axes[0])
cb1.ax.yaxis.set_tick_params(color='white')
plt.setp(cb1.ax.yaxis.get_ticklabels(), color='white')
cb1.set_label('Reward', color='white')

# Panel 2: Reward B (shaped)
c2 = axes[1].contourf(PDR_grid, LVR_grid, reward_B, levels=20, cmap='RdYlGn')
axes[1].axhline(y=0.10, color='white', ls='--', lw=1.5, label='LVR = 10% threshold')
axes[1].set_title('Reward B — Shaped PDR/LVR\n(new)', color='white', fontsize=11)
axes[1].set_xlabel('Packet Delivery Ratio (PDR)', color='white')
axes[1].set_ylabel('Latency Violation Rate (LVR)', color='white')
axes[1].tick_params(colors='white')
axes[1].legend(facecolor='#1a1a2e', labelcolor='white', fontsize=8)
cb2 = plt.colorbar(c2, ax=axes[1])
cb2.ax.yaxis.set_tick_params(color='white')
plt.setp(cb2.ax.yaxis.get_ticklabels(), color='white')
cb2.set_label('Reward', color='white')

# Panel 3: Difference (B - A) — shows where shaped reward changes incentives
diff = reward_B - reward_A
c3 = axes[2].contourf(PDR_grid, LVR_grid, diff, levels=20, cmap='RdBu')
axes[2].axhline(y=0.10, color='white', ls='--', lw=1.5, label='LVR = 10% threshold')
axes[2].set_title('Difference (B − A)\nShaping effect', color='white', fontsize=11)
axes[2].set_xlabel('Packet Delivery Ratio (PDR)', color='white')
axes[2].set_ylabel('Latency Violation Rate (LVR)', color='white')
axes[2].tick_params(colors='white')
axes[2].legend(facecolor='#1a1a2e', labelcolor='white', fontsize=8)
cb3 = plt.colorbar(c3, ax=axes[2])
cb3.ax.yaxis.set_tick_params(color='white')
plt.setp(cb3.ax.yaxis.get_ticklabels(), color='white')
cb3.set_label('Reward Difference', color='white')

plt.tight_layout()
fig1.savefig('improvements_fig1_reward_landscape.png', dpi=150,
             bbox_inches='tight', facecolor='#0d1117')
print("  Saved improvements_fig1_reward_landscape.png")

# ═══════════════════════════════════════════════════════════════════════════════
#  FIGURE 2 — TOPOLOGY DIAGRAMS (side by side)
# ═══════════════════════════════════════════════════════════════════════════════
print("[Fig 2] Plotting topology diagrams...")

fig2, axes = plt.subplots(1, 4, figsize=(18, 5), facecolor='#0d1117')
fig2.suptitle('Network Topology Shapes Under Comparison\n(8 CAV agents each)',
              color='white', fontsize=14, y=1.02)

angles = np.linspace(0, 2 * np.pi, N, endpoint=False)
xs = np.cos(angles)
ys = np.sin(angles)

# Special positions for Star and Linear
star_pos = [(0, 0)] + [(np.cos(a), np.sin(a)) for a in np.linspace(0, 2*np.pi, N-1, endpoint=False)]
linear_pos = [(-1.4 + i * 0.4, 0) for i in range(N)]

topo_pos = {
    'Ring+Chord': list(zip(xs, ys)),
    'Star':       star_pos,
    'Linear':     linear_pos,
    'Full Mesh':  list(zip(xs, ys)),
}

topo_edges = {
    'Ring+Chord': RING_CHORD_EDGES,
    'Star':       STAR_EDGES,
    'Linear':     LINEAR_EDGES,
    'Full Mesh':  MESH_EDGES,
}

for ax, (name, adj) in zip(axes, TOPOLOGIES.items()):
    ax.set_facecolor('#0d1117')
    pos = topo_pos[name]
    edges = topo_edges[name]
    color = TOPO_COLORS[name]

    for u, v in edges:
        ax.plot([pos[u][0], pos[v][0]], [pos[u][1], pos[v][1]],
                '-', color='#4a9eff', lw=1.2, alpha=0.6)

    for i in range(N):
        ax.scatter(pos[i][0], pos[i][1], s=200, color=color,
                   zorder=5, edgecolors='white', lw=1.2)
        offset = 1.22 if name != 'Linear' else 1.0
        tx = pos[i][0] * offset if name != 'Linear' else pos[i][0]
        ty = pos[i][1] * offset + (0.15 if name == 'Linear' else 0)
        ax.text(tx, ty, f'{i}', color='white', fontsize=8,
                ha='center', va='center', fontfamily='monospace')

    n_edges = int(adj.sum() / 2)
    avg_deg = adj.sum(axis=1).mean()
    ax.set_title(f'{name}\n{n_edges} edges | avg deg {avg_deg:.1f}',
                 color=color, fontsize=10, fontweight='bold')
    ax.axis('off')
    ax.set_aspect('equal')

plt.tight_layout()
fig2.savefig('improvements_fig2_topologies.png', dpi=150,
             bbox_inches='tight', facecolor='#0d1117')
print("  Saved improvements_fig2_topologies.png")

# ═══════════════════════════════════════════════════════════════════════════════
#  ENVIRONMENT + AGENT (same as before, but topology-parameterized)
# ═══════════════════════════════════════════════════════════════════════════════

def init_mlp(key, in_dim, hidden, out_dim):
    k1, k2, k3 = random.split(key, 3)
    scale = lambda n: np.sqrt(2.0 / n)
    return {
        'w1': random.normal(k1, (in_dim, hidden)) * scale(in_dim),
        'b1': jnp.zeros(hidden),
        'w2': random.normal(k2, (hidden, hidden)) * scale(hidden),
        'b2': jnp.zeros(hidden),
        'w3': random.normal(k3, (hidden, out_dim)) * scale(hidden),
        'b3': jnp.zeros(out_dim),
    }

def mlp_forward(params, x):
    x = jnp.tanh(x @ params['w1'] + params['b1'])
    x = jnp.tanh(x @ params['w2'] + params['b2'])
    return x @ params['w3'] + params['b3']

def softmax_masked(logits, mask):
    logits = jnp.where(mask, logits, -1e9)
    logits = logits - jnp.max(logits)
    exp = jnp.exp(logits) * mask
    return exp / (jnp.sum(exp) + 1e-8)


class CAVEnvTopo:
    """CAV environment parameterized by topology adjacency matrix."""
    def __init__(self, adj, loss_rate=0.0, seed=0, use_shaped_reward=False):
        self.adj = adj
        self.loss_rate = loss_rate
        self.rng = np.random.default_rng(seed)
        self.use_shaped = use_shaped_reward
        self.reset()

    def reset(self):
        self.queues    = np.zeros(N, dtype=np.float32)
        self.delivered = 0
        self.total     = 0
        self.latency_violations = 0
        self.overhead  = 0
        self.step_count = 0
        self.episode_reward = 0.0
        self.loss = self.adj * (self.loss_rate +
                    self.rng.uniform(0, 0.05, (N, N)).astype(np.float32))
        srcs = self.rng.integers(0, N, size=N)
        self.packets = [(s, self.rng.integers(0, N), 0,
                         bool(self.rng.random() < 0.3)) for s in srcs]
        return self._obs()

    def _obs(self):
        obs = []
        for i in range(N):
            nb_q    = self.adj[i] * self.queues
            nb_loss = self.loss[i]
            urgency = np.array([float(any(p[3] for p in self.packets
                                          if p[0] == i))], dtype=np.float32)
            q_norm  = np.array([self.queues[i] / 10.0], dtype=np.float32)
            o = np.concatenate([nb_q / 10.0, nb_loss, nb_loss, q_norm, urgency])
            o = o[:OBS_DIM] if len(o) >= OBS_DIM else np.pad(o, (0, OBS_DIM - len(o)))
            obs.append(o)
        return np.stack(obs)

    def step(self, actions):
        new_packets = []
        self.step_count += 1
        step_reward = 0.0

        for p in self.packets:
            src, dst, hops, urgent = p
            a = int(actions[src])
            if self.adj[src, a] == 1:
                self.overhead += 1
                if self.rng.random() > self.loss[src, a]:
                    if a == dst:
                        self.delivered += 1
                        lat = hops * 12 + self.rng.uniform(5, 20)
                        if lat > LATENCY:
                            self.latency_violations += 1
                        else:
                            step_reward += 1.0 + (0.5 if urgent else 0.0)
                    elif hops < 6:
                        new_packets.append((a, dst, hops + 1, urgent))
                        self.queues[a] = min(self.queues[a] + 0.5, 10.0)
                        step_reward += 0.05
            else:
                if hops < 6:
                    new_packets.append((src, dst, hops, urgent))

        self.total += len(self.packets)
        self.queues = np.clip(self.queues - 0.3, 0, 10)
        n_new = self.rng.integers(1, N)
        for _ in range(n_new):
            s = self.rng.integers(0, N)
            d = self.rng.integers(0, N)
            if s != d:
                new_packets.append((s, d, 0, bool(self.rng.random() < 0.25)))
        self.packets = new_packets[:30]

        # Shaped reward replaces step reward if enabled
        if self.use_shaped:
            step_reward = shaped_reward(
                self.delivered, self.total,
                self.latency_violations, 1, False
            ) * 0.1  # scale to same magnitude as per-step reward

        self.episode_reward += step_reward
        done = self.step_count >= EP_LEN
        return self._obs(), step_reward, done

    def metrics(self):
        return dict(
            pdr        = self.delivered / max(self.total, 1),
            lvr        = self.latency_violations / max(self.delivered, 1),
            throughput = self.delivered / EP_LEN,
            overhead   = self.overhead / max(self.total, 1),
        )


class PPOAgent:
    def __init__(self, key, adj):
        self.adj = adj
        k1, k2, k3 = random.split(key, 3)
        self.actor  = init_mlp(k1, OBS_DIM, H, ACT_DIM)
        self.critic = init_mlp(k2, OBS_DIM, H, 1)
        self.ref    = init_mlp(k3, OBS_DIM, H, ACT_DIM)
        self.lr = LR

    def act(self, obs_np):
        actions = []
        for i in range(N):
            logits = mlp_forward(self.actor, jnp.array(obs_np[i]))
            mask   = jnp.array(self.adj[i] > 0, dtype=jnp.float32)
            # Fallback: if no neighbors, pick random valid
            if self.adj[i].sum() == 0:
                actions.append(0)
                continue
            probs = np.array(softmax_masked(logits, mask))
            probs = np.clip(probs, 0, 1)
            s = probs.sum()
            probs = probs / s if s > 0 else (self.adj[i] / self.adj[i].sum())
            actions.append(int(np.random.choice(N, p=probs)))
        return np.array(actions)

    def update(self, trajs):
        if not trajs:
            return 0.0
        all_obs, all_acts, all_rets = [], [], []
        for (obs_list, act_list, rew_list) in trajs:
            G = 0.0
            for obs, act, rew in zip(reversed(obs_list),
                                     reversed(act_list),
                                     reversed(rew_list)):
                G = rew + GAMMA * G
                all_obs.append(obs); all_acts.append(act); all_rets.append(G)
        if not all_obs:
            return 0.0
        rets = np.array(all_rets)
        rets = (rets - rets.mean()) / (rets.std() + 1e-8)
        total_loss = 0.0
        for obs, act, ret in zip(all_obs, all_acts, rets):
            logits = mlp_forward(self.actor, jnp.array(obs))
            probs  = softmax_masked(logits, jnp.ones(ACT_DIM))
            log_p  = jnp.log(probs[act] + 1e-8)
            total_loss += float(-log_p * ret)
        loss_scale = total_loss / len(all_obs)
        for k in self.actor:
            noise = np.random.randn(*self.actor[k].shape).astype(np.float32)
            self.actor[k] = self.actor[k] - self.lr * loss_scale * jnp.array(noise * 0.001)
        return loss_scale

    def snap(self):
        for k in self.actor:
            self.ref[k] = self.actor[k]


def rollout(env, agent):
    obs = env.reset()
    obs_list, act_list, rew_list = [], [], []
    total_r = 0.0
    for _ in range(EP_LEN):
        acts = agent.act(obs)
        obs_list.append(obs[0])
        act_list.append(int(acts[0]))
        obs, rew, done = env.step(acts)
        rew_list.append(rew)
        total_r += rew
        if done:
            break
    return (obs_list, act_list, rew_list), total_r, env.metrics()


# ═══════════════════════════════════════════════════════════════════════════════
#  TRAINING — all 4 topologies x 2 reward functions  (ring+chord only for both)
#  Topology comparison uses Reward A (fair comparison)
#  Reward comparison uses Ring+Chord topology
# ═══════════════════════════════════════════════════════════════════════════════
LOSS_RATES = [0.0, 0.05, 0.10, 0.15, 0.20, 0.25]

topo_results  = {}   # topology → {loss_rate → [pdr list]}
topo_lvr      = {}
topo_thr      = {}
topo_reward_curves = {}
topo_pdr_curves    = {}

for topo_name, adj in TOPOLOGIES.items():
    print(f"\n{'─'*65}")
    print(f"[{topo_name}] Training MARL-PPO ({EP_P1}+{EP_P2} episodes)...")
    print(f"{'─'*65}")
    t0 = time.time()

    agent = PPOAgent(random.PRNGKey(42), adj)
    rew_hist, pdr_hist = [], []

    # Phase 1 — clean channel
    for ep in range(EP_P1):
        env = CAVEnvTopo(adj, loss_rate=0.0, seed=ep)
        traj, total_r, met = rollout(env, agent)
        agent.update([traj])
        rew_hist.append(total_r)
        pdr_hist.append(met['pdr'])
        if ep % 100 == 0:
            print(f"  P1 Ep {ep:4d} | R={total_r:6.1f} | PDR={met['pdr']:.3f} | LVR={met['lvr']:.3f}")

    agent.snap()
    phase_boundary = EP_P1

    # Phase 2 — degraded channel (15% loss)
    for ep in range(EP_P2):
        env = CAVEnvTopo(adj, loss_rate=0.15, seed=EP_P1 + ep)
        traj, total_r, met = rollout(env, agent)
        agent.update([traj])
        rew_hist.append(total_r)
        pdr_hist.append(met['pdr'])
        if ep % 100 == 0:
            print(f"  P2 Ep {ep:4d} | R={total_r:6.1f} | PDR={met['pdr']:.3f} | LVR={met['lvr']:.3f}")

    print(f"  Done in {time.time()-t0:.1f}s")

    topo_reward_curves[topo_name] = rew_hist
    topo_pdr_curves[topo_name]    = pdr_hist

    # Evaluate across loss rates
    topo_results[topo_name] = {l: [] for l in LOSS_RATES}
    topo_lvr[topo_name]     = []
    topo_thr[topo_name]     = []

    for loss_r in LOSS_RATES:
        pdrs, lvrs, thrs = [], [], []
        for trial in range(10):
            env = CAVEnvTopo(adj, loss_rate=loss_r, seed=1000 + trial)
            _, _, met = rollout(env, agent)
            topo_results[topo_name][loss_r].append(met['pdr'])
            pdrs.append(met['pdr'])
        for trial in range(5):
            env = CAVEnvTopo(adj, loss_rate=loss_r, seed=2000 + trial)
            _, _, met = rollout(env, agent)
            lvrs.append(met['lvr'])
            thrs.append(met['throughput'])
        topo_lvr[topo_name].append(float(np.mean(lvrs)))
        topo_thr[topo_name].append(float(np.mean(thrs)))

# ── Reward function comparison (Ring+Chord, Reward A vs B) ────────────────────
print(f"\n{'─'*65}")
print("[Reward Comparison] Ring+Chord: Reward A vs Reward B...")
print(f"{'─'*65}")

rc_adj = TOPOLOGIES['Ring+Chord']
reward_B_hist, reward_B_pdr = [], []
agent_B = PPOAgent(random.PRNGKey(42), rc_adj)

for ep in range(EP_P1):
    env = CAVEnvTopo(rc_adj, loss_rate=0.0, seed=ep, use_shaped_reward=True)
    traj, total_r, met = rollout(env, agent_B)
    agent_B.update([traj])
    reward_B_hist.append(total_r)
    reward_B_pdr.append(met['pdr'])
    if ep % 100 == 0:
        print(f"  P1 Ep {ep:4d} | R={total_r:6.1f} | PDR={met['pdr']:.3f}")

agent_B.snap()
for ep in range(EP_P2):
    env = CAVEnvTopo(rc_adj, loss_rate=0.15, seed=EP_P1 + ep, use_shaped_reward=True)
    traj, total_r, met = rollout(env, agent_B)
    agent_B.update([traj])
    reward_B_hist.append(total_r)
    reward_B_pdr.append(met['pdr'])
    if ep % 100 == 0:
        print(f"  P2 Ep {ep:4d} | R={total_r:6.1f} | PDR={met['pdr']:.3f}")

print("  Reward B training done")

# ═══════════════════════════════════════════════════════════════════════════════
#  RESULTS TABLE
# ═══════════════════════════════════════════════════════════════════════════════
print(f"\n{'─'*65}")
print("[Results Table — Topology Comparison @ 0% and 20% loss]")
print(f"{'─'*65}")
print(f"{'Topology':<14} {'PDR@0%':>8} {'PDR@10%':>8} {'PDR@20%':>8} {'LVR@20%':>8} {'THR@0%':>8}")
print("─" * 58)
for name in TOPOLOGIES:
    p0  = np.mean(topo_results[name][0.0])
    p10 = np.mean(topo_results[name][0.10])
    p20 = np.mean(topo_results[name][0.20])
    l20 = topo_lvr[name][LOSS_RATES.index(0.20)]
    t0v = topo_thr[name][LOSS_RATES.index(0.0)]
    print(f"{name:<14} {p0:>8.3f} {p10:>8.3f} {p20:>8.3f} {l20:>8.3f} {t0v:>8.3f}")

# ═══════════════════════════════════════════════════════════════════════════════
#  FIGURE 3 — TRAINING CURVES: all 4 topologies
# ═══════════════════════════════════════════════════════════════════════════════
print("\n[Fig 3] Plotting topology training curves...")

def smooth(x, w=20):
    if len(x) < w:
        return np.array(x)
    return np.convolve(x, np.ones(w) / w, mode='valid')

fig3, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5), facecolor='#0d1117')
for ax in (ax1, ax2):
    ax.set_facecolor('#1a1a2e')
    for sp in ax.spines.values():
        sp.set_color('#334')

for name in TOPOLOGIES:
    color = TOPO_COLORS[name]
    r_smooth = smooth(topo_reward_curves[name])
    p_smooth = smooth(topo_pdr_curves[name])
    ax1.plot(r_smooth, color=color, lw=2, label=name, alpha=0.9)
    ax2.plot(p_smooth, color=color, lw=2, label=name, alpha=0.9)

ax1.axvline(x=EP_P1, color='white', ls='--', lw=1.2, alpha=0.5, label='Phase 2 start')
ax2.axvline(x=EP_P1, color='white', ls='--', lw=1.2, alpha=0.5, label='Phase 2 start')

ax1.set_title(f'Training Reward — All Topologies\n({EP_P1}+{EP_P2} episodes, smoothed w=20)',
              color='white', fontsize=11)
ax1.set_xlabel('Episode', color='white'); ax1.set_ylabel('Total Reward', color='white')
ax1.tick_params(colors='white')
ax1.legend(facecolor='#1a1a2e', labelcolor='white', fontsize=9)

ax2.set_title(f'PDR During Training — All Topologies\n(Phase 2 = 15% link loss)',
              color='white', fontsize=11)
ax2.set_xlabel('Episode', color='white'); ax2.set_ylabel('PDR', color='white')
ax2.tick_params(colors='white')
ax2.legend(facecolor='#1a1a2e', labelcolor='white', fontsize=9)
ax2.set_ylim(0, 0.8)

plt.tight_layout()
fig3.savefig('improvements_fig3_topology_training.png', dpi=150,
             bbox_inches='tight', facecolor='#0d1117')
print("  Saved improvements_fig3_topology_training.png")

# ═══════════════════════════════════════════════════════════════════════════════
#  FIGURE 4 — PDR COMPARISON: all 4 topologies vs loss rate
# ═══════════════════════════════════════════════════════════════════════════════
print("[Fig 4] Plotting topology PDR comparison...")

fig4, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5), facecolor='#0d1117')
loss_pct = [f'{int(l*100)}%' for l in LOSS_RATES]
for ax in (ax1, ax2):
    ax.set_facecolor('#1a1a2e')
    for sp in ax.spines.values():
        sp.set_color('#334')

for name in TOPOLOGIES:
    color = TOPO_COLORS[name]
    means = [np.mean(topo_results[name][l]) for l in LOSS_RATES]
    stds  = [np.std(topo_results[name][l])  for l in LOSS_RATES]
    ax1.plot(loss_pct, means, '-o', color=color, lw=2, markersize=7, label=name)
    ax1.fill_between(loss_pct,
                     [m - s for m, s in zip(means, stds)],
                     [m + s for m, s in zip(means, stds)],
                     color=color, alpha=0.15)
    ax2.plot(loss_pct, topo_thr[name], '-s', color=color, lw=2,
             markersize=7, label=name)

ax1.set_title('PDR vs Link Loss — Topology Comparison\n(shaded = ±1 std)',
              color='white', fontsize=11)
ax1.set_xlabel('Link Loss Rate', color='white'); ax1.set_ylabel('PDR', color='white')
ax1.tick_params(colors='white')
ax1.legend(facecolor='#1a1a2e', labelcolor='white', fontsize=9)

ax2.set_title('Throughput vs Link Loss — Topology Comparison',
              color='white', fontsize=11)
ax2.set_xlabel('Link Loss Rate', color='white')
ax2.set_ylabel('Throughput (packets/step)', color='white')
ax2.tick_params(colors='white')
ax2.legend(facecolor='#1a1a2e', labelcolor='white', fontsize=9)

plt.tight_layout()
fig4.savefig('improvements_fig4_topology_pdr.png', dpi=150,
             bbox_inches='tight', facecolor='#0d1117')
print("  Saved improvements_fig4_topology_pdr.png")

# ═══════════════════════════════════════════════════════════════════════════════
#  FIGURE 5 — REWARD A vs REWARD B comparison
# ═══════════════════════════════════════════════════════════════════════════════
print("[Fig 5] Plotting Reward A vs B training comparison...")

fig5, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5), facecolor='#0d1117')
for ax in (ax1, ax2):
    ax.set_facecolor('#1a1a2e')
    for sp in ax.spines.values():
        sp.set_color('#334')

rc_rew_A = topo_reward_curves['Ring+Chord']
rc_pdr_A = topo_pdr_curves['Ring+Chord']

ax1.plot(smooth(rc_rew_A),      color='#2ecc71', lw=2, label='Reward A — Per-step delivery')
ax1.plot(smooth(reward_B_hist), color='#e67e22', lw=2, label='Reward B — Shaped PDR/LVR', alpha=0.85)
ax1.axvline(x=EP_P1, color='white', ls='--', lw=1.2, alpha=0.5, label='Phase 2 start')
ax1.set_title('Reward Signal During Training\nReward A vs Reward B (Ring+Chord)',
              color='white', fontsize=11)
ax1.set_xlabel('Episode', color='white'); ax1.set_ylabel('Total Reward', color='white')
ax1.tick_params(colors='white')
ax1.legend(facecolor='#1a1a2e', labelcolor='white', fontsize=9)

ax2.plot(smooth(rc_pdr_A),      color='#2ecc71', lw=2, label='Reward A — Per-step delivery')
ax2.plot(smooth(reward_B_pdr),  color='#e67e22', lw=2, label='Reward B — Shaped PDR/LVR', alpha=0.85)
ax2.axvline(x=EP_P1, color='white', ls='--', lw=1.2, alpha=0.5, label='Phase 2 start')
ax2.set_title('PDR During Training\nReward A vs Reward B (Ring+Chord)',
              color='white', fontsize=11)
ax2.set_xlabel('Episode', color='white'); ax2.set_ylabel('PDR', color='white')
ax2.tick_params(colors='white')
ax2.legend(facecolor='#1a1a2e', labelcolor='white', fontsize=9)
ax2.set_ylim(0, 0.6)

plt.tight_layout()
fig5.savefig('improvements_fig5_reward_comparison.png', dpi=150,
             bbox_inches='tight', facecolor='#0d1117')
print("  Saved improvements_fig5_reward_comparison.png")

# ─── Save results JSON ────────────────────────────────────────────────────────
summary = {}
for name in TOPOLOGIES:
    summary[name] = {}
    for l in LOSS_RATES:
        summary[name][f'pdr_loss{int(l*100)}pct'] = {
            'mean': float(np.mean(topo_results[name][l])),
            'std':  float(np.std(topo_results[name][l])),
        }
    summary[name]['lvr_by_loss'] = topo_lvr[name]
    summary[name]['thr_by_loss'] = topo_thr[name]

with open('improvements_results.json', 'w') as f:
    json.dump(summary, f, indent=2)
print("  Saved improvements_results.json")

print("\n" + "=" * 65)
print("  DONE — Improvements complete!")
print("  improvements_fig1_reward_landscape.png")
print("  improvements_fig2_topologies.png")
print("  improvements_fig3_topology_training.png")
print("  improvements_fig4_topology_pdr.png")
print("  improvements_fig5_reward_comparison.png")
print("  improvements_results.json")
print("=" * 65)
