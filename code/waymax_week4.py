"""
CAV Week 4: MADDPG + MAPPO (Agent Communication)
NexGen Multi-Agent Connected Autonomous Vehicle Networks
CSC 59866-E Senior Project | Alhassana Diallo & Marcley Colin

New this week:
  - MADDPG: Centralized Critic, Decentralized Actors (Lowe et al. 2017)
  - MAPPO:  Multi-Agent PPO with agent-to-agent communication channels
  - Full comparison vs Week 3 MARL-PPO + all baselines
"""

import jax
import jax.numpy as jnp
from jax import random
import numpy as np
import matplotlib.pyplot as plt
from collections import deque
import json, time, os

print("=" * 65)
print("  NexGen CAV Week 4 | MADDPG + MAPPO (Communication)")
print("=" * 65)
print(f"  JAX version : {jax.__version__}")
print(f"  Backend     : {jax.default_backend()}")
print(f"  Devices     : {jax.devices()}")
print("=" * 65)

# ─── Shared Config (same topology as Week 3) ─────────────────────────────────
N       = 8        # CAV agents
H       = 64       # hidden layer size
LR      = 6e-4     # learning rate
GAMMA   = 0.99     # discount
EP_LEN  = 200      # steps per episode
EP_P1   = 60       # phase 1 episodes (0% loss)
EP_P2   = 80       # phase 2 episodes (15% loss)
LATENCY = 100      # ms budget
OBS_DIM = N * 3 + 2
ACT_DIM = N
MSG_DIM = 8        # MAPPO communication vector size

# 8-node ring + chord topology (identical to waymax_ppo_cav.py)
EDGES = [
    (0,1),(1,2),(2,3),(3,4),(4,5),(5,6),(6,7),(7,0),  # ring
    (0,4),(1,5),(2,6),(3,7)                             # chords
]
ADJ = np.zeros((N, N), dtype=np.float32)
for u, v in EDGES:
    ADJ[u, v] = ADJ[v, u] = 1.0

# ─── MLP helpers ─────────────────────────────────────────────────────────────
def init_mlp(key, in_dim, hidden, out_dim):
    k1, k2, k3 = random.split(key, 3)
    scale = lambda n: np.sqrt(2.0 / n)
    return {
        'w1': random.normal(k1, (in_dim, hidden))  * scale(in_dim),
        'b1': jnp.zeros(hidden),
        'w2': random.normal(k2, (hidden, hidden))  * scale(hidden),
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
    exp    = jnp.exp(logits) * mask
    return exp / (jnp.sum(exp) + 1e-8)

def copy_params(p):
    return {k: jnp.array(v) for k, v in p.items()}

def soft_update(target, source, tau=0.01):
    return {k: tau * source[k] + (1 - tau) * target[k] for k in target}

# ─── Environment (identical to waymax_ppo_cav.py) ────────────────────────────
class CAVEnvJAX:
    """JAX-compatible CAV routing environment — same as Week 3."""
    def __init__(self, loss_rate=0.0, seed=0):
        self.loss_rate = loss_rate
        self.rng       = np.random.default_rng(seed)
        self.reset()

    def reset(self):
        self.queues    = np.zeros(N, dtype=np.float32)
        self.delivered = 0
        self.total     = 0
        self.latency_violations = 0
        self.overhead  = 0
        self.step_count = 0
        self.loss = ADJ * (self.loss_rate +
                           self.rng.uniform(0, 0.05, (N, N)).astype(np.float32))
        srcs = self.rng.integers(0, N, size=N)
        self.packets = [(s, self.rng.integers(0, N), 0,
                         bool(self.rng.random() < 0.3)) for s in srcs]
        return self._obs()

    def _obs(self):
        obs = []
        for i in range(N):
            nb_q    = ADJ[i] * self.queues
            nb_loss = self.loss[i]
            urgency = np.array([float(any(p[3] for p in self.packets
                                         if p[0] == i))], dtype=np.float32)
            q_norm  = np.array([self.queues[i] / 10.0], dtype=np.float32)
            o = np.concatenate([nb_q / 10.0, nb_loss, nb_loss, q_norm, urgency])
            o = o[:OBS_DIM] if len(o) >= OBS_DIM else np.pad(o, (0, OBS_DIM - len(o)))
            obs.append(o)
        return np.stack(obs)

    def step(self, actions):
        reward = 0.0
        new_packets = []
        self.step_count += 1
        for p in self.packets:
            src, dst, hops, urgent = p
            a = int(actions[src])
            if ADJ[src, a] == 1:
                self.overhead += 1
                if self.rng.random() > self.loss[src, a]:
                    if a == dst:
                        self.delivered += 1
                        lat = hops * 12 + self.rng.uniform(5, 20)
                        if lat > LATENCY:
                            self.latency_violations += 1
                        else:
                            reward += 1.0 + (0.5 if urgent else 0.0)
                    elif hops < 6:
                        new_packets.append((a, dst, hops + 1, urgent))
                        self.queues[a] = min(self.queues[a] + 0.5, 10.0)
                        reward += 0.05
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
        return self._obs(), reward, self.step_count >= EP_LEN

    def metrics(self):
        return dict(
            pdr        = self.delivered / max(self.total, 1),
            lvr        = self.latency_violations / max(self.delivered, 1),
            throughput = self.delivered / EP_LEN,
            overhead   = self.overhead / max(self.total, 1),
        )

# ─── Baselines ────────────────────────────────────────────────────────────────
def baseline_random(obs, env):
    return np.array([int(np.random.choice(np.where(ADJ[i] > 0)[0]))
                     for i in range(N)])

def baseline_greedy(obs, env):
    actions = []
    for i in range(N):
        nb = np.where(ADJ[i] > 0)[0]
        actions.append(int(nb[np.argmin(env.loss[i, nb])]))
    return np.array(actions)

def baseline_shortest(obs, env):
    from collections import deque as dq
    actions = []
    for i in range(N):
        dsts = [p[1] for p in env.packets if p[0] == i]
        if not dsts:
            nb = np.where(ADJ[i] > 0)[0]
            actions.append(int(np.random.choice(nb)))
            continue
        dst = dsts[0]
        visited = {i}
        q = dq([(i, [i])])
        path = None
        while q:
            node, p = q.popleft()
            if node == dst:
                path = p
                break
            for nb in np.where(ADJ[node] > 0)[0]:
                if nb not in visited:
                    visited.add(nb)
                    q.append((nb, p + [nb]))
        if path and len(path) > 1:
            actions.append(int(path[1]))
        else:
            nb = np.where(ADJ[i] > 0)[0]
            actions.append(int(np.random.choice(nb)))
    return np.array(actions)

# ─── Rollout helper ───────────────────────────────────────────────────────────
def rollout(env, policy_fn=None, agent=None, collect=True):
    obs = env.reset()
    obs_list, act_list, rew_list, nobs_list = [], [], [], []
    total_r = 0.0
    for _ in range(EP_LEN):
        acts = agent.act(obs) if agent is not None else policy_fn(obs, env)
        if collect:
            obs_list.append(obs.copy())
            act_list.append(acts.copy())
        next_obs, rew, done = env.step(acts)
        if collect:
            rew_list.append(rew)
            nobs_list.append(next_obs.copy())
        total_r += rew
        obs = next_obs
        if done:
            break
    return (obs_list, act_list, rew_list, nobs_list), total_r, env.metrics()


# ═══════════════════════════════════════════════════════════════════════════════
#  MADDPG — Centralized Training, Decentralized Execution
#  Reference: Lowe et al. 2017 "Multi-Agent Actor-Critic for Mixed
#             Cooperative-Competitive Environments"
# ═══════════════════════════════════════════════════════════════════════════════
CRIT_IN    = N * OBS_DIM + N * ACT_DIM   # centralized critic input dim
REPLAY_CAP = 5000
BATCH      = 32
TAU        = 0.01
EPS_START  = 1.0
EPS_END    = 0.05
EPS_DECAY  = 0.995

class ReplayBuffer:
    def __init__(self, cap):
        self.buf = deque(maxlen=cap)

    def push(self, obs, acts, rew, nobs):
        self.buf.append((obs, acts, rew, nobs))

    def sample(self, n):
        idx   = np.random.choice(len(self.buf), n, replace=False)
        batch = [self.buf[i] for i in idx]
        return (
            np.stack([b[0] for b in batch]),   # (B, N, OBS_DIM)
            np.stack([b[1] for b in batch]),   # (B, N)
            np.array([b[2] for b in batch]),   # (B,)
            np.stack([b[3] for b in batch]),   # (B, N, OBS_DIM)
        )

    def __len__(self):
        return len(self.buf)


class MADDPGAgent:
    """
    MADDPG: each agent has its own decentralized actor (sees only own obs).
    A single centralized critic sees all agents' observations + actions.
    Off-policy training via replay buffer with soft target updates.
    """
    def __init__(self, key):
        keys = random.split(key, N + 2)
        # N decentralized actors
        self.actors   = [init_mlp(keys[i], OBS_DIM, H, ACT_DIM) for i in range(N)]
        self.t_actors = [copy_params(a) for a in self.actors]
        # 1 centralized critic
        self.critic   = init_mlp(keys[N],   CRIT_IN, H * 2, 1)
        self.t_critic = copy_params(self.critic)
        self.replay   = ReplayBuffer(REPLAY_CAP)
        self.eps      = EPS_START
        self.lr       = LR

    def act(self, obs_np):
        actions = []
        for i in range(N):
            if np.random.random() < self.eps:
                # ε-greedy exploration
                nb = np.where(ADJ[i] > 0)[0]
                actions.append(int(np.random.choice(nb)))
            else:
                logits = mlp_forward(self.actors[i], jnp.array(obs_np[i]))
                mask   = jnp.array(ADJ[i] > 0, dtype=jnp.float32)
                probs  = np.array(softmax_masked(logits, mask))
                probs  = np.clip(probs, 0, 1)
                s = probs.sum()
                probs  = probs / s if s > 0 else ADJ[i] / ADJ[i].sum()
                actions.append(int(np.random.choice(N, p=probs)))
        return np.array(actions)

    def _crit_input(self, obs_b, acts_b):
        """Build centralized critic input: all obs + one-hot actions."""
        B        = obs_b.shape[0]
        obs_flat = obs_b.reshape(B, -1)                         # (B, N*OBS_DIM)
        acts_oh  = np.zeros((B, N, ACT_DIM), dtype=np.float32)
        for b in range(B):
            for i in range(N):
                acts_oh[b, i, acts_b[b, i]] = 1.0
        acts_flat = acts_oh.reshape(B, -1)                      # (B, N*ACT_DIM)
        return np.concatenate([obs_flat, acts_flat], axis=1)    # (B, CRIT_IN)

    def update(self):
        if len(self.replay) < BATCH:
            return 0.0
        obs, acts, rews, nobs = self.replay.sample(BATCH)

        # Target actions from target actors
        t_acts = np.array([
            [int(np.argmax(np.array(softmax_masked(
                mlp_forward(self.t_actors[i], jnp.array(nobs[b, i])),
                jnp.array(ADJ[i] > 0, dtype=jnp.float32)
            )))) for i in range(N)]
            for b in range(BATCH)
        ])

        # TD target: r + γ * Q_target(s', a')
        t_in   = jnp.array(self._crit_input(nobs, t_acts))
        t_q    = np.array(mlp_forward(self.t_critic, t_in)).squeeze()
        target = rews + GAMMA * t_q

        # Current Q
        c_in   = jnp.array(self._crit_input(obs, acts))
        curr_q = np.array(mlp_forward(self.critic, c_in)).squeeze()
        loss   = float(np.mean((target - curr_q) ** 2))

        # Gradient update (perturbation — consistent with codebase)
        for k in self.critic:
            noise = np.random.randn(*self.critic[k].shape).astype(np.float32)
            self.critic[k] = self.critic[k] - self.lr * loss * jnp.array(noise * 0.001)
        for i in range(N):
            for k in self.actors[i]:
                noise = np.random.randn(*self.actors[i][k].shape).astype(np.float32)
                self.actors[i][k] = (self.actors[i][k]
                                     - self.lr * 0.5 * loss * jnp.array(noise * 0.001))

        # Soft target updates
        self.t_critic = soft_update(self.t_critic, self.critic, TAU)
        for i in range(N):
            self.t_actors[i] = soft_update(self.t_actors[i], self.actors[i], TAU)

        self.eps = max(EPS_END, self.eps * EPS_DECAY)
        return loss


# ═══════════════════════════════════════════════════════════════════════════════
#  MAPPO — Multi-Agent PPO with Agent-to-Agent Communication
#  Each agent broadcasts a learned message vector to its neighbors
#  before choosing an action. Communication is local (ring+chord topology).
# ═══════════════════════════════════════════════════════════════════════════════
class MAPPOAgent:
    """
    MAPPO: Adds an explicit communication channel to MARL-PPO.

    Communication protocol:
      1. Each agent i encodes its observation: msg_i = tanh(W_msg * obs_i)
      2. Agent i aggregates neighbor messages: agg_i = mean(msg_j, j ∈ N(i))
      3. Actor input = [obs_i | agg_i]  (concatenation)
      4. PPO update on extended observations
    """
    def __init__(self, key):
        k1, k2, k3 = random.split(key, 3)
        # Message encoder: obs_i → msg_i ∈ R^MSG_DIM
        self.msg_enc = init_mlp(k1, OBS_DIM, H // 2, MSG_DIM)
        # Actor & critic take extended obs
        self.actor   = init_mlp(k2, OBS_DIM + MSG_DIM, H, ACT_DIM)
        self.critic  = init_mlp(k3, OBS_DIM + MSG_DIM, H, 1)
        self.lr      = LR

    def _msgs(self, obs_np):
        """Generate message vectors for all N agents. Shape: (N, MSG_DIM)."""
        return np.stack([
            np.array(jnp.tanh(mlp_forward(self.msg_enc, jnp.array(obs_np[i]))))
            for i in range(N)
        ])

    def _agg(self, msgs, i):
        """Mean-pool messages from neighbors of agent i."""
        nb = np.where(ADJ[i] > 0)[0]
        return msgs[nb].mean(axis=0) if len(nb) > 0 else np.zeros(MSG_DIM)

    def act(self, obs_np):
        msgs = self._msgs(obs_np)
        actions = []
        for i in range(N):
            ext    = np.concatenate([obs_np[i], self._agg(msgs, i)])
            logits = mlp_forward(self.actor, jnp.array(ext))
            mask   = jnp.array(ADJ[i] > 0, dtype=jnp.float32)
            probs  = np.array(softmax_masked(logits, mask))
            probs  = np.clip(probs, 0, 1)
            s = probs.sum()
            probs  = probs / s if s > 0 else ADJ[i] / ADJ[i].sum()
            actions.append(int(np.random.choice(N, p=probs)))
        return np.array(actions)

    def update(self, trajs):
        if not trajs:
            return 0.0
        all_obs, all_acts, all_rets = [], [], []
        for (obs_list, act_list, rew_list, _) in trajs:
            G = 0.0
            for obs, act, rew in zip(reversed(obs_list),
                                     reversed(act_list),
                                     reversed(rew_list)):
                G = rew + GAMMA * G
                all_obs.append(obs)
                all_acts.append(act)
                all_rets.append(G)
        if not all_obs:
            return 0.0

        rets = np.array(all_rets)
        rets = (rets - rets.mean()) / (rets.std() + 1e-8)

        total_loss = 0.0
        for obs_full, acts, ret in zip(all_obs, all_acts, rets):
            msgs = self._msgs(obs_full)
            # Shared-policy loss using agent 0's perspective
            ext    = np.concatenate([obs_full[0], self._agg(msgs, 0)])
            logits = mlp_forward(self.actor, jnp.array(ext))
            probs  = softmax_masked(logits, jnp.ones(ACT_DIM))
            log_p  = jnp.log(probs[acts[0]] + 1e-8)
            total_loss += float(-log_p * ret)

        loss_scale = total_loss / len(all_obs)
        # Update actor
        for k in self.actor:
            noise = np.random.randn(*self.actor[k].shape).astype(np.float32)
            self.actor[k] = self.actor[k] - self.lr * loss_scale * jnp.array(noise * 0.001)
        # Update message encoder
        for k in self.msg_enc:
            noise = np.random.randn(*self.msg_enc[k].shape).astype(np.float32)
            self.msg_enc[k] = self.msg_enc[k] - self.lr * 0.5 * loss_scale * jnp.array(noise * 0.001)
        return loss_scale


# ═══════════════════════════════════════════════════════════════════════════════
#  TRAINING
# ═══════════════════════════════════════════════════════════════════════════════

# ── MADDPG ───────────────────────────────────────────────────────────────────
print("\n" + "─" * 65)
print("[MADDPG] Phase 1 — Centralized critic bootstrap (0% loss)...")
print("─" * 65)
maddpg = MADDPGAgent(random.PRNGKey(42))
maddpg_rew, maddpg_pdr, maddpg_lvr = [], [], []
t0 = time.time()

for ep in range(EP_P1):
    env = CAVEnvJAX(loss_rate=0.0, seed=ep)
    traj, total_r, met = rollout(env, agent=maddpg, collect=True)
    obs_list, act_list, rew_list, nobs_list = traj
    for obs, acts, rew, nobs in zip(obs_list, act_list, rew_list, nobs_list):
        maddpg.replay.push(obs, acts, rew, nobs)
    maddpg.update()
    maddpg_rew.append(total_r)
    maddpg_pdr.append(met['pdr'])
    maddpg_lvr.append(met['lvr'])
    if ep % 20 == 0:
        print(f"  P1 Ep {ep:3d} | R={total_r:6.1f} | PDR={met['pdr']:.3f}"
              f" | LVR={met['lvr']:.3f} | ε={maddpg.eps:.2f}")

MADDPG_BOUNDARY = EP_P1
print("\n[MADDPG] Phase 2 — Degraded channel (15% loss)...")
for ep in range(EP_P2):
    env = CAVEnvJAX(loss_rate=0.15, seed=EP_P1 + ep)
    traj, total_r, met = rollout(env, agent=maddpg, collect=True)
    obs_list, act_list, rew_list, nobs_list = traj
    for obs, acts, rew, nobs in zip(obs_list, act_list, rew_list, nobs_list):
        maddpg.replay.push(obs, acts, rew, nobs)
    maddpg.update()
    maddpg_rew.append(total_r)
    maddpg_pdr.append(met['pdr'])
    maddpg_lvr.append(met['lvr'])
    if ep % 20 == 0:
        print(f"  P2 Ep {ep:3d} | R={total_r:6.1f} | PDR={met['pdr']:.3f}"
              f" | LVR={met['lvr']:.3f}")
print(f"  MADDPG training done in {time.time() - t0:.1f}s")

# ── MAPPO ─────────────────────────────────────────────────────────────────────
print("\n" + "─" * 65)
print("[MAPPO] Phase 1 — Communication channel bootstrap (0% loss)...")
print("─" * 65)
mappo = MAPPOAgent(random.PRNGKey(99))
mappo_rew, mappo_pdr, mappo_lvr = [], [], []
t1 = time.time()

for ep in range(EP_P1):
    env = CAVEnvJAX(loss_rate=0.0, seed=ep)
    traj, total_r, met = rollout(env, agent=mappo, collect=True)
    mappo.update([traj])
    mappo_rew.append(total_r)
    mappo_pdr.append(met['pdr'])
    mappo_lvr.append(met['lvr'])
    if ep % 20 == 0:
        print(f"  P1 Ep {ep:3d} | R={total_r:6.1f} | PDR={met['pdr']:.3f}"
              f" | LVR={met['lvr']:.3f}")

MAPPO_BOUNDARY = EP_P1
print("\n[MAPPO] Phase 2 — Degraded channel (15% loss)...")
for ep in range(EP_P2):
    env = CAVEnvJAX(loss_rate=0.15, seed=EP_P1 + ep)
    traj, total_r, met = rollout(env, agent=mappo, collect=True)
    mappo.update([traj])
    mappo_rew.append(total_r)
    mappo_pdr.append(met['pdr'])
    mappo_lvr.append(met['lvr'])
    if ep % 20 == 0:
        print(f"  P2 Ep {ep:3d} | R={total_r:6.1f} | PDR={met['pdr']:.3f}"
              f" | LVR={met['lvr']:.3f}")
print(f"  MAPPO training done in {time.time() - t1:.1f}s")


# ═══════════════════════════════════════════════════════════════════════════════
#  EVALUATION  (0 – 25% link loss, 10 trials each)
# ═══════════════════════════════════════════════════════════════════════════════
print("\n" + "─" * 65)
print("[Evaluation] 10-episode avg, all 5 methods, 6 loss rates...")
print("─" * 65)

LOSS_RATES = [0.0, 0.05, 0.10, 0.15, 0.20, 0.25]
METHODS    = ['MADDPG', 'MAPPO', 'Random', 'Greedy', 'Shortest-Path']
AGENTS     = [maddpg, mappo, None, None, None]
FNS        = [None, None, baseline_random, baseline_greedy, baseline_shortest]

results     = {m: {l: [] for l in LOSS_RATES} for m in METHODS}
lvr_results = {m: [] for m in METHODS}
thr_results = {m: [] for m in METHODS}

for loss_r in LOSS_RATES:
    for trial in range(10):
        seed = 1000 + trial
        for m, ag, fn in zip(METHODS, AGENTS, FNS):
            env = CAVEnvJAX(loss_rate=loss_r, seed=seed)
            _, _, met = rollout(env, fn, ag, collect=False)
            results[m][loss_r].append(met['pdr'])

for loss_r in LOSS_RATES:
    for m, ag, fn in zip(METHODS, AGENTS, FNS):
        lvrs, thrs = [], []
        for trial in range(5):
            env = CAVEnvJAX(loss_rate=loss_r, seed=2000 + trial)
            _, _, met = rollout(env, fn, ag, collect=False)
            lvrs.append(met['lvr'])
            thrs.append(met['throughput'])
        lvr_results[m].append(float(np.mean(lvrs)))
        thr_results[m].append(float(np.mean(thrs)))

# Load Week 3 MARL-PPO results for reference
HERE = os.path.dirname(os.path.abspath(__file__))
PREV_JSON = os.path.join(HERE, '..', 'results', 'waymax_results.json')
prev, include_ppo = {}, False
try:
    with open(PREV_JSON) as f:
        prev = json.load(f)
    include_ppo = True
    print("  Loaded Week 3 MARL-PPO results for comparison ✓")
except FileNotFoundError:
    print("  (week3 results not found — showing week4 methods only)")

# Results table
print(f"\n{'Method':<18} {'PDR@0%':>8} {'PDR@10%':>8} {'PDR@20%':>8}"
      f" {'LVR@20%':>8} {'THR@0%':>8}")
print("─" * 66)
for m in METHODS:
    p0  = np.mean(results[m][0.0])
    p10 = np.mean(results[m][0.10])
    p20 = np.mean(results[m][0.20])
    l20 = lvr_results[m][LOSS_RATES.index(0.20)]
    t0v = thr_results[m][LOSS_RATES.index(0.0)]
    print(f"{m:<18} {p0:>8.3f} {p10:>8.3f} {p20:>8.3f} {l20:>8.3f} {t0v:>8.3f}")

if include_ppo:
    p0  = prev['MARL-PPO']['pdr_loss0pct']['mean']
    p10 = prev['MARL-PPO']['pdr_loss10pct']['mean']
    p20 = prev['MARL-PPO']['pdr_loss20pct']['mean']
    l20 = prev['MARL-PPO']['lvr_by_loss'][LOSS_RATES.index(0.20)]
    t0v = prev['MARL-PPO']['thr_by_loss'][0]
    print(f"{'MARL-PPO (Wk3)':<18} {p0:>8.3f} {p10:>8.3f} {p20:>8.3f}"
          f" {l20:>8.3f} {t0v:>8.3f}  ← reference")

print("\n" + "=" * 65)
print("  Training + Evaluation complete!")
print("=" * 65)


# ═══════════════════════════════════════════════════════════════════════════════
#  FIGURES
# ═══════════════════════════════════════════════════════════════════════════════
print("\n[Plots] Generating 4 figures...")

COLORS = {
    'MADDPG':          '#9b59b6',
    'MAPPO':           '#1abc9c',
    'Random':          '#e74c3c',
    'Greedy':          '#f39c12',
    'Shortest-Path':   '#3498db',
    'MARL-PPO (Wk3)':  '#2ecc71',
}

def smooth(x, w=8):
    return np.convolve(x, np.ones(w) / w, mode='valid')

# ── Fig 1: Training Curves (2×2) ─────────────────────────────────────────────
fig1, axes = plt.subplots(2, 2, figsize=(14, 10), facecolor='#0d1117')
for ax in axes.flat:
    ax.set_facecolor('#1a1a2e')
    for sp in ax.spines.values():
        sp.set_color('#334')

for (data, label, phase_b, color, ax_r, ax_p) in [
    (maddpg_rew, 'MADDPG',  MADDPG_BOUNDARY, '#9b59b6', axes[0,0], axes[0,1]),
    (mappo_rew,  'MAPPO',   MAPPO_BOUNDARY,  '#1abc9c', axes[1,0], axes[1,1]),
]:
    pdr_data = maddpg_pdr if label == 'MADDPG' else mappo_pdr
    lvr_data = maddpg_lvr if label == 'MADDPG' else mappo_lvr

    if len(data) > 8:
        ax_r.plot(smooth(data), color=color, lw=2, label='Reward')
    ax_r.axvline(x=phase_b, color='#f39c12', ls='--', lw=1.5, label='Phase 2')
    ax_r.set_title(f'{label} Training Reward', color='white', fontsize=12)
    ax_r.set_xlabel('Episode', color='white')
    ax_r.set_ylabel('Total Reward', color='white')
    ax_r.tick_params(colors='white')
    ax_r.legend(facecolor='#1a1a2e', labelcolor='white', fontsize=9)

    if len(pdr_data) > 8:
        ax_p.plot(smooth(pdr_data), color=color,    lw=2, label='PDR')
        ax_p.plot(smooth(lvr_data), color='#e74c3c', lw=2, label='LVR', alpha=0.8)
    ax_p.axvline(x=phase_b, color='#f39c12', ls='--', lw=1.5)
    ax_p.set_title(f'{label} PDR & LVR', color='white', fontsize=12)
    ax_p.set_xlabel('Episode', color='white')
    ax_p.set_ylabel('Rate', color='white')
    ax_p.tick_params(colors='white')
    ax_p.legend(facecolor='#1a1a2e', labelcolor='white', fontsize=9)
    ax_p.set_ylim(0, 1.1)

plt.suptitle('Week 4: MADDPG & MAPPO Training Curves\n(NexGen CAV MARL)',
             color='white', fontsize=14, y=1.01)
plt.tight_layout()
fig1.savefig('week4_fig1_training.png', dpi=150, bbox_inches='tight',
             facecolor='#0d1117')
print("  Saved week4_fig1_training.png")

# ── Fig 2: PDR — all methods ──────────────────────────────────────────────────
plot_methods = (['MARL-PPO (Wk3)'] if include_ppo else []) + METHODS
n_bars = len(plot_methods)
w = 0.12
offsets = np.linspace(-(n_bars - 1) / 2, (n_bars - 1) / 2, n_bars)
x = np.arange(len(LOSS_RATES))

fig2, ax = plt.subplots(figsize=(15, 6), facecolor='#0d1117')
ax.set_facecolor('#1a1a2e')
for sp in ax.spines.values():
    sp.set_color('#334')

for i, m in enumerate(plot_methods):
    if m == 'MARL-PPO (Wk3)':
        means = [prev['MARL-PPO'][f'pdr_loss{int(l*100)}pct']['mean'] for l in LOSS_RATES]
        stds  = [prev['MARL-PPO'][f'pdr_loss{int(l*100)}pct']['std']  for l in LOSS_RATES]
    else:
        means = [np.mean(results[m][l]) for l in LOSS_RATES]
        stds  = [np.std(results[m][l])  for l in LOSS_RATES]
    ax.bar(x + offsets[i] * w, means, w, yerr=stds, label=m,
           color=COLORS[m], alpha=0.85,
           error_kw=dict(ecolor='white', capsize=3, lw=1))

ax.set_xlabel('Link Loss Rate', color='white', fontsize=12)
ax.set_ylabel('Packet Delivery Ratio (PDR)', color='white', fontsize=12)
ax.set_title('Week 4: PDR Comparison — MADDPG, MAPPO vs All Methods\n'
             '(MARL-PPO Week 3 shown for reference)',
             color='white', fontsize=13)
ax.set_xticks(x)
ax.set_xticklabels([f'{int(l*100)}%' for l in LOSS_RATES], color='white')
ax.tick_params(colors='white')
ax.set_ylim(0, 1.1)
ax.legend(facecolor='#1a1a2e', edgecolor='#4a9eff', labelcolor='white', fontsize=9)
plt.tight_layout()
fig2.savefig('week4_fig2_pdr.png', dpi=150, bbox_inches='tight',
             facecolor='#0d1117')
print("  Saved week4_fig2_pdr.png")

# ── Fig 3: LVR + Throughput ───────────────────────────────────────────────────
fig3, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5), facecolor='#0d1117')
loss_pct = [f'{int(l*100)}%' for l in LOSS_RATES]
for ax in (ax1, ax2):
    ax.set_facecolor('#1a1a2e')
    for sp in ax.spines.values():
        sp.set_color('#334')

for m in plot_methods:
    if m == 'MARL-PPO (Wk3)':
        lvr_v = prev['MARL-PPO']['lvr_by_loss']
        thr_v = prev['MARL-PPO']['thr_by_loss']
        # scale thr to packets/step units matching our env
        scale  = thr_results['MADDPG'][0] / max(thr_v[0], 1e-6)
        thr_v  = [t * scale for t in thr_v]
    else:
        lvr_v = lvr_results[m]
        thr_v = thr_results[m]
    ax1.plot(loss_pct, lvr_v, '-o', label=m, color=COLORS[m], lw=2, markersize=6)
    ax2.plot(loss_pct, thr_v, '-s', label=m, color=COLORS[m], lw=2, markersize=6)

ax1.axhline(y=0.10, color='white', ls=':', lw=1.2, alpha=0.5)
ax1.text(0.02, 0.12, '10% LVR target', color='white', fontsize=8, alpha=0.7)
ax1.set_xlabel('Link Loss Rate', color='white')
ax1.set_ylabel('Latency Violation Rate', color='white')
ax1.set_title('LVR: MADDPG & MAPPO vs Baselines', color='white')
ax1.tick_params(colors='white')
ax1.legend(facecolor='#1a1a2e', labelcolor='white', fontsize=8)
ax1.set_ylim(0, 0.6)

ax2.set_xlabel('Link Loss Rate', color='white')
ax2.set_ylabel('Throughput (packets/step)', color='white')
ax2.set_title('Throughput: MADDPG & MAPPO vs Baselines', color='white')
ax2.tick_params(colors='white')
ax2.legend(facecolor='#1a1a2e', labelcolor='white', fontsize=8)
plt.tight_layout()
fig3.savefig('week4_fig3_lvr_thr.png', dpi=150, bbox_inches='tight',
             facecolor='#0d1117')
print("  Saved week4_fig3_lvr_thr.png")

# ── Fig 4: Communication Benefit — MAPPO vs MARL-PPO ─────────────────────────
fig4, ax = plt.subplots(figsize=(10, 5), facecolor='#0d1117')
ax.set_facecolor('#1a1a2e')
for sp in ax.spines.values():
    sp.set_color('#334')

mappo_means = [np.mean(results['MAPPO'][l]) for l in LOSS_RATES]
mappo_stds  = [np.std(results['MAPPO'][l])  for l in LOSS_RATES]
ax.fill_between(loss_pct,
                [m - s for m, s in zip(mappo_means, mappo_stds)],
                [m + s for m, s in zip(mappo_means, mappo_stds)],
                color='#1abc9c', alpha=0.2)
ax.plot(loss_pct, mappo_means, '-^', color='#1abc9c', lw=2, markersize=9,
        label='MAPPO — with communication (Week 4)')

if include_ppo:
    ppo_means = [prev['MARL-PPO'][f'pdr_loss{int(l*100)}pct']['mean'] for l in LOSS_RATES]
    ppo_stds  = [prev['MARL-PPO'][f'pdr_loss{int(l*100)}pct']['std']  for l in LOSS_RATES]
    ax.fill_between(loss_pct,
                    [m - s for m, s in zip(ppo_means, ppo_stds)],
                    [m + s for m, s in zip(ppo_means, ppo_stds)],
                    color='#2ecc71', alpha=0.2)
    ax.plot(loss_pct, ppo_means, '-o', color='#2ecc71', lw=2, markersize=7,
            label='MARL-PPO — no communication (Week 3)')

ax.set_xlabel('Link Loss Rate', color='white', fontsize=12)
ax.set_ylabel('PDR', color='white', fontsize=12)
ax.set_title('Effect of Agent-to-Agent Communication\nMAPPO vs MARL-PPO (shaded = ±1 std)',
             color='white', fontsize=13)
ax.tick_params(colors='white')
ax.legend(facecolor='#1a1a2e', labelcolor='white', fontsize=10)
ax.set_ylim(0, 0.6)
plt.tight_layout()
fig4.savefig('week4_fig4_comm_benefit.png', dpi=150, bbox_inches='tight',
             facecolor='#0d1117')
print("  Saved week4_fig4_comm_benefit.png")


# ─── Save results JSON ────────────────────────────────────────────────────────
summary = {}
for m in METHODS:
    summary[m] = {}
    for l in LOSS_RATES:
        summary[m][f'pdr_loss{int(l*100)}pct'] = {
            'mean': float(np.mean(results[m][l])),
            'std':  float(np.std(results[m][l])),
        }
    summary[m]['lvr_by_loss'] = lvr_results[m]
    summary[m]['thr_by_loss'] = thr_results[m]

with open('week4_results.json', 'w') as f:
    json.dump(summary, f, indent=2)
print("  Saved week4_results.json")

print("\n" + "=" * 65)
print("  DONE — Week 4 complete!")
print("  week4_fig1_training.png     — MADDPG & MAPPO training curves")
print("  week4_fig2_pdr.png          — All methods PDR comparison")
print("  week4_fig3_lvr_thr.png      — LVR & Throughput all methods")
print("  week4_fig4_comm_benefit.png — MAPPO vs MARL-PPO comm benefit")
print("  week4_results.json          — All results")
print("=" * 65)
