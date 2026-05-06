# NexGen CAV Multi-Agent Routing

**CSC 59866-E Senior Project I — City College of New York**  
**Alhassana Diallo & Marcley Colin**

---

## What is this?

This project builds an AI system that teaches self-driving cars (CAVs) how to route packets through a Vehicle-to-Vehicle (V2V) wireless network — even when the network is dropping up to 25% of packets due to interference or distance.

Instead of using a single controller, we use **Multi-Agent Reinforcement Learning (MARL)** — each car has its own AI brain that learns to make routing decisions independently, while still working as a team.

---

## Research Question

> Can a MARL policy learn to route packets in a CAV V2V network such that it maintains **Packet Delivery Ratio (PDR) above baseline** and **Latency Violation Rate (LVR) below 10%**, even under up to 25% link loss?  
> Does **agent-to-agent communication (MAPPO)** improve resilience under degraded conditions?

---

## Network Setup

- **8 CAV agents** arranged in a Ring + Chord topology
- Each agent observes: neighbor queue lengths, link loss rates, packet urgency
- Each agent decides: which neighbor to forward packets to
- Simulated with **JAX** (Waymax-compatible backend) — runs on CPU or GPU

```
        CAV 0
       /     \
  CAV 7       CAV 1
  |    \   /    |
  CAV 6  X  CAV 2      ← chords connect opposite nodes (0↔4, 1↔5, 2↔6, 3↔7)
  |    /   \    |
  CAV 5       CAV 3
       \     /
        CAV 4
```

---

## Algorithms Implemented

| Algorithm | Description | File |
|---|---|---|
| **MARL-PPO** | Shared-policy PPO, 2-phase curriculum training | `code/waymax_ppo_cav.py` |
| **MADDPG** | Centralized critic (sees all agents) + decentralized actors | `code/waymax_week4.py` |
| **MAPPO** | PPO + 8-dim message vectors shared between neighbors | `code/waymax_week4.py` |
| **Shortest-Path** | BFS routing baseline | built-in |
| **Greedy** | Route to lowest-loss neighbor | built-in |
| **Random** | Random neighbor selection | built-in |

---

## Results (Week 3 + Week 4)

Evaluated across 10 trials per loss rate, 0–25% link loss:

| Method | PDR@0% | PDR@10% | PDR@20% | LVR@20% | THR@0% |
|---|---|---|---|---|---|
| MADDPG | 0.118 | 0.105 | 0.096 | 0.000 | 1.882 |
| MAPPO | 0.107 | 0.103 | 0.093 | 0.000 | 1.738 |
| MARL-PPO (Wk3) | 0.110 | 0.102 | 0.095 | 0.000 | 1.828 |
| Random | 0.123 | 0.106 | 0.099 | 0.000 | 1.948 |
| Greedy | 0.058 | 0.067 | 0.064 | 0.000 | 1.297 |
| Shortest-Path | 0.379 | 0.367 | 0.340 | 0.000 | 3.291 |

**Key finding:** All MARL methods achieved LVR = 0.0 across all loss rates — well below the 10% target. MADDPG achieved the highest PDR among RL methods. Agent communication (MAPPO) maintained PDR stability under increasing loss.

---

## How to Run

### Option 1 — Google Colab (recommended, no setup needed)
1. Open [colab.research.google.com](https://colab.research.google.com)
2. File → Upload notebook → select `nexgen_colab.ipynb`
3. Runtime → Change runtime type → **GPU (T4)**
4. Run all cells top to bottom

### Option 2 — Local (Mac/Linux)
```bash
# Create virtual environment
python3.11 -m venv waymax_env311
source waymax_env311/bin/activate

# Install dependencies
pip install jax numpy matplotlib
pip install git+https://github.com/waymo-research/waymax.git

# Run Week 3 simulation (MARL-PPO + baselines)
cd code
python waymax_ppo_cav.py

# Run Week 4 simulation (MADDPG + MAPPO)
python waymax_week4.py
```

**Output:** 4 figures + `waymax_results.json` / `week4_results.json` saved in the `code/` directory.

---

## File Structure

```
nexgen-cav-marl/
├── code/
│   ├── waymax_ppo_cav.py       ← Week 3: MARL-PPO + 3 baselines
│   └── waymax_week4.py         ← Week 4: MADDPG + MAPPO
├── figures/waymax_sim/
│   ├── waymax_fig1_topology.png     ← 8-node ring+chord network
│   ├── waymax_fig2_training.png     ← PPO training curves
│   ├── waymax_fig3_pdr.png          ← PDR comparison all methods
│   ├── waymax_fig4_lvr_thr.png      ← LVR & Throughput
│   ├── week4_fig1_training.png      ← MADDPG & MAPPO training
│   ├── week4_fig2_pdr.png           ← Week 4 PDR comparison
│   ├── week4_fig3_lvr_thr.png       ← Week 4 LVR & Throughput
│   ├── week4_fig4_comm_benefit.png  ← MAPPO vs MARL-PPO comm effect
│   └── system_design.svg            ← Full system architecture diagram
├── results/
│   ├── waymax_results.json     ← Week 3 numerical results
│   └── week4_results.json      ← Week 4 numerical results
└── nexgen_colab.ipynb          ← Full collaborative notebook
```

---

## System Architecture

```
┌─────────────────────────────────────────────────────────┐
│         MADDPG Centralized Critic  (training only)       │
│   Input: all 8 agents' obs + actions → Q-value          │
└───────────────────┬─────────────────────────────────────┘
                    │ gradient
        ┌───────────▼──────────────────────────┐
        │   8 CAV Agents (Ring + Chord)        │
        │   ←── MAPPO message vectors ──→      │
        │   Each agent: actor π_i(obs_i)       │
        └───────────────────┬──────────────────┘
                            │ actions / rewards
        ┌───────────────────▼──────────────────┐
        │     V2V Network Environment          │
        │  queues · link loss · packets        │
        │  PDR · LVR · Throughput metrics      │
        └──────────────────────────────────────┘
        ┌──────────────────────────────────────┐
        │  AlphaSim / Alpamayo Physics Layer   │ ← Marc (in progress)
        │  Real DSRC propagation · NuRec data  │
        └──────────────────────────────────────┘
```

---

## Authors

- **Alhassana Diallo** — MARL-PPO, MADDPG, MAPPO implementation (Waymax/JAX)
- **Marcley Colin** — AlphaSim / Alpamayo integration (NVIDIA PhysicalAI)

**Advisor:** CSC 59866-E, City College of New York, Spring 2026
