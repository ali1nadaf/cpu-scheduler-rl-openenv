# 🖥️ CPU Scheduler — Reinforcement Learning Environment

> **OpenEnv-compliant RL environment for CPU scheduling optimization**  
> Bridges Operating Systems theory and Artificial Intelligence in a hackathon-ready package.

---

## 📌 Overview

Modern operating systems must efficiently schedule hundreds of competing processes for limited CPU time. Poor decisions cascade into high latency, wasted resources, and degraded user experience. This project models **CPU scheduling as a Reinforcement Learning problem**, where an agent learns — through trial and error — to make better scheduling decisions than hand-crafted rules.

The environment is fully **OpenEnv-compliant** (`reset()` / `step()` / `state()`) and supports three difficulty levels, five benchmark policies, and a built-in Q-learning agent.

---

## 🎯 Problem Statement

| Challenge | Impact |
|-----------|--------|
| High waiting time | Processes starve; user latency grows |
| Low CPU utilisation | Wasted compute; reduced throughput |
| Priority inversion | Critical tasks delayed by low-priority work |
| Dynamic arrivals | Agent must adapt without full information |

The agent must learn which process to schedule at each tick to **minimise waiting time**, **prevent starvation**, and **maximise throughput** — all simultaneously.

---

## ⚙️ Environment Design

### State Space

The observation returned by `state()` / `reset()` is a Python `dict`:

```python
{
    "time":             int,        # current simulation clock
    "ready_queue":      list[dict], # schedulable processes
    "queue_length":     int,        # processes ready now
    "pending_arrivals": int,        # processes not yet arrived
    "completed_count":  int,        # processes finished this episode
    "idle_ticks":       int,        # CPU idle time so far
    "difficulty":       str,        # "easy" | "medium" | "hard"
}
```

Each process in `ready_queue` exposes:

| Field | Type | Description |
|-------|------|-------------|
| `id` | int | Unique process identifier |
| `arrival` | int | Tick the process entered the system |
| `remaining` | int | CPU ticks still needed |
| `priority` | int (1–5) | Scheduling importance (5 = highest) |
| `waiting` | int | Accumulated ticks spent waiting |

### Action Space

**Discrete integer** — the index of the process in `ready_queue` to run for one tick.  
`None` or an out-of-range index causes the CPU to idle.

```python
action: int | None   # 0 … len(ready_queue)-1
```

### Internal Data Model

Processes are represented internally as typed `@dataclass` objects (`Process`), ensuring correctness and IDE support. The environment serialises them to plain dicts for the agent.

---

## 💰 Reward Function

The reward is designed to produce **dense, informative signals** at every tick:

| Component | Value | Trigger |
|-----------|-------|---------|
| Base step cost | −1.0 | Every scheduled tick |
| Partial progress | +0.5 | Reducing a process's remaining time |
| Starvation penalty | −0.05 × max\_wait | Proportional to queue waiting time |
| Idle penalty | −3.0 | CPU left idle when jobs are ready |
| Invalid action | −8.0 | Out-of-range index |
| Completion bonus | +10.0 | Process finishes |
| Priority bonus | +2 × priority | High-priority completion |
| Turnaround bonus | max(0, 20 − turnaround) | Low turnaround time |

**Reward range:** `[−100, +100]` (clipped)

---

## 📊 Difficulty Levels

| Level | Processes | Arrivals | Priorities | Dynamic Injection |
|-------|-----------|----------|------------|-------------------|
| **Easy** | 4 | All at t=0 | Uniform (3) | ✗ |
| **Medium** | 6 | Staggered t∈[0,6] | Varied 1–5 | ✗ |
| **Hard** | 8 | Staggered t∈[0,5] | Varied 1–5 | ✓ (2–3 bursts at t∈[5,20]) |

---

## 📈 Benchmark Results

> Results averaged over **30 episodes per policy**, seed-controlled for reproducibility.

### Easy

| Policy | Avg Reward | Avg Wait | Avg Turnaround | CPU Util % |
|--------|-----------|----------|----------------|------------|
| FCFS | ~52 | ~2.5 | ~6.0 | ~100 |
| **SJF** | **~61** | **~1.8** | **~5.2** | ~100 |
| Priority | ~58 | ~2.1 | ~5.6 | ~100 |
| RoundRobin | ~48 | ~3.1 | ~7.0 | ~100 |
| Random | ~40 | ~3.8 | ~7.8 | ~98 |

### Medium

| Policy | Avg Reward | Avg Wait | Avg Turnaround | CPU Util % |
|--------|-----------|----------|----------------|------------|
| FCFS | ~38 | ~4.2 | ~9.1 | ~94 |
| **SJF** | **~49** | **~2.9** | **~7.4** | ~96 |
| Priority | ~45 | ~3.4 | ~8.0 | ~95 |
| RoundRobin | ~33 | ~5.0 | ~10.2 | ~93 |
| Random | ~22 | ~6.1 | ~11.8 | ~89 |

### Hard

| Policy | Avg Reward | Avg Wait | Avg Turnaround | CPU Util % |
|--------|-----------|----------|----------------|------------|
| FCFS | ~21 | ~6.8 | ~13.2 | ~91 |
| SJF | ~35 | ~4.5 | ~10.0 | ~94 |
| **Priority** | **~37** | **~4.1** | **~9.6** | ~94 |
| RoundRobin | ~18 | ~7.5 | ~14.0 | ~90 |
| Random | ~8 | ~9.2 | ~16.4 | ~85 |

---

## 🧠 RL Training — Q-Learning

### Algorithm

The agent uses **tabular ε-greedy Q-Learning**:

```
Q(s, a) ← Q(s, a) + α × [r + γ × max Q(s', a') − Q(s, a)]
```

| Hyperparameter | Value |
|---------------|-------|
| Learning rate (α) | 0.15 |
| Discount factor (γ) | 0.97 |
| Initial ε | 1.0 |
| ε minimum | 0.05 |
| ε decay | 0.997 per episode |

### State Discretisation

The raw state is mapped to a compact 4-tuple for tabular storage:

```python
(queue_length, min_remaining, max_waiting, max_priority)
           ↓            ↓              ↓              ↓
        [0..8]       [0..10]        [0..20]        [1..5]
```

This keeps the Q-table small (~few hundred states) while preserving the most decision-relevant information.

### Training Progress (medium, 400 episodes)

```
Ep  100  ε=0.741  train_avg=  28.4  eval_avg=  31.2  Q-states=87
Ep  200  ε=0.549  train_avg=  35.1  eval_avg=  38.6  Q-states=142
Ep  300  ε=0.407  train_avg=  41.3  eval_avg=  44.9  Q-states=178
Ep  400  ε=0.301  train_avg=  45.8  eval_avg=  47.3  Q-states=201
```

---

## 🚀 How to Run

### Local

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Full benchmark (all policies × all difficulty levels)
python inference.py --mode benchmark

# 3. Verbose single-episode demo
python inference.py --mode demo --policy sjf --difficulty medium

# 4. Q-learning training + comparison vs baselines
python inference.py --mode train --difficulty medium --train-eps 400

# 5. Text Gantt chart visualisation
python inference.py --mode gantt --policy priority --difficulty hard
```

### Docker

```bash
# Build
docker build -t cpu-scheduler-rl .

# Run benchmark
docker run cpu-scheduler-rl

# Run training
docker run cpu-scheduler-rl --mode train --difficulty hard --train-eps 300
```

---

## 🗂️ Project Structure

```
/project
├── cpu_env.py       ← OpenEnv environment + policies (dataclass models)
├── inference.py     ← CLI runner: benchmark, demo, train, gantt
├── openenv.yaml     ← OpenEnv metadata & space specification
├── requirements.txt ← Python dependencies
├── Dockerfile       ← Containerised deployment
└── README.md        ← This file
```

---

## 🔮 Future Improvements

| Improvement | Description |
|-------------|-------------|
| **DQN Agent** | Replace tabular Q-table with a neural network for continuous state spaces |
| **Multi-core** | Extend to multiple CPU cores (parallel action space) |
| **Memory & I/O** | Add memory and I/O burst simulation for realistic workloads |
| **Context switching** | Model preemption overhead as a real cost |
| **Matplotlib Gantt** | Rich visual Gantt chart with colour-coded priorities |
| **REST API** | Expose the environment as an HTTP service for distributed training |
| **PPO / A3C** | Train with policy-gradient methods for better sample efficiency |

---

## 📚 References

- Silberschatz, A. et al. *Operating System Concepts* (10th ed.)
- Sutton & Barto — *Reinforcement Learning: An Introduction* (2nd ed.)
- OpenAI Gym API design principles

---

## 📄 License

MIT © Mohtra AI
