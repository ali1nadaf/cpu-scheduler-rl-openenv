"""
inference.py — CPU Scheduler Benchmark, RL Training & Visualisation
====================================================================
Entry point for the OpenEnv CPU Scheduler project.

Usage
-----
    python inference.py                          # full benchmark (all policies × all levels)
    python inference.py --mode demo              # verbose single-episode walkthrough
    python inference.py --mode train             # Q-learning training on medium
    python inference.py --mode train --difficulty hard --episodes 600
    python inference.py --mode benchmark --episodes 50
    python inference.py --mode gantt             # text Gantt chart demo
    python inference.py --policy sjf --difficulty hard --mode demo
"""

from __future__ import annotations

import argparse
import random
import sys
from collections import defaultdict
from typing import Callable, Optional

from cpu_env import (
    CPUSchedulerEnv,
    policy_fcfs,
    policy_sjf,
    policy_priority,
    policy_round_robin,
    policy_random,
)

# ── terminal colours (gracefully disabled if not supported) ──────────────────
try:
    from colorama import Fore, Style, init as _cinit
    _cinit(autoreset=True)
    C = {
        "cyan":    Fore.CYAN,
        "green":   Fore.GREEN,
        "yellow":  Fore.YELLOW,
        "red":     Fore.RED,
        "magenta": Fore.MAGENTA,
        "bold":    Style.BRIGHT,
        "reset":   Style.RESET_ALL,
    }
except ImportError:
    C = defaultdict(str)


# ══════════════════════════════════════════════════════════════════════════════
#  Pretty-print helpers
# ══════════════════════════════════════════════════════════════════════════════

W = 70

def _sep(char="─"):  print(char * W)
def _dsep():         print("═" * W)

def _header(title: str):
    _dsep()
    print(f"{C['bold']}{C['cyan']}  {title}{C['reset']}")
    _dsep()

def _section(title: str):
    print()
    _sep()
    print(f"{C['bold']}  {title}{C['reset']}")
    _sep()


# ══════════════════════════════════════════════════════════════════════════════
#  Episode runner
# ══════════════════════════════════════════════════════════════════════════════

def run_episode(
    env: CPUSchedulerEnv,
    policy_fn: Callable,
    max_steps: int = 300,
    verbose: bool = False,
) -> dict:
    state        = env.reset()
    total_reward = 0.0
    steps        = 0
    ep_stats     = {}

    for step in range(max_steps):
        if verbose:
            q = state["ready_queue"]
            print(f"\n  {C['yellow']}── t={state['time']:>3}  "
                  f"ready={state['queue_length']}  "
                  f"pending={state['pending_arrivals']}  "
                  f"done={state['completed_count']}{C['reset']}")
            if q:
                hdr = f"  {'PID':>4}  {'rem':>4}  {'pri':>4}  {'wait':>5}  {'arr':>4}"
                print(hdr)
                for p in q:
                    print(f"  {p['id']:>4}  {p['remaining']:>4}  "
                          f"{p['priority']:>4}  {p['waiting']:>5}  {p['arrival']:>4}")
            else:
                print("   (queue empty)")

        action = policy_fn(state)

        if verbose:
            if action is not None and action < len(state["ready_queue"]):
                pid = state["ready_queue"][action]["id"]
                print(f"  {C['green']}→ Schedule PID {pid}  (action={action}){C['reset']}")
            else:
                print(f"  {C['red']}→ IDLE{C['reset']}")

        state, reward, done, info = env.step(action)
        total_reward += reward
        steps        += 1

        if verbose:
            ev = info.get("event", "")
            print(f"  reward={reward:+.2f}  event={ev}")

        if done:
            ep_stats = info.get("episode_stats", {})
            if verbose:
                print(f"\n  {C['cyan']}✓ Episode complete in {steps} steps{C['reset']}")
            break

    return {"total_reward": round(total_reward, 2), "steps": steps, **ep_stats}


# ══════════════════════════════════════════════════════════════════════════════
#  Benchmark
# ══════════════════════════════════════════════════════════════════════════════

def _benchmark_policy(
    name: str,
    policy_fn: Callable,
    difficulty: str,
    n_episodes: int,
) -> dict:
    results = []
    for seed in range(n_episodes):
        env   = CPUSchedulerEnv(difficulty=difficulty, seed=seed)
        stats = run_episode(env, policy_fn)
        results.append(stats)

    def avg(key):
        vals = [r[key] for r in results if key in r]
        return round(sum(vals) / len(vals), 2) if vals else 0.0

    return {
        "policy":         name,
        "difficulty":     difficulty,
        "avg_reward":     avg("total_reward"),
        "avg_steps":      avg("steps"),
        "avg_turnaround": avg("avg_turnaround"),
        "avg_waiting":    avg("avg_waiting_time"),
        "avg_response":   avg("avg_response_time"),
        "avg_util_%":     avg("cpu_utilization"),
        "avg_throughput": avg("throughput"),
    }


def _make_policies() -> dict[str, Callable]:
    rr = [-1]
    return {
        "FCFS":       policy_fcfs,
        "SJF":        policy_sjf,
        "Priority":   policy_priority,
        "RoundRobin": lambda s: policy_round_robin(s, rr),
        "Random":     policy_random,
    }


def run_benchmark(n_episodes: int = 30):
    _header("CPU SCHEDULER — POLICY BENCHMARK")

    col = f"{'Policy':<12}  {'Reward':>8}  {'Wait':>6}  {'Turn':>6}  {'Util%':>6}  {'Thru':>5}"
    best_by_level: dict[str, list] = {}

    for difficulty in ("easy", "medium", "hard"):
        _section(f"Difficulty: {difficulty.upper()}")
        policies   = _make_policies()
        level_rows = []

        print(f"  {col}")
        _sep()
        for name, fn in policies.items():
            r = _benchmark_policy(name, fn, difficulty, n_episodes)
            level_rows.append(r)
            flag = ""
            print(
                f"  {name:<12}  {r['avg_reward']:>8.1f}  "
                f"{r['avg_waiting']:>6.1f}  {r['avg_turnaround']:>6.1f}  "
                f"{r['avg_util_%']:>6.1f}  {r['avg_throughput']:>5.1f}{flag}"
            )

        best_r = max(level_rows, key=lambda x: x["avg_reward"])
        best_w = min(level_rows, key=lambda x: x["avg_waiting"])
        best_u = max(level_rows, key=lambda x: x["avg_util_%"])
        best_by_level[difficulty] = level_rows

        print(f"\n  {C['green']}🏆 Best reward    → {best_r['policy']}{C['reset']}")
        print(f"  {C['cyan']}⏱  Least wait     → {best_w['policy']}{C['reset']}")
        print(f"  {C['yellow']}⚡ Best CPU util   → {best_u['policy']}{C['reset']}")

    print()
    _dsep()
    print(f"{C['bold']}  Benchmark complete  ({n_episodes} episodes/policy){C['reset']}")
    _dsep()
    return best_by_level


# ══════════════════════════════════════════════════════════════════════════════
#  Q-Learning agent
# ══════════════════════════════════════════════════════════════════════════════

class QLearningAgent:
    """
    Tabular Q-Learning for the CPU Scheduler.

    State key is a compact discretisation:
        (queue_length, min_remaining, max_waiting, max_priority)

    For larger state spaces a DQN would be used; this illustrates the RL loop
    end-to-end with reproducible, readable results.
    """

    def __init__(
        self,
        alpha:       float = 0.15,
        gamma:       float = 0.97,
        epsilon:     float = 1.0,
        eps_min:     float = 0.05,
        eps_decay:   float = 0.997,
    ):
        self.alpha     = alpha
        self.gamma     = gamma
        self.epsilon   = epsilon
        self.eps_min   = eps_min
        self.eps_decay = eps_decay
        self.q: dict[tuple, dict[int, float]] = defaultdict(lambda: defaultdict(float))
        self._rng = random.Random(0)

    def _key(self, state: dict) -> tuple:
        q = state["ready_queue"]
        if not q:
            return (0, 0, 0, 0)
        return (
            min(len(q), 8),
            min(min(p["remaining"] for p in q), 10),
            min(max(p["waiting"]   for p in q), 20),
            max(p["priority"]      for p in q),
        )

    def act(self, state: dict) -> Optional[int]:
        q = state["ready_queue"]
        if not q:
            return None
        n = len(q)
        if self._rng.random() < self.epsilon:
            return self._rng.randint(0, n - 1)
        key   = self._key(state)
        qvals = {a: self.q[key][a] for a in range(n)}
        return max(qvals, key=qvals.get)

    def update(self, s: dict, a: Optional[int], r: float, s2: dict):
        if a is None:
            return
        key  = self._key(s)
        key2 = self._key(s2)
        n2   = s2["queue_length"]
        best_next = max((self.q[key2][a2] for a2 in range(n2)), default=0.0)
        td_target = r + self.gamma * best_next
        self.q[key][a] += self.alpha * (td_target - self.q[key][a])

    def decay(self):
        self.epsilon = max(self.eps_min, self.epsilon * self.eps_decay)

    def greedy_act(self, state: dict) -> Optional[int]:
        """Act greedily (no exploration) — for evaluation."""
        old, self.epsilon = self.epsilon, 0.0
        action = self.act(state)
        self.epsilon = old
        return action


def train_q_agent(
    difficulty:   str = "medium",
    n_episodes:   int = 500,
    max_steps:    int = 300,
    eval_every:   int = 100,
    eval_eps:     int = 20,
    seed:         int = 42,
) -> QLearningAgent:
    _header(f"Q-LEARNING TRAINING  |  {difficulty.upper()}  |  {n_episodes} episodes")

    agent      = QLearningAgent()
    train_hist: list[float] = []

    for ep in range(1, n_episodes + 1):
        env   = CPUSchedulerEnv(difficulty=difficulty, seed=ep + seed)
        state = env.reset()
        total = 0.0

        for _ in range(max_steps):
            action            = agent.act(state)
            s2, r, done, _    = env.step(action)
            agent.update(state, action, r, s2)
            state  = s2
            total += r
            if done:
                break

        agent.decay()
        train_hist.append(total)

        if ep % eval_every == 0:
            greedy_rewards = []
            for eseed in range(eval_eps):
                env2  = CPUSchedulerEnv(difficulty=difficulty, seed=9000 + eseed)
                s2    = env2.reset()
                gr    = 0.0
                for _ in range(max_steps):
                    a2         = agent.greedy_act(s2)
                    s2, r2, d2, _ = env2.step(a2)
                    gr += r2
                    if d2:
                        break
                greedy_rewards.append(gr)

            avg_gr   = sum(greedy_rewards) / len(greedy_rewards)
            avg_tr   = sum(train_hist[-eval_every:]) / eval_every
            qtable_n = len(agent.q)
            print(
                f"  Ep {ep:>4}/{n_episodes}  "
                f"ε={agent.epsilon:.3f}  "
                f"train_avg={avg_tr:>8.1f}  "
                f"eval_avg={avg_gr:>8.1f}  "
                f"Q-states={qtable_n}"
            )

    print(f"\n  {C['green']}✓ Training complete{C['reset']}  "
          f"(final ε={agent.epsilon:.4f}, Q-table size={len(agent.q)})")
    return agent


def compare_agent_vs_baselines(
    agent: QLearningAgent,
    difficulty: str = "medium",
    n_episodes: int = 30,
):
    _section(f"Q-Agent vs Baselines  |  {difficulty.upper()}  |  {n_episodes} eps")

    def agent_policy(s):
        return agent.greedy_act(s)

    contenders = {
        "Q-Agent":  agent_policy,
        "SJF":      policy_sjf,
        "Priority": policy_priority,
        "FCFS":     policy_fcfs,
    }

    rows = []
    for name, fn in contenders.items():
        r = _benchmark_policy(name, fn, difficulty, n_episodes)
        rows.append(r)
        print(
            f"  {name:<12}  reward={r['avg_reward']:>8.1f}  "
            f"wait={r['avg_waiting']:>5.1f}  "
            f"util={r['avg_util_%']:>5.1f}%  "
            f"turn={r['avg_turnaround']:>5.1f}"
        )

    winner = max(rows, key=lambda x: x["avg_reward"])
    print(f"\n  {C['green']}🏆 Best overall: {winner['policy']}{C['reset']}")


# ══════════════════════════════════════════════════════════════════════════════
#  Text Gantt chart
# ══════════════════════════════════════════════════════════════════════════════

def _text_gantt(gantt: list[dict], max_width: int = 60):
    """Render a compact text-based Gantt chart."""
    if not gantt:
        print("  (no Gantt data)")
        return

    pids = sorted({g["pid"] for g in gantt})
    tick_map = {g["tick"]: g["pid"] for g in gantt}
    ticks    = sorted(tick_map)
    total    = len(ticks)

    # compress if too wide
    step = max(1, total // max_width)

    print(f"\n  {'PID':<8}", end="")
    for t in ticks[::step]:
        print(f"{t:<3}", end="")
    print()
    _sep()

    for pid in pids:
        label = f"IDLE" if pid == "IDLE" else f"P{pid}"
        print(f"  {label:<8}", end="")
        for t in ticks[::step]:
            cell = tick_map.get(t, "IDLE")
            sym  = "█" if cell == pid else "·"
            print(f"{sym:<3}", end="")
        print()
    print()


def run_gantt_demo(difficulty: str = "medium", policy_name: str = "sjf"):
    _header(f"GANTT CHART DEMO  |  policy={policy_name.upper()}  |  {difficulty.upper()}")

    fn_map = {
        "fcfs":     policy_fcfs,
        "sjf":      policy_sjf,
        "priority": policy_priority,
        "random":   policy_random,
    }
    fn  = fn_map.get(policy_name.lower(), policy_sjf)
    env = CPUSchedulerEnv(difficulty=difficulty, seed=7)

    state        = env.reset()
    total_reward = 0.0

    while True:
        action            = fn(state)
        state, r, done, _ = env.step(action)
        total_reward      += r
        if done:
            break

    gantt = env.get_gantt()
    _text_gantt(gantt)
    stats = env._episode_stats()
    print(f"  Total reward:     {total_reward:.1f}")
    print(f"  Avg turnaround:   {stats.avg_turnaround}")
    print(f"  Avg waiting time: {stats.avg_waiting_time}")
    print(f"  CPU utilisation:  {stats.cpu_utilization}%")


# ══════════════════════════════════════════════════════════════════════════════
#  Verbose demo episode
# ══════════════════════════════════════════════════════════════════════════════

def run_demo(policy_name: str = "sjf", difficulty: str = "medium"):
    fn_map = {
        "fcfs":     policy_fcfs,
        "sjf":      policy_sjf,
        "priority": policy_priority,
        "random":   policy_random,
    }
    fn = fn_map.get(policy_name.lower(), policy_sjf)
    _header(f"SINGLE EPISODE DEMO  |  {policy_name.upper()}  |  {difficulty.upper()}")

    env   = CPUSchedulerEnv(difficulty=difficulty, seed=42)
    stats = run_episode(env, fn, verbose=True)

    _section("Episode Summary")
    for k, v in stats.items():
        print(f"  {k:<26} {v}")


# ══════════════════════════════════════════════════════════════════════════════
#  CLI
# ══════════════════════════════════════════════════════════════════════════════

def _parse_args():
    p = argparse.ArgumentParser(
        description="CPU Scheduler RL — OpenEnv benchmark & training",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    p.add_argument(
        "--mode", default="benchmark",
        choices=["benchmark", "demo", "train", "gantt"],
        help=(
            "benchmark : run all policies across all levels\n"
            "demo      : verbose single-episode walkthrough\n"
            "train     : Q-learning training + comparison\n"
            "gantt     : text Gantt chart visualisation"
        ),
    )
    p.add_argument("--difficulty", default="medium",
                   choices=["easy", "medium", "hard"])
    p.add_argument("--policy",    default="sjf",
                   choices=["fcfs", "sjf", "priority", "random"],
                   help="Policy for demo/gantt modes")
    p.add_argument("--episodes",  type=int, default=30,
                   help="Episodes per policy in benchmark mode")
    p.add_argument("--train-eps", type=int, default=400,
                   help="Training episodes for Q-learning")
    p.add_argument("--seed",      type=int, default=42)
    return p.parse_args()


def main():
    args = _parse_args()
    random.seed(args.seed)

    if args.mode == "benchmark":
        run_benchmark(n_episodes=args.episodes)

    elif args.mode == "demo":
        run_demo(policy_name=args.policy, difficulty=args.difficulty)

    elif args.mode == "train":
        agent = train_q_agent(
            difficulty=args.difficulty,
            n_episodes=args.train_eps,
            seed=args.seed,
        )
        compare_agent_vs_baselines(agent, difficulty=args.difficulty)

    elif args.mode == "gantt":
        run_gantt_demo(difficulty=args.difficulty, policy_name=args.policy)

    else:
        print("Unknown mode.", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
