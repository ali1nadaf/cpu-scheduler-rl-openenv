"""
cpu_env.py — Production-Ready CPU Scheduler RL Environment
===========================================================
OpenEnv-compliant environment for CPU scheduling optimization.

API
---
    env = CPUSchedulerEnv(difficulty="medium", seed=42)
    state          = env.reset()                        # → dict
    state, r, done, info = env.step(action)             # → (dict, float, bool, dict)
    obs            = env.state()                        # → dict (current snapshot)

Difficulty Levels
-----------------
    easy   – 4 processes, t=0 arrivals, equal priority
    medium – 6 processes, staggered arrivals, varied priority
    hard   – 8 + dynamic mid-episode injections, complex constraints

Reward Components
-----------------
    -1.0   per step (base cost)
    -0.5 * waiting_time_penalty  (starvation)
    -3.0   idle tick
    -8.0   invalid action
    +10.0  process completion
    +2*p   priority bonus  (p = priority 1-5)
    +0..20 turnaround bonus (faster = higher)
    +1.0   partial progress (remaining reduced)
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field, asdict
from typing import Optional


# ══════════════════════════════════════════════════════════════════════════════
#  Typed data models
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class Process:
    """Internal representation of a CPU process (PCB-style)."""
    id:        int
    arrival:   int          # time step at which it enters the ready queue
    burst:     int          # total CPU time required
    remaining: int          # CPU time still needed
    priority:  int          # 1 (lowest) … 5 (highest)
    waiting:   int = 0      # cumulative time spent waiting
    start:     int = -1     # first tick it was scheduled  (-1 = never)
    finish:    int = -1     # tick it completed             (-1 = not done)

    def to_obs(self) -> dict:
        """Serialisable observation slice (no internal bookkeeping fields)."""
        return {
            "id":        self.id,
            "arrival":   self.arrival,
            "remaining": self.remaining,
            "priority":  self.priority,
            "waiting":   self.waiting,
        }

    @property
    def turnaround(self) -> int:
        if self.finish == -1:
            return -1
        return self.finish - self.arrival

    @property
    def response_time(self) -> int:
        if self.start == -1:
            return -1
        return self.start - self.arrival


@dataclass
class EpisodeStats:
    """Computed at episode end."""
    avg_turnaround:   float
    avg_waiting_time: float
    avg_response_time: float
    throughput:       int
    cpu_utilization:  float   # 0-100 %
    total_time:       int
    idle_ticks:       int

    def to_dict(self) -> dict:
        return asdict(self)


# ══════════════════════════════════════════════════════════════════════════════
#  Environment
# ══════════════════════════════════════════════════════════════════════════════

class CPUSchedulerEnv:
    """
    OpenEnv-compliant CPU scheduling RL environment.

    Parameters
    ----------
    difficulty : str
        One of "easy", "medium", "hard".
    seed : int | None
        Fixed seed for reproducibility.  None = non-deterministic.
    """

    LEVELS      = ("easy", "medium", "hard")
    REWARD_RANGE = (-100.0, 100.0)

    # ── init ──────────────────────────────────────────────────────────────────

    def __init__(self, difficulty: str = "medium", seed: Optional[int] = None):
        if difficulty not in self.LEVELS:
            raise ValueError(f"difficulty must be one of {self.LEVELS}")
        self.difficulty = difficulty
        self._seed      = seed
        self.rng        = random.Random(seed)

        self.ready_queue:  list[Process] = []
        self.future_queue: list[Process] = []
        self.completed:    list[Process] = []
        self.time:         int = 0
        self.idle_ticks:   int = 0
        self._next_pid:    int = 1
        self._inject_at:   list[int] = []
        self._gantt:       list[dict] = []   # Gantt chart trace

    # ── OpenEnv API ───────────────────────────────────────────────────────────

    def reset(self) -> dict:
        """Reset the environment and return the initial observation."""
        self.rng        = random.Random(self._seed)
        self.time       = 0
        self.idle_ticks = 0
        self._next_pid  = 1
        self._inject_at = []
        self._gantt     = []
        self.ready_queue  = []
        self.future_queue = []
        self.completed    = []

        for p in self._generate_processes():
            if p.arrival <= 0:
                self.ready_queue.append(p)
            else:
                self.future_queue.append(p)

        self.future_queue.sort(key=lambda p: p.arrival)

        if self.difficulty == "hard":
            self._schedule_injections()

        return self.state()

    def step(self, action: Optional[int]) -> tuple[dict, float, bool, dict]:
        """
        Execute one scheduling decision (one CPU time unit).

        Parameters
        ----------
        action : int | None
            Index into ready_queue.  None or out-of-range → idle tick.

        Returns
        -------
        (state, reward, done, info)
        """
        reward = 0.0
        info   = {"time": self.time}

        # ── tick waiting for non-selected processes ───────────────────────────
        for i, p in enumerate(self.ready_queue):
            if i != action:
                p.waiting += 1

        # ── execute action ────────────────────────────────────────────────────
        if not self.ready_queue or action is None:
            self.idle_ticks += 1
            reward          -= 3.0
            info["event"]    = "idle"
            self._gantt.append({"tick": self.time, "pid": "IDLE", "priority": 0})

        elif action >= len(self.ready_queue):
            self.idle_ticks += 1
            reward          -= 8.0
            info["event"]    = "invalid_action"
            self._gantt.append({"tick": self.time, "pid": "IDLE", "priority": 0})

        else:
            proc = self.ready_queue[action]

            if proc.start == -1:
                proc.start = self.time

            proc.remaining -= 1
            reward         -= 1.0                          # base step cost
            reward         += 0.5                          # partial progress signal

            # starvation penalty
            max_wait = max((p.waiting for p in self.ready_queue), default=0)
            reward  -= 0.05 * max_wait

            self._gantt.append({"tick": self.time, "pid": proc.id, "priority": proc.priority})

            if proc.remaining <= 0:
                proc.finish = self.time + 1
                reward     += 10.0                         # completion
                reward     += proc.priority * 2.0          # priority bonus
                turnaround  = proc.turnaround
                reward     += max(0.0, 20.0 - turnaround)  # turnaround bonus

                self.completed.append(proc)
                self.ready_queue.pop(action)

                info["event"]      = "completed"
                info["pid"]        = proc.id
                info["priority"]   = proc.priority
                info["turnaround"] = turnaround
                info["waiting"]    = proc.waiting
            else:
                info["event"]     = "running"
                info["pid"]       = proc.id
                info["remaining"] = proc.remaining

        # ── advance clock, admit new arrivals ─────────────────────────────────
        self.time += 1
        self._admit_arrivals()

        if self.difficulty == "hard":
            self._try_inject()

        done = (not self.ready_queue) and (not self.future_queue)

        if done:
            stats               = self._episode_stats()
            info["episode_stats"] = stats.to_dict()

        # clip reward to declared range
        reward = float(max(self.REWARD_RANGE[0], min(self.REWARD_RANGE[1], reward)))

        return self.state(), reward, done, info

    def state(self) -> dict:
        """Return the current serialisable observation."""
        return {
            "time":             self.time,
            "ready_queue":      [p.to_obs() for p in self.ready_queue],
            "queue_length":     len(self.ready_queue),
            "pending_arrivals": len(self.future_queue),
            "completed_count":  len(self.completed),
            "idle_ticks":       self.idle_ticks,
            "difficulty":       self.difficulty,
        }

    # ── process generation ────────────────────────────────────────────────────

    def _new_process(self, arrival: int, burst: int, priority: int) -> Process:
        p = Process(
            id=self._next_pid,
            arrival=arrival,
            burst=burst,
            remaining=burst,
            priority=priority,
        )
        self._next_pid += 1
        return p

    def _generate_processes(self) -> list[Process]:
        rng = self.rng

        if self.difficulty == "easy":
            return [self._new_process(0, rng.randint(2, 6), 3) for _ in range(4)]

        if self.difficulty == "medium":
            procs = []
            for _ in range(6):
                procs.append(self._new_process(
                    arrival=rng.randint(0, 6),
                    burst=rng.randint(2, 8),
                    priority=rng.randint(1, 5),
                ))
            return procs

        # hard
        procs = []
        for _ in range(8):
            procs.append(self._new_process(
                arrival=rng.randint(0, 5),
                burst=rng.randint(1, 10),
                priority=rng.randint(1, 5),
            ))
        return procs

    def _admit_arrivals(self):
        still_future = []
        for p in self.future_queue:
            if p.arrival <= self.time:
                self.ready_queue.append(p)
            else:
                still_future.append(p)
        self.future_queue = still_future

    # ── hard-mode dynamic injection ───────────────────────────────────────────

    def _schedule_injections(self):
        n = self.rng.randint(2, 3)
        self._inject_at = sorted(self.rng.randint(5, 20) for _ in range(n))

    def _try_inject(self):
        while self._inject_at and self.time == self._inject_at[0]:
            self._inject_at.pop(0)
            p = self._new_process(
                arrival=self.time,
                burst=self.rng.randint(2, 8),
                priority=self.rng.randint(1, 5),
            )
            self.ready_queue.append(p)

    # ── episode statistics ────────────────────────────────────────────────────

    def _episode_stats(self) -> EpisodeStats:
        if not self.completed:
            return EpisodeStats(0, 0, 0, 0, 0.0, self.time, self.idle_ticks)

        turnarounds    = [p.turnaround    for p in self.completed if p.turnaround    >= 0]
        waiting_times  = [p.waiting       for p in self.completed]
        response_times = [p.response_time for p in self.completed if p.response_time >= 0]

        def _avg(lst):
            return round(sum(lst) / len(lst), 2) if lst else 0.0

        return EpisodeStats(
            avg_turnaround    = _avg(turnarounds),
            avg_waiting_time  = _avg(waiting_times),
            avg_response_time = _avg(response_times),
            throughput        = len(self.completed),
            cpu_utilization   = round(
                (self.time - self.idle_ticks) / max(self.time, 1) * 100, 1
            ),
            total_time  = self.time,
            idle_ticks  = self.idle_ticks,
        )

    def get_gantt(self) -> list[dict]:
        """Return the Gantt chart trace (list of tick→pid mappings)."""
        return list(self._gantt)


# ══════════════════════════════════════════════════════════════════════════════
#  Benchmark policies
# ══════════════════════════════════════════════════════════════════════════════

def policy_fcfs(state: dict) -> Optional[int]:
    """First-Come-First-Served: earliest arrival wins."""
    q = state["ready_queue"]
    return None if not q else min(range(len(q)), key=lambda i: q[i]["arrival"])


def policy_sjf(state: dict) -> Optional[int]:
    """Shortest-Job-First: least remaining time wins."""
    q = state["ready_queue"]
    return None if not q else min(range(len(q)), key=lambda i: q[i]["remaining"])


def policy_priority(state: dict) -> Optional[int]:
    """Priority Scheduling: highest priority wins; SJF tie-break."""
    q = state["ready_queue"]
    return None if not q else max(
        range(len(q)),
        key=lambda i: (q[i]["priority"], -q[i]["remaining"])
    )


def policy_round_robin(state: dict, rr_ptr: list) -> Optional[int]:
    """
    Round-Robin: rotate index each call.
    Pass a single-element list as `rr_ptr` to preserve state across calls.
    """
    q = state["ready_queue"]
    if not q:
        return None
    idx        = (rr_ptr[0] + 1) % len(q)
    rr_ptr[0]  = idx
    return idx


def policy_random(state: dict) -> Optional[int]:
    """Random baseline."""
    q = state["ready_queue"]
    return None if not q else random.randint(0, len(q) - 1)
