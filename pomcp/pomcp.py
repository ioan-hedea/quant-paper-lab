"""
POMCP — Partially Observable Monte Carlo Planning
===================================================
Generic, reusable implementation.

Define your problem by subclassing POMDP and implementing:
  - sample_initial_state() -> Any
  - step(state, action)    -> (next_state, obs: int, reward: float)
  - N_ACTIONS: int          (class attribute)
  - N_OBS:     int          (class attribute)

Then pass an instance to the POMCP planner.
"""

from __future__ import annotations

import math
import random
from abc import ABC, abstractmethod
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Tuple


# ---------------------------------------------------------------------------
# 1. POMDP interface
# ---------------------------------------------------------------------------

class POMDP(ABC):
    """
    Abstract generative model G(s, a) -> (s', o, r).

    Subclass this for each problem. Set N_ACTIONS and N_OBS as class
    attributes; implement sample_initial_state and step.
    """

    N_ACTIONS: int   # number of discrete actions
    N_OBS:     int   # number of discrete observations

    @abstractmethod
    def sample_initial_state(self) -> Any:
        """Sample a state from the initial distribution."""

    @abstractmethod
    def step(self, state: Any, action: int) -> Tuple[Any, int, float]:
        """
        Black-box generative model.
        Returns (next_state, observation, reward).
        """


# ---------------------------------------------------------------------------
# 2. Tree node  — one history h in the search tree
# ---------------------------------------------------------------------------

@dataclass
class TreeNode:
    """
    History node storing Q-value estimates, visit counts, and particles.

    visit_count    : N(h)
    action_counts  : N(h, a) per action
    action_values  : Q(h, a) running mean per action
    particles      : states that reached h (implicit belief)
    children       : (action, obs) -> child TreeNode
    """
    visit_count:   int  = 0
    action_counts: dict = field(default_factory=lambda: defaultdict(int))
    action_values: dict = field(default_factory=lambda: defaultdict(float))
    particles:     list = field(default_factory=list)
    children:      dict = field(default_factory=dict)


# ---------------------------------------------------------------------------
# 3. POMCP planner
# ---------------------------------------------------------------------------

class POMCP:
    """
    POMCP planner for any POMDP with a black-box generative model.

    Parameters
    ----------
    env           : POMDP   — generative model
    n_sims        : int     — simulations per real step
    max_depth     : int     — max tree depth per simulation
    ucb_c         : float   — UCB exploration constant
    gamma         : float   — discount factor
    n_particles   : int     — root belief particle count
    rollout_depth : int     — random rollout length beyond tree frontier
    """

    def __init__(
        self,
        env:           POMDP,
        n_sims:        int   = 500,
        max_depth:     int   = 20,
        ucb_c:         float = 100.0,
        gamma:         float = 0.95,
        n_particles:   int   = 500,
        rollout_depth: int   = 10,
    ):
        self.env           = env
        self.n_sims        = n_sims
        self.max_depth     = max_depth
        self.ucb_c         = ucb_c
        self.gamma         = gamma
        self.n_particles   = n_particles
        self.rollout_depth = rollout_depth

        self.belief_particles: list = [
            env.sample_initial_state() for _ in range(n_particles)
        ]
        self.root = TreeNode()
        self.root.particles = list(self.belief_particles)

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def plan(self) -> int:
        """
        Run n_sims simulations from the current root belief.
        Returns the greedy-best action (no UCB bonus at execution time).
        """
        for _ in range(self.n_sims):
            s = random.choice(self.belief_particles)
            self._simulate(s, depth=0, node=self.root)
        return self._best_action(self.root)

    def update_belief(self, action: int, observation: int) -> None:
        """
        Rejection-sampling belief update after taking action and receiving
        observation. Re-roots the tree at the corresponding child node.
        """
        new_particles = []
        attempts      = 0
        max_attempts  = self.n_particles * 200

        while len(new_particles) < self.n_particles and attempts < max_attempts:
            s = random.choice(self.belief_particles)
            s_next, o_sim, _ = self.env.step(s, action)
            if o_sim == observation:
                new_particles.append(s_next)
            attempts += 1

        # Particle depletion fallbacks
        if len(new_particles) == 0:
            new_particles = [self.env.sample_initial_state()
                             for _ in range(self.n_particles)]
        elif len(new_particles) < self.n_particles:
            deficit = self.n_particles - len(new_particles)
            new_particles += random.choices(new_particles, k=deficit)

        self.belief_particles = new_particles

        # Re-root
        key = (action, observation)
        if key in self.root.children:
            self.root = self.root.children[key]
            self.root.particles = list(new_particles)
        else:
            self.root = TreeNode()
            self.root.particles = list(new_particles)

    def q_summary(self, action_names: dict | None = None) -> str:
        """Q-values and visit counts for all actions at root."""
        lines = []
        for a in range(self.env.N_ACTIONS):
            q    = self.root.action_values[a]
            n    = self.root.action_counts[a]
            name = action_names[a] if action_names else str(a)
            lines.append(f"  {name:<14}  Q={q:+7.2f}  N={n}")
        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Core POMCP recursion
    # ------------------------------------------------------------------

    def _simulate(self, s: Any, depth: int, node: TreeNode) -> float:
        """
        SIMULATE(s, depth, node) — recursive tree search.
        Returns the discounted return estimate from this node.
        """
        if depth >= self.max_depth:
            return 0.0

        action              = self._ucb_action(node)
        s_next, obs, reward = self.env.step(s, action)

        key = (action, obs)
        if key in node.children:
            child  = node.children[key]
            future = self._simulate(s_next, depth + 1, child)
        else:
            child  = TreeNode()
            node.children[key] = child
            future = self._rollout(s_next, depth + 1)

        child.particles.append(s_next)

        G = reward + self.gamma * future

        node.visit_count           += 1
        node.action_counts[action] += 1
        n = node.action_counts[action]
        node.action_values[action] += (G - node.action_values[action]) / n

        return G

    def _rollout(self, s: Any, depth: int) -> float:
        """Random rollout from state s beyond the tree frontier."""
        total  = 0.0
        factor = 1.0
        for _ in range(self.rollout_depth):
            if depth >= self.max_depth:
                break
            action       = random.randint(0, self.env.N_ACTIONS - 1)
            s, _, reward = self.env.step(s, action)
            total       += factor * reward
            factor      *= self.gamma
            depth       += 1
        return total

    # ------------------------------------------------------------------
    # Action selection helpers
    # ------------------------------------------------------------------

    def _ucb_action(self, node: TreeNode) -> int:
        """
        UCB1: prefer unvisited actions first, then
        argmax [ Q(h,a) + c * sqrt(ln N(h) / N(h,a)) ].
        """
        unvisited = [a for a in range(self.env.N_ACTIONS)
                     if node.action_counts[a] == 0]
        if unvisited:
            return random.choice(unvisited)

        log_n      = math.log(node.visit_count)
        best_score = -math.inf
        best_a     = 0
        for a in range(self.env.N_ACTIONS):
            score = (node.action_values[a]
                     + self.ucb_c * math.sqrt(log_n / node.action_counts[a]))
            if score > best_score:
                best_score = score
                best_a     = a
        return best_a

    def _best_action(self, node: TreeNode) -> int:
        """Greedy action: argmax_a Q(h, a)."""
        return max(range(self.env.N_ACTIONS),
                   key=lambda a: node.action_values[a])
