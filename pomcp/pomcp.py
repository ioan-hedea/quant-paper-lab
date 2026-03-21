"""
POMCP — Partially Observable Monte Carlo Planning
===================================================
Full implementation on the Tiger POMDP benchmark.

Structure
---------
  TigerPOMDP        — the environment (generative model G)
  TreeNode          — history node storing Q-values, counts, particles
  POMCP             — the planner (simulate, rollout, UCB, belief update)
  main()            — runs one full episode and prints a trace

The Tiger problem
-----------------
  States  : tiger_left, tiger_right
  Actions : listen, open_left, open_right
  Obs     : hear_left, hear_right
  Rewards :
    listen                          → -1   (always)
    open correct door (no tiger)    → +10
    open wrong door   (tiger)       → -100
  Obs model:
    listen  → hear correct side with prob 0.85, wrong side 0.15
    open_*  → hear_left / hear_right each with prob 0.5 (reset, uninformative)
  Transition:
    open_*  → tiger resets to left or right uniformly (new episode)
    listen  → tiger stays in place
"""

from __future__ import annotations

import math
import random
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Optional


# ---------------------------------------------------------------------------
# 1. ENVIRONMENT — the generative model G(s, a) → (s', o, r)
# ---------------------------------------------------------------------------

TIGER_LEFT  = 0
TIGER_RIGHT = 1

LISTEN      = 0
OPEN_LEFT   = 1
OPEN_RIGHT  = 2

HEAR_LEFT   = 0
HEAR_RIGHT  = 1

ACTION_NAMES = {LISTEN: "listen    ", OPEN_LEFT: "open_left ", OPEN_RIGHT: "open_right"}
OBS_NAMES    = {HEAR_LEFT: "hear_left ", HEAR_RIGHT: "hear_right"}
STATE_NAMES  = {TIGER_LEFT: "tiger_LEFT ", TIGER_RIGHT: "tiger_RIGHT"}


class TigerPOMDP:
    """
    Generative model for the Tiger POMDP.

    G(state, action) -> (next_state, observation, reward)

    After open_left or open_right the tiger is reset uniformly —
    this models the start of a new "sub-episode" within one long episode.
    """

    CORRECT_OBS_PROB = 0.85   # P(hear correct side | listen)
    N_STATES  = 2
    N_ACTIONS = 3
    N_OBS     = 2

    def sample_initial_state(self) -> int:
        """Sample a state from the uniform initial distribution."""
        return random.randint(0, 1)

    def step(self, state: int, action: int) -> tuple[int, int, float]:
        """
        Sample one transition.
        Returns (next_state, observation, reward).
        """
        if action == LISTEN:
            next_state = state   # tiger does not move
            obs = self._listen_obs(state)
            reward = -1.0

        elif action == OPEN_LEFT:
            reward = -100.0 if state == TIGER_LEFT else +10.0
            next_state = random.randint(0, 1)   # reset
            obs = random.randint(0, 1)           # uninformative after reset

        else:  # OPEN_RIGHT
            reward = -100.0 if state == TIGER_RIGHT else +10.0
            next_state = random.randint(0, 1)
            obs = random.randint(0, 1)

        return next_state, obs, reward

    def _listen_obs(self, state: int) -> int:
        """
        When listening:
          P(hear_left  | tiger_left)  = 0.85
          P(hear_right | tiger_right) = 0.85
        """
        if random.random() < self.CORRECT_OBS_PROB:
            return HEAR_LEFT if state == TIGER_LEFT else HEAR_RIGHT
        else:
            return HEAR_RIGHT if state == TIGER_LEFT else HEAR_LEFT


# ---------------------------------------------------------------------------
# 2. TREE NODE — stores statistics and particles for one history node h
# ---------------------------------------------------------------------------

@dataclass
class TreeNode:
    """
    One node in the POMCP search tree, corresponding to a history h.

    Attributes
    ----------
    visit_count : int
        N(h) — total times this node was visited across all simulations.
    action_counts : dict[int, int]
        N(h, a) — per-action visit counts.
    action_values : dict[int, float]
        Q(h, a) — running mean return estimate per action.
    particles : list[int]
        States that reached this node (implicit belief representation).
    children : dict[tuple[int,int], TreeNode]
        Map from (action, observation) to child history node.
    """
    visit_count:    int                          = 0
    action_counts:  dict = field(default_factory=lambda: defaultdict(int))
    action_values:  dict = field(default_factory=lambda: defaultdict(float))
    particles:      list = field(default_factory=list)
    children:       dict = field(default_factory=dict)


# ---------------------------------------------------------------------------
# 3. POMCP PLANNER
# ---------------------------------------------------------------------------

class POMCP:
    """
    POMCP planner for a POMDP with a black-box generative model.

    Parameters
    ----------
    env        : TigerPOMDP  — the generative model
    n_sims     : int         — number of SIMULATE calls per real step
    max_depth  : int         — maximum tree depth per simulation
    ucb_c      : float       — exploration constant c in UCB
    gamma      : float       — discount factor
    n_particles: int         — number of root particles (belief size)
    rollout_depth : int      — max steps in rollout (beyond tree)
    """

    def __init__(
        self,
        env:          TigerPOMDP,
        n_sims:       int   = 500,
        max_depth:    int   = 20,
        ucb_c:        float = 100.0,
        gamma:        float = 0.95,
        n_particles:  int   = 500,
        rollout_depth: int  = 10,
    ):
        self.env           = env
        self.n_sims        = n_sims
        self.max_depth     = max_depth
        self.ucb_c         = ucb_c
        self.gamma         = gamma
        self.n_particles   = n_particles
        self.rollout_depth = rollout_depth

        # Root belief: uniform over states, represented as particles
        self.belief_particles: list[int] = [
            env.sample_initial_state() for _ in range(n_particles)
        ]
        self.root: TreeNode = TreeNode()

        # Add initial particles to root
        self.root.particles = list(self.belief_particles)

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def plan(self) -> int:
        """
        Run n_sims simulations from the current root belief.
        Returns the best action (greedy on Q, not UCB).
        """
        for _ in range(self.n_sims):
            # Sample one root particle — this is Thompson sampling
            s = random.choice(self.belief_particles)
            self._simulate(s, depth=0, node=self.root)

        return self._best_action(self.root)

    def update_belief(self, action: int, observation: int) -> None:
        """
        Execute a real action, receive a real observation.
        Update the belief via rejection sampling and re-root the tree.
        """
        # --- Rejection sampling ---
        new_particles = []
        attempts      = 0
        max_attempts  = self.n_particles * 200   # safety ceiling

        while len(new_particles) < self.n_particles and attempts < max_attempts:
            s = random.choice(self.belief_particles)
            _, o, _ = self.env.step(s, action)
            # Propagate s forward and check if simulated obs matches real obs
            s_next, o_sim, _ = self.env.step(s, action)
            if o_sim == observation:
                new_particles.append(s_next)
            attempts += 1

        # If rejection sampling finds too few, fill remainder by resampling
        # (handles low-prob observations — particle depletion fallback)
        if len(new_particles) == 0:
            # Total depletion: reinitialise uniformly (last resort)
            new_particles = [self.env.sample_initial_state()
                             for _ in range(self.n_particles)]
        elif len(new_particles) < self.n_particles:
            deficit = self.n_particles - len(new_particles)
            new_particles += random.choices(new_particles, k=deficit)

        self.belief_particles = new_particles

        # --- Re-root the tree at child node (action, observation) ---
        key = (action, observation)
        if key in self.root.children:
            self.root = self.root.children[key]
            # Replenish child particles with updated belief
            self.root.particles = list(new_particles)
        else:
            # Child node was never visited — create fresh root
            self.root = TreeNode()
            self.root.particles = list(new_particles)

    # ------------------------------------------------------------------
    # Core POMCP recursion
    # ------------------------------------------------------------------

    def _simulate(self, s: int, depth: int, node: TreeNode) -> float:
        """
        Recursive SIMULATE(s, depth, node).

        s     : state sample propagated from parent
        depth : current depth in the tree
        node  : current history node h

        Returns the discounted return estimate from this node.
        """
        # --- Base case ---
        if depth >= self.max_depth:
            return 0.0

        # --- Select action via UCB ---
        action = self._ucb_action(node)

        # --- Step through generative model ---
        s_next, obs, reward = self.env.step(s, action)

        # --- Recurse or rollout ---
        key = (action, obs)
        if key in node.children:
            # Tree node exists — recurse deeper
            child = node.children[key]
            future = self._simulate(s_next, depth + 1, child)
        else:
            # Leaf — expand one new node, then rollout
            child = TreeNode()
            node.children[key] = child
            future = self._rollout(s_next, depth + 1)

        # --- Accumulate particle at child ---
        child.particles.append(s_next)

        # --- Back up return ---
        G = reward + self.gamma * future

        # --- Update statistics at this node (incremental mean) ---
        node.visit_count             += 1
        node.action_counts[action]   += 1
        n = node.action_counts[action]
        # Q(h,a) ← Q(h,a) + (G − Q(h,a)) / N(h,a)
        node.action_values[action]   += (G - node.action_values[action]) / n

        return G

    def _rollout(self, s: int, depth: int) -> float:
        """
        Random rollout from state s beyond the tree frontier.
        Uses a random policy for rollout_depth steps.
        """
        total  = 0.0
        factor = 1.0

        for _ in range(self.rollout_depth):
            if depth >= self.max_depth:
                break
            action          = random.randint(0, self.env.N_ACTIONS - 1)
            s, _, reward    = self.env.step(s, action)
            total          += factor * reward
            factor         *= self.gamma
            depth          += 1

        return total

    # ------------------------------------------------------------------
    # Action selection
    # ------------------------------------------------------------------

    def _ucb_action(self, node: TreeNode) -> int:
        """
        UCB1 action selection.

        If any action is unvisited, select it first (N=0 → infinite UCB).
        Otherwise: a* = argmax [ Q(h,a) + c * sqrt(ln N(h) / N(h,a)) ]
        """
        n_actions = self.env.N_ACTIONS

        # Unvisited actions first
        unvisited = [a for a in range(n_actions)
                     if node.action_counts[a] == 0]
        if unvisited:
            return random.choice(unvisited)

        log_n = math.log(node.visit_count)
        best_score = -math.inf
        best_action = 0

        for a in range(n_actions):
            q     = node.action_values[a]
            n_a   = node.action_counts[a]
            score = q + self.ucb_c * math.sqrt(log_n / n_a)
            if score > best_score:
                best_score  = score
                best_action = a

        return best_action

    def _best_action(self, node: TreeNode) -> int:
        """
        Greedy action: argmax Q(h, a) — no UCB bonus at execution time.
        """
        return max(range(self.env.N_ACTIONS),
                   key=lambda a: node.action_values[a])

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------

    def belief_summary(self) -> str:
        """
        Return a string showing the empirical belief distribution
        over states from the current particle set.
        """
        n     = len(self.belief_particles)
        p_L   = self.belief_particles.count(TIGER_LEFT)  / n
        p_R   = self.belief_particles.count(TIGER_RIGHT) / n
        return f"P(tiger_left)={p_L:.2f}  P(tiger_right)={p_R:.2f}"

    def q_summary(self) -> str:
        """Print Q-values and visit counts for all actions at root."""
        lines = []
        for a in range(self.env.N_ACTIONS):
            q = self.root.action_values[a]
            n = self.root.action_counts[a]
            lines.append(f"  {ACTION_NAMES[a]}  Q={q:+7.2f}  N={n}")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# 4. MAIN — run one episode and print a trace
# ---------------------------------------------------------------------------

def main():
    print("=" * 60)
    print("POMCP on the Tiger POMDP")
    print("=" * 60)

    random.seed(42)

    env   = TigerPOMDP()
    agent = POMCP(
        env,
        n_sims        = 1000,
        max_depth     = 20,
        ucb_c         = 100.0,
        gamma         = 0.95,
        n_particles   = 1000,
        rollout_depth = 10,
    )

    # Ground-truth initial state (hidden from agent)
    true_state    = env.sample_initial_state()
    total_reward  = 0.0
    n_steps       = 15

    print(f"\nTrue initial state: {STATE_NAMES[true_state]}")
    print(f"Agent's initial belief: {agent.belief_summary()}\n")
    print("-" * 60)

    for step in range(1, n_steps + 1):
        # --- Plan ---
        action = agent.plan()

        # --- Q-value summary before acting ---
        print(f"\nStep {step:2d} | True state: {STATE_NAMES[true_state]}")
        print(f"  Belief   : {agent.belief_summary()}")
        print(f"  Q-values :\n{agent.q_summary()}")
        print(f"  Chosen   : {ACTION_NAMES[action]}")

        # --- Execute in real environment ---
        true_state, obs, reward = env.step(true_state, action)
        total_reward += reward

        print(f"  Obs      : {OBS_NAMES[obs]}")
        print(f"  Reward   : {reward:+.1f}   (cumulative: {total_reward:+.1f})")

        # --- Update belief via rejection sampling + re-root ---
        agent.update_belief(action, obs)

    print("\n" + "=" * 60)
    print(f"Episode finished.  Total reward: {total_reward:+.1f}")
    print("=" * 60)


if __name__ == "__main__":
    main()