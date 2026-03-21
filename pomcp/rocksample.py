"""
RockSample POMDP — runs on the generic POMCP planner.

RockSample(N=4, K=3):  4x4 grid, 3 rocks.

The agent starts at the middle-left cell and must navigate to the east
exit while collecting good rocks along the way.

States  : (row, col, rock_mask)
            row, col in [0, N-1]
            rock_mask: K-bit int, bit i=1 means rock i is still good
          Special terminal state: TERMINAL = (-1,-1,-1) after east exit

Actions : NORTH (0), EAST (1), SOUTH (2), WEST (3),
          SAMPLE (4),
          CHECK_0..CHECK_{K-1}  (5..4+K)

Obs     : NONE (0)  — after movement or sample
          GOOD (1)  — check action, rock is good
          BAD  (2)  — check action, rock is bad
          Sensor accuracy decreases with distance:
            accuracy(d) = (1 + 2^{-d/d_half}) / 2
            where d_half = N / 2

Rewards :
  sample good rock    -> +10  (rock becomes bad)
  sample bad rock     -> -10
  sample empty cell   -> -10
  east exit           -> +10  (episode over)
  all other actions   ->   0
"""

from __future__ import annotations

import math
import random
from collections import Counter
from pomcp import POMDP, POMCP


# ---------------------------------------------------------------------------
# Problem parameters
# ---------------------------------------------------------------------------

N      = 4                          # grid size (N x N)
K      = 3                          # number of rocks
D_HALF = N / 2.0                    # half-efficiency distance for sensor

# Rock positions: (row, col) — fixed layout
ROCK_POS = [(0, 1), (2, 3), (3, 0)]

# Terminal absorbing state (agent exited east)
TERMINAL = (-1, -1, -1)

# ---------------------------------------------------------------------------
# Action / observation constants
# ---------------------------------------------------------------------------

A_NORTH  = 0
A_EAST   = 1
A_SOUTH  = 2
A_WEST   = 3
A_SAMPLE = 4
# CHECK_i = 5 + i  (rock index i in [0, K-1])

ACTION_NAMES = {
    A_NORTH:  "NORTH  ",
    A_EAST:   "EAST   ",
    A_SOUTH:  "SOUTH  ",
    A_WEST:   "WEST   ",
    A_SAMPLE: "SAMPLE ",
    **{5 + i: f"CHECK_{i}" for i in range(K)},
}

OBS_NONE = 0
OBS_GOOD = 1
OBS_BAD  = 2

OBS_NAMES = {OBS_NONE: "none", OBS_GOOD: "GOOD", OBS_BAD: "BAD "}


# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------

class RockSamplePOMDP(POMDP):
    """
    Generative model for RockSample(N, K).
    States are (row, col, rock_mask) tuples or TERMINAL.
    """

    N_ACTIONS = 4 + K   # move×4 + sample + K check actions
    N_OBS     = 3       # none, good, bad

    def sample_initial_state(self):
        # Agent starts at middle-left; each rock is good with prob 0.5
        rock_mask = random.randint(0, (1 << K) - 1)
        return (N // 2, 0, rock_mask)

    def step(self, state, action):
        # Absorbing terminal
        if state == TERMINAL:
            return TERMINAL, OBS_NONE, 0.0

        row, col, rock_mask = state

        # --- Movement ---
        if action == A_NORTH:
            row = max(0, row - 1)
            return (row, col, rock_mask), OBS_NONE, 0.0

        if action == A_SOUTH:
            row = min(N - 1, row + 1)
            return (row, col, rock_mask), OBS_NONE, 0.0

        if action == A_WEST:
            col = max(0, col - 1)
            return (row, col, rock_mask), OBS_NONE, 0.0

        if action == A_EAST:
            col += 1
            if col >= N:
                return TERMINAL, OBS_NONE, +10.0
            return (row, col, rock_mask), OBS_NONE, 0.0

        # --- Sample ---
        if action == A_SAMPLE:
            rock_here = next(
                (i for i, (r, c) in enumerate(ROCK_POS) if r == row and c == col),
                None,
            )
            if rock_here is not None and (rock_mask & (1 << rock_here)):
                reward    = +10.0
                rock_mask = rock_mask & ~(1 << rock_here)   # mark as bad
            else:
                reward = -10.0   # bad rock or empty cell
            return (row, col, rock_mask), OBS_NONE, reward

        # --- Check rock i ---
        i    = action - 5           # rock index
        r_r, c_r = ROCK_POS[i]
        dist = math.sqrt((row - r_r) ** 2 + (col - c_r) ** 2)
        accuracy = (1.0 + 2.0 ** (-dist / D_HALF)) / 2.0

        is_good = bool(rock_mask & (1 << i))
        if random.random() < accuracy:
            obs = OBS_GOOD if is_good else OBS_BAD
        else:
            obs = OBS_BAD  if is_good else OBS_GOOD

        return (row, col, rock_mask), obs, 0.0


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def belief_summary(agent: POMCP) -> str:
    """Show marginal P(rock_i is good) and most likely agent position."""
    particles = [p for p in agent.belief_particles if p != TERMINAL]
    n = len(particles)
    if n == 0:
        return "all particles at TERMINAL"

    rock_lines = []
    for i in range(K):
        p_good = sum(1 for p in particles if p[2] & (1 << i)) / n
        rock_lines.append(f"rock_{i}@{ROCK_POS[i]} P(good)={p_good:.2f}")

    pos_counts = Counter((p[0], p[1]) for p in particles)
    top_pos, top_cnt = pos_counts.most_common(1)[0]
    agent_str = f"agent most likely {top_pos} ({top_cnt/n*100:.0f}%)"

    return "  |  ".join(rock_lines) + f"  |  {agent_str}"


def state_summary(state) -> str:
    if state == TERMINAL:
        return "TERMINAL"
    row, col, mask = state
    rocks = "".join("G" if mask & (1 << i) else "B" for i in range(K))
    return f"({row},{col}) rocks=[{rocks}]"


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("=" * 70)
    print("POMCP on RockSample(N=4, K=3)")
    print(f"Grid: {N}x{N}  |  Rocks at {ROCK_POS}")
    print("=" * 70)

    random.seed(7)

    env   = RockSamplePOMDP()
    agent = POMCP(
        env,
        n_sims        = 2000,
        max_depth     = 30,
        ucb_c         = 50.0,
        gamma         = 0.95,
        n_particles   = 1000,
        rollout_depth = 15,
    )

    true_state   = env.sample_initial_state()
    total_reward = 0.0
    n_steps      = 25

    print(f"\nTrue initial state : {state_summary(true_state)}")
    print(f"Belief             : {belief_summary(agent)}\n")
    print("-" * 70)

    for step in range(1, n_steps + 1):
        if true_state == TERMINAL:
            print("\nAgent reached the east exit — episode over.")
            break

        action = agent.plan()

        print(f"\nStep {step:2d} | True state: {state_summary(true_state)}")
        print(f"  Belief   : {belief_summary(agent)}")
        print(f"  Q-values :\n{agent.q_summary(ACTION_NAMES)}")
        print(f"  Chosen   : {ACTION_NAMES[action]}")

        true_state, obs, reward = env.step(true_state, action)
        total_reward += reward

        print(f"  Obs      : {OBS_NAMES[obs]}")
        print(f"  Reward   : {reward:+.1f}   (cumulative: {total_reward:+.1f})")
        print(f"  -> New state: {state_summary(true_state)}")

        agent.update_belief(action, obs)

    print("\n" + "=" * 70)
    print(f"Episode finished.  Total reward: {total_reward:+.1f}")
    print("=" * 70)


if __name__ == "__main__":
    main()
