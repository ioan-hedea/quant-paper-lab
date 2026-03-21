"""
Tiger POMDP benchmark — runs on the generic POMCP planner.

States  : tiger_left (0), tiger_right (1)
Actions : listen (0), open_left (1), open_right (2)
Obs     : hear_left (0), hear_right (1)
Rewards :
  listen                      -> -1
  open correct door (no tiger)-> +10
  open wrong door   (tiger)   -> -100
Obs model:
  listen -> hear correct side with prob 0.85
  open_* -> hear_left / hear_right each with prob 0.5 (uninformative)
Transition:
  open_* -> tiger resets uniformly (new sub-episode)
  listen -> tiger stays in place
"""

from __future__ import annotations

import random
from pomcp import POMDP, POMCP


# ---------------------------------------------------------------------------
# Constants
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


# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------

class TigerPOMDP(POMDP):
    """
    Generative model for the Tiger POMDP.
    G(state, action) -> (next_state, observation, reward)
    """

    N_ACTIONS        = 3
    N_OBS            = 2
    CORRECT_OBS_PROB = 0.85

    def sample_initial_state(self) -> int:
        return random.randint(0, 1)

    def step(self, state: int, action: int) -> tuple[int, int, float]:
        if action == LISTEN:
            next_state = state
            obs        = self._listen_obs(state)
            reward     = -1.0

        elif action == OPEN_LEFT:
            reward     = -100.0 if state == TIGER_LEFT else +10.0
            next_state = random.randint(0, 1)
            obs        = random.randint(0, 1)

        else:  # OPEN_RIGHT
            reward     = -100.0 if state == TIGER_RIGHT else +10.0
            next_state = random.randint(0, 1)
            obs        = random.randint(0, 1)

        return next_state, obs, reward

    def _listen_obs(self, state: int) -> int:
        if random.random() < self.CORRECT_OBS_PROB:
            return HEAR_LEFT if state == TIGER_LEFT else HEAR_RIGHT
        else:
            return HEAR_RIGHT if state == TIGER_LEFT else HEAR_LEFT


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def belief_summary(agent: POMCP) -> str:
    n   = len(agent.belief_particles)
    p_L = agent.belief_particles.count(TIGER_LEFT)  / n
    p_R = agent.belief_particles.count(TIGER_RIGHT) / n
    return f"P(tiger_left)={p_L:.2f}  P(tiger_right)={p_R:.2f}"


# ---------------------------------------------------------------------------
# Main
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

    true_state   = env.sample_initial_state()
    total_reward = 0.0
    n_steps      = 15

    print(f"\nTrue initial state: {STATE_NAMES[true_state]}")
    print(f"Agent's initial belief: {belief_summary(agent)}\n")
    print("-" * 60)

    for step in range(1, n_steps + 1):
        action = agent.plan()

        print(f"\nStep {step:2d} | True state: {STATE_NAMES[true_state]}")
        print(f"  Belief   : {belief_summary(agent)}")
        print(f"  Q-values :\n{agent.q_summary(ACTION_NAMES)}")
        print(f"  Chosen   : {ACTION_NAMES[action]}")

        true_state, obs, reward = env.step(true_state, action)
        total_reward += reward

        print(f"  Obs      : {OBS_NAMES[obs]}")
        print(f"  Reward   : {reward:+.1f}   (cumulative: {total_reward:+.1f})")

        agent.update_belief(action, obs)

    print("\n" + "=" * 60)
    print(f"Episode finished.  Total reward: {total_reward:+.1f}")
    print("=" * 60)


if __name__ == "__main__":
    main()
