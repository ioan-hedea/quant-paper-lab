import gym
import simple_grid
from q_learning_skeleton import *


def act_loop(env, agent, num_episodes):
    """
    TODO: Complete the loop of the Q-learning agent.
    """
    print("Hello from q_learning_main.py")
    for episode in range(num_episodes):
        pass
    env.close()


if __name__ == "__main__":
    env = simple_grid.DrunkenWalkEnv(map_name="walkInThePark")
    # env = simple_grid.DrunkenWalkEnv(map_name="theAlley")
    num_a = env.action_space.n

    if (type(env.observation_space) is gym.spaces.discrete.Discrete):
        num_o = env.observation_space.n
    else:
        raise ("QTable only works for discrete observations")

    discount = DEFAULT_DISCOUNT
    ql = QLearner(num_o, num_a, discount)  # <- QTable
    act_loop(env, ql, NUM_EPISODES)
