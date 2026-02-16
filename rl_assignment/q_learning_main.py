import gym
import simple_grid
from q_learning_skeleton import *


def act_loop(env, agent, num_episodes):
    """
    Completed the loop of the Q-learning agent.
    """
    for episode in range(num_episodes):
        state = env.reset()
        agent.reset_episode(state)

        print('---episode %d---' % episode)

        for t in range(MAX_EPISODE_LENGTH):
            action = agent.select_action(state)
            next_state, reward, done, _ = env.step(action)
            agent.process_experience(state, action, next_state, reward, done)
            state = next_state

            if done:
                print("Episode finished after {} timesteps".format(t + 1))
                agent.report()
                break
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
