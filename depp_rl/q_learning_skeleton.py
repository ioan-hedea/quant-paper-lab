
import random

import numpy as np

NUM_EPISODES = 1000
MAX_EPISODE_LENGTH = 500

DEFAULT_DISCOUNT = 0.9
EPSILON = 0.05
LEARNINGRATE = 0.1


class QLearner():
    """
    Completed the Q-learning agent.
    """
    def __init__(self, num_states, num_actions,
                 discount=DEFAULT_DISCOUNT, learning_rate=LEARNINGRATE):
        self.name = "agent1"
        self.num_states = num_states
        self.num_actions = num_actions
        self.discount = discount
        self.learning_rate = learning_rate
        self.epsilon = EPSILON

        self.Q = np.zeros((num_states, num_actions), dtype=float)

        self.last_state = None
        self.episode = 0
        self.cumulative_r = 0
        self.dis_r = 0
        self.tot_r = 0
        self.stage = 0
        self.tot_stages = 0

    def reset_episode(self, initial_state):
        """
        You may want to update some of the statistics here.
        """
        self.last_state = int(initial_state)
        self.tot_r += self.cumulative_r
        self.tot_stages += self.stage
        self.cumulative_r = 0
        self.dis_r = 0
        self.stage = 0
        self.episode += 1

    def process_experience(self, state, action, next_state, reward, done):
        """
        Update the Q-value based on the state, action, next state and reward.
        """
        state = int(state)
        action = int(action)
        next_state = int(next_state)

        self.cumulative_r += reward
        self.dis_r += reward * (self.discount ** self.stage)
        self.stage += 1
        self.last_state = next_state

        q_old = self.Q[state, action]
        if done:
            target = reward
        else:
            target = reward + self.discount * np.max(self.Q[next_state])

        self.Q[state, action] = (1 - self.learning_rate) * q_old + self.learning_rate * target

    def select_action(self, state):
        """
        Returns an action, selected based on the current state
        """
        state = int(state)
        if random.random() < self.epsilon:
            return random.randint(0, self.num_actions - 1)

        q_vals = self.Q[state]
        best_actions = np.flatnonzero(q_vals == np.max(q_vals))
        return int(np.random.choice(best_actions))

    def report(self):
        """
        Print status information in the main loop
        """
        print("---")
        print("%s: episode: %d" % (self.name, self.episode))
        print("%s: stage:   %d" % (self.name, self.stage))
        print("%s: total stages: %d" % (self.name, self.tot_stages))
        print("%s: epsilon: %f" % (self.name, self.epsilon))
        print("%s: cum_r:   %s" % (self.name, self.cumulative_r))
        print("%s: dis_r:   %s" % (self.name, self.dis_r))
        mean_r_this_ep = self.cumulative_r / self.stage if self.stage > 0 else "undef"
        mean_r = self.tot_r / self.tot_stages if self.tot_stages > 0 else "undef"
        mean_r_ep = self.tot_r / self.episode if self.episode > 0 else "undef"
        print("%s: mean r in this episode:  %s" % (self.name, mean_r_this_ep))
        print("%s: mean r in lifetime:      %s" % (self.name, mean_r))
        print("%s: mean return per episode: %s" % (self.name, mean_r_ep))
