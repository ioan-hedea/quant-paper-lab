from keras.models import Sequential
from keras.layers import Dense
import numpy as np
import nptyping as npt
from typing import List, Optional


class Agent:
    """
    This class defines an Agent that navigates in a 2D grid and tries to collect
    as many targets as possible while minimizing the distance travelled.

    Attributes:
        grid_size (int): The size of the grid.
        agent_position (List[int, int]): The current position of the agent.
        target_positions (List[List[int, int]]): The positions of the targets.
        model (Sequential): The neural network model used to predict actions.
        map (npt.NDArray[npt.Shape["grid_size, grid_size", int]]): A numpy array
            representing the map with the location of the targets and the agent.
        total_collected (int): The total number of targets collected.
        previously_collected (int): The index of the last target collected.
    """

    def __init__(
        self,
        grid_size: int = 10,
        weights: Optional[str] = None,
        seed: int = 1805,
    ) -> None:
        """
        Initializes the Agent object with default or provided values.

        Args:
            grid_size (int): The size of the grid. Default is 10.
        """
        super(Agent, self).__init__()

        # Set the random seed for reproducibility.
        np.random.seed(seed)

        self.grid_size: int = grid_size

        # The sprites used to represent the agent, targets and empty cells.
        self.sprites = {
            "empty": 0,
            "robot": 1,
            "target": 2,
        }

        # Initialize the agent's position and the targets' positions.s
        self.agent_position: List[int] = []
        self.target_positions: List[List[int]] = []

        # Initialize the neural network model.
        self.model = Sequential()
        self.model.add(Dense(units=200, activation="relu", input_dim=6))
        self.model.add(Dense(units=100, activation="relu"))
        self.model.add(Dense(units=4, activation="linear"))
        if weights is not None:
            self.model.load_weights(weights)

        # Initialize the map and reset it to populate it with the agent and targets.
        self.map: npt.NDArray[
            npt.Shape["grid_size, grid_size", int], np.dtype[np.int32]
        ]
        self.reset_map()

        # Initialize the total number of targets collected and the index of the last for bookkeeping.
        self.total_collected: int = 0
        self.previously_collected: List[int] = []

    def reset_map(
        self,
        agent_pos: Optional[List[int]] = None,
        target_pos: Optional[List[List[int]]] = None,
    ) -> None:
        """
        Resets the map with new or default values.

        Args:
            agent_pos (List[int, int]): The position of the agent. If None, a random
                position will be generated. Default is None.
            target_pos (List[List[int, int]]): The positions of the targets. If None,
                three random positions will be generated. Default is None.
        """
        if agent_pos is None:
            self.agent_position = self.random_state()
        if target_pos is None:  # Create 3 distinct random targets
            for i in range(3):
                self.target_positions.append(self.random_state(self.target_positions + [self.agent_position]))

        self.map = np.zeros((self.grid_size, self.grid_size))

        for target in self.target_positions:
            self.map[target[0], target[1]] = self.sprites["target"]

        self.map[self.agent_position[0], self.agent_position[1]] = self.sprites["robot"]

    def get_action(self, observation: npt.NDArray[npt.Shape["1, 6"], npt.Int]) -> int:
        """
        Returns the action predicted by the model based on the observation.

        Args:
            observation (npt.NDArray[npt.Shape["1, 6"], npt.Int]): The observation of the
                current state of the environment.

        Returns:
            int: The predicted action (0 for 'north', 1 for 'east', 2 for 'south', 3 for 'west').
        """

        # index 0 because we only have 1 observation
        predicted_class: npt.NDArray[npt.Shape["1, 4"], npt.Float] = self.model.predict(
            observation, verbose=0
        )[0]
        # return the action with the highest value
        return int(np.argmax(predicted_class))

    def random_state(self, exclude: List[List[int]] = []) -> List[int]:
        """
        Generates a random state on the grid not in the "exclude" list.

        Args:
            exclude (List[List[int]]): A list of positions (row, column) that must be selected.

        Returns:
            List[int, int]: A randomly generated state on the grid.
        """
        state = list(np.random.randint(0, self.grid_size, 2))
        while state in exclude:
            state = list(np.random.randint(0, self.grid_size, 2))
        return state

    def get_state(self) -> npt.NDArray[npt.Shape["1, 6"], npt.Int]:
        """
        Gets the current state (manhattan distance between agent and the targets) of the grid.

        Returns:
            npt.NDArray[npt.Shape["1, 6"], npt.Int]: The current state of the grid.
        """
        # Convert to numpy array for easier calculations
        target_locations = np.array(self.target_positions)
        agent_location = np.array(self.agent_position)

        return (target_locations - agent_location).reshape(1, 6)

    def move(self, a: int) -> float:
        """
        Moves the agent in the given direction and updates the state of the grid.

        Args:
            a (int): The direction in which to move the agent (0 for 'north', 1 for 'east', 2 for 'south', 3 for 'west').

        Returns:
            float: The reward obtained from the move.
        """
        # Store the current position of the agent
        [row, column] = self.agent_position

        # Remove the agent from the current position (to move elsewhere)
        self.map[row, column] = self.sprites["empty"]

        bonusReward: int = 100

        # Get the previous distance reward
        prev_dis_reward = self._get_dis_reward()

        # If the agent has collected a target, remove it from the map
        if len(self.previously_collected) != 0:
            self.map[self.previously_collected[0]][
                self.previously_collected[1]
            ] = self.sprites["empty"]
            self.previously_collected = []

        # Move the agent in the given direction
        if a == 0:  # north
            row = row - 1
        elif a == 1:  # east
            column = column + 1
        elif a == 2:  # south
            row = row + 1
        elif a == 3:  # west
            column = column - 1

        agent_reward: float = 0.0

        # Penalty for going out of bounds
        if row < 0 or row >= self.grid_size or column < 0 or column >= self.grid_size:
            [row, column] = self.agent_position  # Reset to previous position
            agent_reward = -10

        # Get the new distance reward
        dis_reward = self._get_dis_reward([row, column])

        # Check if the agent has collected a target
        for i in range(len(self.target_positions)):
            if self.target_positions[i] == [row, column]:
                # Reward the agent for collecting a target
                agent_reward = bonusReward

                # Store the position of the target that was collected
                self.previously_collected = self.target_positions[i]

                # Generate a new target (blocking the position of the current target positions)
                self.target_positions[i] = self.random_state(self.target_positions)
                [g1, g2] = self.target_positions[i]

                # Update the map
                self.map[g1][g2] = self.sprites["target"]

                # Update the total number of targets collected
                self.total_collected += 1

        # If the agent has collected a target or received no penalty, add the distance reward
        if agent_reward >= 0:
            agent_reward += dis_reward - prev_dis_reward

        # Update the position of the agent on the map
        self.map[row][column] = self.sprites["robot"]
        self.agent_position = [row, column]
        return agent_reward

    def _get_dis_reward(self, agent_pos: Optional[List[int]] = None) -> float:
        """
        Gets the distance reward for the current state of the grid. If an optional agent position is given,
        the distance reward is calculated for that position.

        Args:
            agent_pos (Optional[List[int]]): The position of the agent on the grid. Defaults to None.

        Returns:
            float: The distance reward.
        """
        if agent_pos is None:
            agent_pos = np.array(self.agent_position)
        else:
            agent_pos = np.array(agent_pos)

        target_pos = np.array(self.target_positions)

        # Calculate the manhattan distance between the agent and the targets
        dis = np.abs(agent_pos[0] - target_pos[:, 0]) + np.abs(
            agent_pos[1] - target_pos[:, 1]
        )

        # If the agent is on a target, set the distance to a large number
        # This is so the reward is 0 from the distance but the agent still gets the bonus reward
        dis[dis == 0] = 10**9
        return float(np.max(1.0 / dis))

    def print_map(self):
        """
        Print the current state of the game map.

        The function loops through the rows and columns of the game map, and prints a character representing each sprite type
        in the corresponding cell. An empty space is represented by a period (.), the robot is represented by a capital letter R,
        and the target is represented by a dollar sign ($).

        Example usage:
            agent.print_map()

        :return: None
        """

        print("--------------------")
        for row in range(0, self.map.shape[0]):
            for col in range(0, self.map.shape[1]):
                if self.map[row, col] == self.sprites["empty"]:  # Empty space
                    print(".", end="")
                if self.map[row, col] == self.sprites["robot"]:  # The agent/robot
                    print("R", end="")
                if self.map[row, col] == self.sprites["target"]:  # Target
                    print("$", end="")
            print()
        print("--------------------")

    @staticmethod
    def clear_lines(n_lines: Optional[int] = 12) -> None:
        """Clears the specified number of lines from the console output.

        Args:
            n_lines (int, optional): The number of lines to clear. Defaults to 12.
        """
        for _ in range(n_lines):
            print("\033[F\033[K", end="")
