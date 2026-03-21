from typing import Dict, List, Optional, Tuple

from z3 import And, Bool, CheckSatResult, If, Int, Not, Solver


def init_environment(
    timesteps: int, grid_size: Optional[int] = 10
) -> Tuple[
    Solver,
    Dict[str, List[Int]],
    Dict[int, Dict[str, List[Int]]],
    Dict[int, Dict[str, List[Int]]],
    Dict[str, List[Bool]],
]:
    """
    Initializes the Z3 variables and constraints for the environment.

    Args:
        timesteps (int): The number of timesteps in the planning horizon.
        grid_size (int, optional): The size of the grid. Defaults to 10.

    Returns:
        Tuple[
            Solver,
            Dict[str, List[Int]],
            Dict[int, Dict[str, List[Int]]],
            Dict[int, Dict[str, List[Int]]],
            Dict[str, List[Bool]],
        ]:
            A tuple containing the following:
                Solver: The Z3 solver.
                Dict[str, List[Int]]: A dictionary containing the agent's position (row, column) at each timestep.
                Dict[int, Dict[str, List[Int]]]: A dictionary containing the targets' position (row, column) at each timestep.
                Dict[int, Dict[str, List[Int]]]: A dictionary containing the distances between each target and the agent at each timestep.
                Dict[str, List[Bool]]: A dictionary containing the agent's direction of movement at each timestep (0 for 'north', 1 for 'east', 2 for 'south', 3 for 'west').
    """
    # Dictionary to store the agent position (row, column) at each timestep
    agent_position: Dict[str, List[Int]] = {"row": [], "column": []}
    for axis in ["row", "column"]:
        agent_position[axis] = [Int("a_{}{}".format(axis, t)) for t in range(timesteps)]

    # Dictionary to store the targets' position (row, column) at each timestep
    target_positions: Dict[int, Dict[str, List[Int]]] = {
        0: {"row": [], "column": []},
        1: {"row": [], "column": []},
        2: {"row": [], "column": []},
    }
    for target in target_positions:
        for axis in ["row", "column"]:
            target_positions[target][axis] = [
                Int("t_{}{}{}".format(target, axis, t)) for t in range(timesteps)
            ]

    # Dictionary to store each of the target distances from the agent at each timestep
    targets_distance: Dict[int, Dict[str, List[Int]]] = {
        0: {"row": [], "column": []},
        1: {"row": [], "column": []},
        2: {"row": [], "column": []},
    }
    for target in targets_distance:
        for axis in ["row", "column"]:
            targets_distance[target][axis] = [
                Int("d_{}{}{}".format(target, axis, t)) for t in range(timesteps)
            ]

    # Dictionary to store each action at each timestep (boolean)
    agent_direction: Dict[str, List[Bool]] = {
        "north": [Bool("north_{}".format(t)) for t in range(timesteps)],
        "east": [Bool("east_{}".format(t)) for t in range(timesteps)],
        "south": [Bool("south_{}".format(t)) for t in range(timesteps)],
        "west": [Bool("west_{}".format(t)) for t in range(timesteps)],
    }

    # Initialize the solver
    solver = Solver()

    # Constraints to ensure that the agent and all the targets are within the grid at each timestep
    for t in range(timesteps):
        for axis in ["row", "column"]:
            for i in range(3):
                solver.add(target_positions[i][axis][t] < grid_size)
                solver.add(target_positions[i][axis][t] >= 0)

            solver.add(agent_position[axis][t] < grid_size)
            solver.add(agent_position[axis][t] >= 0)

    # Constraint to ensure that if one target is picked up, a new target appears at a different location
    for t in range(1, timesteps):
        for i in range(3):
            solver.add(
                If(  # if: the agent was at the target's position in the previous timestep
                    And(
                        target_positions[i]["row"][t - 1] == agent_position["row"][t - 1],
                        target_positions[i]["column"][t - 1] == agent_position["column"][t - 1],
                    ),
                    # then: either the row or column position of the target must have changed since the last timestep
                    Not(And(
                        target_positions[i]["row"][t] != target_positions[i]["row"][t - 1],
                        target_positions[i]["column"][t] != target_positions[i]["column"][t - 1],
                    )),
                    # else: no further constraints
                    True
                )
            )

    # Constraint to ensure that all the targets are in different locations at each timestep
    for t in range(timesteps):
        for i in range(3):
            for j in range(i):
                solver.add(
                    Not(
                        And(
                            target_positions[i]["row"][t] == target_positions[j]["row"][t],
                            target_positions[i]["column"][t] == target_positions[j]["column"][t],
                        )
                    )
                )

    # Constraint to ensure that distance between the agent and the targets is valid at each timestep
    for t in range(timesteps):
        for axis in ["row", "column"]:
            for i in range(3):
                solver.add(
                    targets_distance[i][axis][t]
                    == target_positions[i][axis][t] - agent_position[axis][t]
                )

    # Constraint to ensure that the agent can only move in one direction at each timestep
    for t in range(1, timesteps):
        solver.add(
            agent_position["row"][t]  # agent's current row position
            == agent_position["row"][t - 1]  # is equal to the agent's previous row position
            + If(
                agent_direction["north"][t - 1],
                -1,  # if the agent moved north, then the agent's current row position is one less than the agent's previous row position
                If(
                    agent_direction["south"][t - 1], 1, False
                ),  # if the agent moved south, then the agent's current row position is one more than the agent's previous row position
            )
        )
        solver.add(
            agent_position["column"][t]  # agent's current column position
            == agent_position["column"][t - 1]  # is equal to the agent's previous column position
            + If(
                agent_direction["east"][t - 1],
                1,  # if the agent moved west, then the agent's current column position is one more than the agent's previous column position
                If(
                    agent_direction["west"][t - 1], -1, False
                ),  # if the agent moved east, then the agent's current column position is one less than the agent's previous column position
            )
        )

    return (
        solver,
        agent_position,
        target_positions,
        targets_distance,
        agent_direction,
    )


def check_run(
    agent_position_list: List[List[int]],
    target_position_list: List[List[List[int]]],
    action_list: List[int],
    grid_size: Optional[int] = 10,
) -> CheckSatResult:
    """
    Check if the given sequence of actions leads the agent from the given initial positions to the target positions on a grid of
    the specified size without colliding with obstacles.

    Args:
        agent_position_list (List[List[int]]): A list of agent positions where each position is a list of two integers
        representing the coordinates (row and column) of the agent at that time step.
        target_position_list (List[List[List[int]]]): A list of target positions where each target position is a list of
        three lists of two integers representing the coordinates (row and column) of the three targets at that time step.
        action_list (List[int]): A list of integers representing the sequence of actions taken by the agent where 0 represents
        north, 1 represents east, 2 represents south, and 3 represents west.
        grid_size (Optional[int]): An optional integer representing the size of the square grid. Defaults to 10.

    Returns:
        CheckSatResult: An enumeration indicating whether the given sequence of actions leads to a valid solution or not.
    """
    # Initialize the environment
    (
        solver,
        agent_position,
        target_positions,
        _,
        agent_direction,
    ) = init_environment(len(action_list), grid_size)

    # Add the run's data as constraints to the solver
    for t in range(len(agent_position_list)):
        solver.add(agent_position["row"][t] == int(agent_position_list[t][0]))
        solver.add(agent_position["column"][t] == int(agent_position_list[t][1]))

        for i in range(3):
            solver.add(
                target_positions[i]["row"][t] == int(target_position_list[t][i][0])
            )
            solver.add(
                target_positions[i]["column"][t] == int(target_position_list[t][i][1])
            )

        solver.add(agent_direction["north"][t] == bool(action_list[t] == 0))
        solver.add(agent_direction["east"][t] == bool(action_list[t] == 1))
        solver.add(agent_direction["south"][t] == bool(action_list[t] == 2))
        solver.add(agent_direction["west"][t] == bool(action_list[t] == 3))

    return solver.check()


def find_loop(
    agent_position_list: List[List[int]],
    target_position_list: List[List[List[int]]],
    action_list: List[int],
    grid_size: Optional[int] = 10,
) -> CheckSatResult:
    """
    Checks if the path of the agent contains any loops of size 2, i.e.: the agent directly moving back to the previous square.

    Args:
        agent_position_list (List[List[int]]): A list of agent positions where each position is a list of two integers
        representing the coordinates (row and column) of the agent at that time step.
        target_position_list (List[List[List[int]]]): A list of target positions where each target position is a list of
        three lists of two integers representing the coordinates (row and column) of the three targets at that time step.
        action_list (List[int]): A list of integers representing the sequence of actions taken by the agent where 0 represents
        north, 1 represents east, 2 represents south, and 3 represents west.
        grid_size (Optional[int]): An optional integer representing the size of the square grid. Defaults to 10.
    Returns:
        CheckSatResult: An enumeration indicating whether the given sequence is free of loops (sat) or not (unsat).
    """
    # TODO: Implement this function as specified. Note that you may not need all arguments made available to you.

def find_efficient_path(
    agent_position_list: List[List[int]],
    target_position_list: List[List[List[int]]],
    action_list: List[int],
    grid_size: Optional[int] = 10,
) -> CheckSatResult:
    """
    Checks if the path of the agent took the most efficient path to a target.

    Args:
        agent_position_list (List[List[int]]): A list of agent positions where each position is a list of two integers
        representing the coordinates (row and column) of the agent at that time step.
        target_position_list (List[List[List[int]]]): A list of target positions where each target position is a list of
        three lists of two integers representing the coordinates (row and column) of the three targets at that time step.
        action_list (List[int]): A list of integers representing the sequence of actions taken by the agent where 0 represents
        north, 1 represents east, 2 represents south, and 3 represents west.
        grid_size (Optional[int]): An optional integer representing the size of the square grid. Defaults to 10.
    Returns:
        CheckSatResult: An enumeration indicating whether the given sequence the shortest path (sat) or not (unsat).
    """
    # TODO: Implement this function as specified. Note that you may not need all arguments made available to you.

def closest_target(
    agent_position_list: List[List[int]],
    target_position_list: List[List[List[int]]],
    action_list: List[int],
    grid_size: Optional[int] = 10,
) -> CheckSatResult:
    """
    Checks if the path of the agent was to the closest possible target

    Args:
        agent_position_list (List[List[int]]): A list of agent positions where each position is a list of two integers
        representing the coordinates (row and column) of the agent at that time step.
        target_position_list (List[List[List[int]]]): A list of target positions where each target position is a list of
        three lists of two integers representing the coordinates (row and column) of the three targets at that time step.
        action_list (List[int]): A list of integers representing the sequence of actions taken by the agent where 0 represents
        north, 1 represents east, 2 represents south, and 3 represents west.
        grid_size (Optional[int]): An optional integer representing the size of the square grid. Defaults to 10.
    Returns:
        CheckSatResult: An enumeration indicating whether the given sequence to the closest target (sat) or not (unsat).
    """
    # TODO: Implement this function as specified. Note that you may not need all arguments made available to you.
