import os

# Ignore tensorflow warnings
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import argparse
import time
from typing import List
from z3 import unsat, sat

from checker import *
from agent import Agent


def arg_parser() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the game with a trained DQN agent."
    )
    parser.add_argument(
        "-w",
        "--weights",
        type=str,
        help="Path to the trained DQN weights.",
        default="DQN_trained",
    )
    parser.add_argument(
        "-c",
        "--num-collect",
        type=int,
        default=30,
        help="Total number of targets to collect.",
    )
    parser.add_argument(
        "--perform-check",
        type=bool,
        default=True,
        help="Perform a check on the model.",
    )
    parser.add_argument(
        "-s",
        "--sleep",
        type=float,
        default=0.1,
        help="Time in seconds to wait between printing timesteps.",
    )

    return parser.parse_args()


args = arg_parser()
agent = Agent(weights=args.weights)
agent_position_list: List[List[int]] = []
target_position_list: List[List[List[int]]] = []
action_list: List[int] = []

# Print the initial map
agent.print_map()

# Run the game until the agent collects all the required number of targets
while agent.total_collected <= args.num_collect:
    # Clear the previous lines
    agent.clear_lines(13)

    # Get the agent's current state
    obs = agent.get_state()

    # Get the agent's action for the current state
    agent_action = agent.get_action(obs)

    # Append current state data to the list for posterity
    agent_position_list.append(agent.agent_position.copy())
    target_position_list.append(agent.target_positions.copy())
    action_list.append(agent_action)

    # Move the agent and get the reward (reward ignored for now)
    _ = agent.move(agent_action)

    # Print the map
    agent.print_map()
    print("collected: " + str(agent.total_collected))
    time.sleep(args.sleep)

if args.perform_check:
    print("\nVerifying the agent's run...")

    if check_run(agent_position_list, target_position_list, action_list) == unsat:
        print("Agent's run failed the initial check! Check the simulation for bugs.")
    else:
        print("Agent's run passed the initial check!")

    print("\nPerforming loop-check on the agent's run...")
    res = find_loop(agent_position_list, target_position_list, action_list)
    if res == unsat:
        print("Agent's run contains a loop.")
    elif res == sat:
        print("Agent's run did not contain a loop.")
    else:
        print(f"The loop detection function has returned {res}, which is not the expected output type. "
              f"Please implement the 'find_loop' function as specified.")
        
    
    res = find_efficient_path(agent_position_list, target_position_list, action_list)
    print("\nPerforming efficiency check on the agent's run...")
    if res == unsat:
        print("Agent took the shortest path.")
    elif res == sat:
        print("Agent did not take the shortest path. You need to provide a counterexample.")
    else:
        print(f"The efficiency function has returned {res}, which is not the expected output type. "
              f"Please implement the 'find_efficient_path' function as specified.")


    res = closest_target(agent_position_list, target_position_list, action_list)
    print("\nPerforming closest target check on the agent's run...")
    if res == unsat:
        print("Agent went for the closest target.")
    elif res == sat:
        print("Agent did not go for the closest target. You need to provide a counterexample.")
    else:
        print(f"The closest target function has returned {res}, which is not the expected output type. "
              f"Please implement the 'closest_target' function as specified.")
