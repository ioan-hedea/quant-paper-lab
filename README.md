# Sequential Decision-Making

This repository contains implementations and exercises for reinforcement learning algorithms, focusing on Q-Learning and Deep Q-Learning approaches for sequential decision-making problems.

## Overview

### Q-Learning (`q_learning_*`)
Classic Q-Learning agent that learns optimal policies in discrete state-action environments using a Q-table.

**Features:**
- Discrete state and action spaces
- Q-table based value function
- Epsilon-greedy action selection
- Temporal Difference (TD) learning updates

**Environment:** DrunkenWalkEnv - A custom grid-world environment where an agent navigates through terrain with:
- `S`: Starting position
- `.`: Normal pavement
- `H`: Pothole (20% chance of tripping with -10 penalty)
- `G`: Goal (reward of +10)

Available maps:
- `"theAlley"`: 1D linear path
- `"walkInThePark"`: 6x8 grid with potholes
- `"4x4"`, `"8x8"`: Classic grid sizes

### Deep Q-Learning (`deep_q_learning_*`)
Deep Q-Network (DQN) agent that learns policies using a neural network to approximate Q-values.

**Features:**
- Neural network (MLP) for Q-value approximation
- Experience replay memory (configurable size)
- Batch training with PyTorch
- Epsilon decay for exploration-exploitation trade-off
- GPU acceleration support (falls back to CPU if unavailable)

**Environment:** LunarLander-v2 (from OpenAI Gym)
- Continuous state space (8 dimensions)
- 4 discrete actions
- Goal: Land the lunar lander safely

**Key Hyperparameters:**
- `NUM_EPISODES`: 500
- `RMSIZE`: 10,000 (replay memory size)
- `BATCH_SIZE`: 256
- `DEFAULT_DISCOUNT`: 0.99
- `EPSILON`: Initial exploration rate
- `LEARNINGRATENET`: 0.0001

## Dependencies

- Python 3.7+
- `gym`: OpenAI Gym environment framework
- `numpy`: Numerical computing
- `torch`: PyTorch deep learning framework

Install dependencies:
```bash
pip install gym numpy torch
```

## Usage

### Q-Learning
```python
python q_learning_main.py
```

Modify environment in `q_learning_main.py`:
```python
env = simple_grid.DrunkenWalkEnv(map_name="walkInThePark")
# or try:
# env = simple_grid.DrunkenWalkEnv(map_name="theAlley")
```

### Deep Q-Learning
```python
python deep_q_learning_main.py
```

Recorded episodes are saved to `./recorded_episodes/` for analysis.

## Exercise Tasks

### Q-Learning Skeleton (`q_learning_skeleton.py`)
Implement the `QLearner` class:
- `__init__`: Initialize Q-table and hyperparameters
- `select_action(state)`: Epsilon-greedy action selection
- `process_experience()`: Q-value updates using TD learning
- `reset_episode()`: Episode initialization and stats tracking
- `report()`: Print agent statistics

### Deep Q-Learning Skeleton (`deep_q_learning_skeleton.py`)
Implement:
- `ReplayMemory`: Store and sample experience transitions
  - `store_experience()`: Add transitions to memory
  - `sample_batch()`: Random batch sampling
- Integrate batch training loop in `process_experience()`
- Implement target network (TODO in main file)

## Algorithm Details

### Q-Learning Update Rule
$$Q(s,a) \leftarrow Q(s,a) + \alpha[r + \gamma \max_a Q(s',a) - Q(s,a)]$$

Where:
- $\alpha$: Learning rate
- $r$: Immediate reward
- $\gamma$: Discount factor
- $s'$: Next state

### DQN with Experience Replay
1. Store transitions $(s, a, r, s', \text{done})$ in replay memory
2. Sample random minibatch from memory
3. Compute target: $y = r + \gamma \max_a Q(s', a; \theta^-)$
4. Update network parameters via gradient descent

## Configuration

Key parameters in skeleton files:
- `NUM_EPISODES`: Number of training episodes
- `MAX_EPISODE_LENGTH`: Maximum steps per episode
- `DEFAULT_DISCOUNT`: Discount factor ($\gamma$)
- `EPSILON`: Exploration rate
- `LEARNINGRATE`: Learning rate ($\alpha$)
- `BATCH_SIZE`: Minibatch size for DQN

## Docker

Build and run in Docker:
```bash
docker build -t rl-assignment ./docker
docker run -it rl-assignment python deep_q_learning_main.py
```

## References

- Sutton, R. S., & Barto, A. G. (2018). *Reinforcement Learning: An Introduction*
- Mnih, V., et al. (2015). Human-level control through deep reinforcement learning. *Nature*
- OpenAI Gym: https://www.gymlibrary.dev/

## Notes

- Time horizon consideration in DQN: Remaining time (1000 - t)/1000 is appended to observations
- Gradient clipping is applied in single Q-updates to prevent exploding gradients
- Episode videos are recorded every 10 episodes in deep Q-learning
