"""Trains the noodle."""

import time

import numpy as np
import torch

from src.neural import visualizations
from src.neural.environment import SnakeGameEnv
from src.neural.qnet import Agent
from src.noodle import model
from src.noodle.model.entities import Direction


def train_model(config: dict) -> None:
    """
    Train an Agent on the Snake game.

    Args:
        config: A dictionary containing environment, agent, and training parameters.
    """
    env = SnakeGameEnv.from_config(config["environment"])
    agent = initialize_agent(config, env)

    # Initialize visualization tools
    training_plotter = visualizations.TrainingPlotter(window_size=100)
    episode_rewards = []

    # Configure steps per second
    steps_per_second = config["training"]["steps_per_second"]
    step_duration = 1.0 / steps_per_second

    # Training loop
    for episode in range(config["training"]["episodes"]):
        state, _ = env.reset()
        done = False
        total_reward = 0.0

        while not done:
            start_time = time.time()  # Record step start time

            # Select an action using epsilon-greedy policy
            action = agent.epsilon_greedy_action(state, env.output_size)

            # Step through the environment
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            # Perform a training step
            agent.train_step(state, action, reward, next_state, done)

            # Update state and accumulate reward
            state = next_state
            total_reward += reward

            # Render Q-values during training
            render_with_q_values(env, agent.qnet, state)

            # Enforce FPS
            elapsed_time = time.time() - start_time
            sleep_time = max(0, step_duration - elapsed_time)
            time.sleep(sleep_time)

        # Update the target network periodically
        if episode % config["training"]["target_update_interval"] == 0:
            agent.update_target_network()

        # Log rewards and update visualization
        episode_rewards.append(total_reward)
        training_plotter.update(total_reward)
        print(
            f"Episode {episode + 1}/{config['training']['episodes']}: Total Reward = {total_reward}"
        )

    env.close()


def render_with_q_values(
    env: SnakeGameEnv, qnet: torch.nn.Module, state: np.ndarray
) -> None:
    """
    Renders the environment with Q-values displayed for possible actions.

    Args:
        env: The game environment.
        qnet: The QNet model.
        state: The current state of the environment as a numpy array.
    """
    # Convert state to tensor
    state_tensor: torch.Tensor = torch.tensor(
        state, dtype=torch.float32
    ).unsqueeze(0)

    with torch.no_grad():
        q_values: np.ndarray = qnet(state_tensor).squeeze(0).numpy()

    # Snake head position
    snake_head = env.model.snake.head()
    head_row, head_col = snake_head.row, snake_head.col

    # Map Q-values to grid cells based on snake movement
    actions = [Direction.UP, Direction.RIGHT, Direction.DOWN, Direction.LEFT]
    deltas = {
        Direction.UP: (-1, 0),
        Direction.RIGHT: (0, 1),
        Direction.DOWN: (1, 0),
        Direction.LEFT: (0, -1),
    }

    q_values_dict = {
        model.entities.Cell(
            head_row + deltas[action][0], head_col + deltas[action][1]
        ): q_value
        for action, q_value in zip(actions, q_values)
    }

    # Render Q-values on the environment
    env.render(q_values=q_values_dict)


def initialize_agent(config: dict, env: SnakeGameEnv) -> Agent:
    """
    Initialize the Agent using the provided configuration.

    Args:
        config: A dictionary containing the agent and training parameters.
        env: The game environment.

    Returns:
        An initialized Agent instance.
    """
    # Extract dimensions for the agent
    input_dim = env.input_size
    output_dim = env.output_size

    # Initialize Agent using from_config
    return Agent.from_config(config["agent"], input_dim, output_dim)
