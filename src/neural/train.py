"""Trains the noodle."""

import time

import numpy as np
import torch
from stable_baselines3 import DQN
from stable_baselines3.common.evaluation import evaluate_policy

from src.neural import environment, visualizations
from src.noodle.model.entities import Direction


def create_dqn_model(env: environment.SnakeGameEnv, dqn_config: dict) -> DQN:
    """Returns the DQN model for training."""
    policy_kwargs = dict(net_arch=dqn_config["network_architecture"])

    return DQN(
        policy=dqn_config["policy"],
        env=env,
        verbose=0,
        buffer_size=dqn_config["buffer_size"],
        learning_rate=dqn_config["learning_rate"],
        batch_size=dqn_config["batch_size"],
        exploration_fraction=dqn_config["exploration_fraction"],
        exploration_initial_eps=dqn_config["exploration_eps"]["initial"],
        exploration_final_eps=dqn_config["exploration_eps"]["final"],
        target_update_interval=dqn_config["target_update"],
        policy_kwargs=policy_kwargs,
    )


def train_model(config: dict) -> None:
    """Trains a DQN model on the Snake game and evaluates its performance."""
    # Extract relevant configuration sections
    dqn_config = config["dqn"]
    training_config = config["training"]
    evaluation_config = config["evaluation"]

    # Create environment and DQN model
    env = environment.SnakeGameEnv.from_config(config["environment"])
    model = create_dqn_model(env, dqn_config)

    # Setup parameters for training
    steps_per_second = training_config.get("steps_per_second", 10)
    time_per_step = 1.0 / steps_per_second
    epsilon = dqn_config["exploration_eps"]["initial"]
    epsilon_decay = (
        dqn_config["exploration_eps"]["initial"]
        - dqn_config["exploration_eps"]["final"]
    ) / (training_config["timesteps"] * dqn_config["exploration_fraction"])

    # Initialize tracking variables
    total_reward = 0
    episode_rewards = []
    training_plotter = visualizations.TrainingPlotter(window_size=100)

    # Reset the environment
    obs, _ = env.reset()

    for step in range(training_config["timesteps"]):
        start_time = time.time()

        # Adjust epsilon for exploration vs. exploitation
        epsilon = max(
            dqn_config["exploration_eps"]["final"], epsilon - epsilon_decay
        )
        action = (
            env.action_space.sample()
            if np.random.rand() < epsilon
            else model.predict(obs, deterministic=True)[0]
        )

        # Take a step in the environment
        obs, reward, terminated, truncated, _ = env.step(action)
        total_reward += reward

        # Handle end of episode
        if terminated or truncated:
            episode_rewards.append(total_reward)
            training_plotter.update(total_reward)
            total_reward = 0
            obs, _ = env.reset()

        # Compute and render Q-values
        render_with_q_values(env, model, obs)

        # Maintain desired FPS
        time.sleep(max(0, time_per_step - (time.time() - start_time)))

    # Save the trained model
    model.save("dqn_snake")

    # Evaluate the model
    evaluate_and_print_results(model, env, evaluation_config["episodes"])

    # Close the environment
    env.close()


def render_with_q_values(
    env: environment.SnakeGameEnv, model: DQN, obs: np.ndarray
) -> None:
    """
    Renders the environment with Q-values displayed for possible actions.

    Args:
        env: The game environment.
        model: The DQN model used for predictions.
        obs: The current observation from the environment.
    """
    # Convert observation to a PyTorch tensor
    obs_tensor = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)

    # Get Q-values from the model
    q_values = model.policy.q_net(obs_tensor).detach().numpy().squeeze(0)

    # Map Q-values to the resulting cells
    snake_head = env.model.snake.head()
    actions = [Direction.UP, Direction.RIGHT, Direction.DOWN, Direction.LEFT]
    q_values_dict = {
        snake_head.move(action): q_value
        for action, q_value in zip(actions, q_values)
    }

    # Render the environment with Q-values
    env.render(q_values=q_values_dict)


def evaluate_and_print_results(
    model: DQN, env: environment.SnakeGameEnv, n_eval_episodes: int = 10
) -> None:
    """Evaluates the trained model and prints the results."""
    mean_reward, std_reward = evaluate_policy(
        model, env, n_eval_episodes=n_eval_episodes
    )
    print(f"Mean reward: {mean_reward}, Std reward: {std_reward}")
