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

    # Set the desired steps per second (FPS)
    steps_per_second = training_config.get("steps_per_second", 10)
    time_per_step = 1.0 / steps_per_second

    # Use the built-in learn method if specified
    if training_config["use_builtin"]:
        model.learn(total_timesteps=training_config["timesteps"])
    else:
        # Initialize tracking variables
        episode_rewards, episode_lengths = [], []
        total_reward = 0

        # Setup for real-time plot updates
        training_plotter = visualizations.TrainingPlotter(window_size=10)

        # Reset the environment
        obs, _ = env.reset()

        # Training loop
        epsilon = dqn_config["exploration_eps"][
            "initial"
        ]  # Start with full exploration
        epsilon_decay = (
            dqn_config["exploration_eps"]["initial"]
            - dqn_config["exploration_eps"]["final"]
        ) / (training_config["timesteps"] * dqn_config["exploration_fraction"])

        for _step in range(training_config["timesteps"]):
            start_time = time.time()

            # Adjust epsilon based on the step
            epsilon = max(
                dqn_config["exploration_eps"]["final"], epsilon - epsilon_decay
            )

            # Choose action based on current epsilon
            if np.random.rand() < epsilon:
                action = env.action_space.sample()  # Explore
            else:
                action, _ = model.predict(obs, deterministic=True)  # Exploit

            obs, reward, terminated, truncated, _ = env.step(action)

            total_reward += reward

            if terminated or truncated:
                # Log the reward and episode length
                episode_rewards.append(total_reward)
                episode_lengths.append(len(episode_rewards))

                # Update the plot with the new data
                training_plotter.update(total_reward)

                # Reset the environment when the episode ends
                total_reward = 0
                obs, _ = env.reset()

            # Render the environment
            obs_tensor = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)
            q_values = model.policy.q_net(obs_tensor).detach().numpy()
            q_values = q_values.squeeze(0)

            # Map each possible action to the resulting cell
            snake_head = env.model.snake.head()

            actions = [
                Direction.UP,
                Direction.RIGHT,
                Direction.DOWN,
                Direction.LEFT,
            ]
            q_values_dict = {
                snake_head.move(action): q_value
                for action, q_value in zip(actions, q_values)
            }

            # Render the environment with Q-values for the 4 possible cells
            env.render(q_values=q_values_dict)

            # Maintain the desired FPS
            elapsed_time = time.time() - start_time
            sleep_time = max(0, time_per_step - elapsed_time)
            time.sleep(sleep_time)

    # Save the trained model
    model.save("dqn_snake")

    # Evaluate the model
    evaluate_and_print_results(model, env, evaluation_config["episodes"])

    # Close the environment and turn off interactive plotting
    env.close()


def evaluate_and_print_results(
    model: DQN, env: environment.SnakeGameEnv, n_eval_episodes: int = 10
) -> None:
    """Evaluates the trained model and prints the results."""
    mean_reward, std_reward = evaluate_policy(
        model, env, n_eval_episodes=n_eval_episodes
    )
    print(f"Mean reward: {mean_reward}, Std reward: {std_reward}")
