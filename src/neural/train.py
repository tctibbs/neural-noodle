"""Trains the noodle."""

import matplotlib.pyplot as plt
import numpy as np
from stable_baselines3 import DQN
from stable_baselines3.common.evaluation import evaluate_policy

from src.neural import SnakeGameEnv


def setup_plot() -> (
    tuple[plt.Figure, plt.Axes, plt.Axes, plt.Line2D, plt.Line2D]
):
    """Sets up the real-time plotting environment."""
    plt.ion()  # Interactive mode on
    fig, (reward_ax, length_ax) = plt.subplots(1, 2, figsize=(12, 6))

    # Initialize lines for rewards and lengths
    (reward_line,) = reward_ax.plot([], [], label="Episode Rewards")
    (length_line,) = length_ax.plot([], [], label="Episode Lengths")

    # Configure axes for rewards
    reward_ax.set_xlabel("Episodes")
    reward_ax.set_ylabel("Rewards")
    reward_ax.set_title("Episode Rewards Over Time")
    reward_ax.legend()

    # Configure axes for lengths
    length_ax.set_xlabel("Episodes")
    length_ax.set_ylabel("Episode Lengths")
    length_ax.set_title("Episode Lengths Over Time")
    length_ax.legend()

    plt.tight_layout()

    return fig, reward_ax, length_ax, reward_line, length_line


def update_plot(
    episode_rewards: list[float],
    episode_lengths: list[int],
    reward_line: plt.Line2D,
    length_line: plt.Line2D,
) -> None:
    """Updates the plot in real-time with new episode data."""
    reward_line.set_xdata(range(len(episode_rewards)))
    reward_line.set_ydata(episode_rewards)

    length_line.set_xdata(range(len(episode_lengths)))
    length_line.set_ydata(episode_lengths)

    # Rescale the axes to accommodate new data
    reward_line.axes.relim()
    reward_line.axes.autoscale_view()

    length_line.axes.relim()
    length_line.axes.autoscale_view()

    # Redraw the plot
    plt.draw()
    plt.pause(0.01)


def create_snake_env(env_config: dict) -> SnakeGameEnv:
    """Returns the Snake game environment."""
    return SnakeGameEnv(
        cols=env_config["grid_size"],
        rows=env_config["grid_size"],
        width=env_config["width"],
        height=env_config["height"],
        fps=env_config["fps"],
    )


def create_dqn_model(env: SnakeGameEnv, dqn_config: dict) -> DQN:
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


def train_snake_dqn(config: dict) -> None:
    """Trains a DQN model on the Snake game and evaluates its performance."""
    # Extract relevant configuration sections
    env_config = config["environment"]
    dqn_config = config["dqn"]
    training_config = config["training"]
    evaluation_config = config["evaluation"]

    # Create environment and DQN model
    env = create_snake_env(env_config)
    model = create_dqn_model(env, dqn_config)

    # Use the built-in learn method if specified
    if training_config["use_builtin"]:
        model.learn(total_timesteps=training_config["timesteps"])
    else:
        # Initialize tracking variables
        episode_rewards, episode_lengths = [], []
        total_reward = 0

        # Setup for real-time plot updates
        fig, reward_ax, length_ax, reward_line, length_line = setup_plot()

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
                update_plot(
                    episode_rewards, episode_lengths, reward_line, length_line
                )

                # Reset the environment when the episode ends
                total_reward = 0
                obs, _ = env.reset()

            # Render the environment
            env.render()

    # Save the trained model
    model.save("dqn_snake")

    # Evaluate the model
    evaluate_and_print_results(model, env, evaluation_config["episodes"])

    # Close the environment and turn off interactive plotting
    env.close()
    plt.ioff()
    plt.show()


def evaluate_and_print_results(
    model: DQN, env: SnakeGameEnv, n_eval_episodes: int = 10
) -> None:
    """Evaluates the trained model and prints the results."""
    mean_reward, std_reward = evaluate_policy(
        model, env, n_eval_episodes=n_eval_episodes
    )
    print(f"Mean reward: {mean_reward}, Std reward: {std_reward}")
