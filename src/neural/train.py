"""Trains the noodle."""
from stable_baselines3 import DQN
from stable_baselines3.common.evaluation import evaluate_policy

from src.neural import SnakeGameEnv


def train_snake_dqn(timesteps: int = 10000) -> None:
    """Trains a DQN model on the Snake game and evaluates its performance."""

    # Create the custom Snake environment
    env = SnakeGameEnv(width=400, height=400, cell_size=25, fps=120)

    # Set up the DQN model
    model = DQN(
        "MlpPolicy",
        env,
        verbose=1,
        buffer_size=50000,
        learning_rate=1e-3,
        batch_size=32,
        exploration_fraction=0.2,
        target_update_interval=1000,
    )

    # Train the model for the defined number of timesteps with visualization
    obs, _ = env.reset()
    for _ in range(timesteps):
        action, _ = model.predict(obs, deterministic=False)
        obs, reward, terminated, truncated, _ = env.step(action)  # type: ignore

        # Reset the environment when episode ends
        if terminated or truncated:
            obs, _ = env.reset()

        env.render()

    # Save the trained model
    model.save("dqn_snake")

    # Evaluate the trained model
    mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=10)
    print(f"Mean reward: {mean_reward}, Std reward: {std_reward}")

    # Close the environment
    env.close()


if __name__ == "__main__":
    train_snake_dqn(timesteps=10000)
