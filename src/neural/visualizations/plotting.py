"""
Plotting Module

This module defines the `TrainingPlotter` class, which facilitates real-time
visualization of training rewards for reinforcement learning agents.
"""

import matplotlib.pyplot as plt
import numpy as np


class TrainingPlotter:
    """
    A class responsible for plotting rewards over time during training.
    """

    def __init__(self, window_size: int = 1, figsize: tuple = (10, 6)) -> None:
        """
        Initialize the TrainingPlotter.

        Args:
            window_size: Size of the moving average window.
            figsize: Size of the matplotlib figure.
        """
        self.window_size = window_size
        self.rewards = []
        self.smoothed_rewards = []

        # Initialize the plot
        plt.ion()
        self.fig, self.ax = plt.subplots(figsize=figsize)

        # Plot elements
        (self.raw_line,) = self.ax.plot(
            [], [], label="Raw Rewards", color="blue", alpha=0.5
        )
        if self.window_size > 1:
            (self.smoothed_line,) = self.ax.plot(
                [],
                [],
                label=f"Moving Average (window={self.window_size})",
                color="red",
                linewidth=2,
            )

        # Configure plot
        self.ax.set_title("Training Rewards Over Time")
        self.ax.set_xlabel("Episode")
        self.ax.set_ylabel("Reward")
        self.ax.grid(True, linestyle="--", alpha=0.6)  # Improved gridlines
        self.ax.spines["top"].set_visible(False)
        self.ax.spines["right"].set_visible(False)
        plt.tight_layout()
        plt.show()

    def update(self, reward: float) -> None:
        """
        Update the plot with a new reward.

        Args:
            reward: The latest reward to add.
        """
        self.rewards.append(reward)
        self.raw_line.set_data(range(len(self.rewards)), self.rewards)

        # Update smoothed rewards if window_size > 1
        if self.window_size > 1:
            self.smoothed_rewards = self._moving_average(
                self.rewards, self.window_size
            )
            # Align the x-axis for smoothed data
            self.smoothed_line.set_data(
                range(len(self.smoothed_rewards)), self.smoothed_rewards
            )

        # Adjust plot limits
        self.ax.relim()
        self.ax.autoscale_view()

        # Redraw the plot
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()

    def _moving_average(
        self, data: list[float], window_size: int
    ) -> np.ndarray:
        """
        Compute the moving average, adjusting for available data.

        Args:
            data: The data points to average.
            window_size: The size of the moving window.

        Returns:
            The smoothed data.
        """
        smoothed = []
        for i in range(len(data)):
            current_window_size = min(window_size, i + 1)
            smoothed.append(np.mean(data[i - current_window_size + 1 : i + 1]))
        return np.array(smoothed)

    def save_plot(self, filename: str) -> None:
        """
        Save the current plot to a file.

        Args:
            filename: The path to save the plot image.
        """
        self.fig.savefig(filename)
