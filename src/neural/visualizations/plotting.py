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
        self.ax.legend()
        self.ax.grid(True)
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
        if self.window_size > 1 and len(self.rewards) >= self.window_size:
            self.smoothed_rewards = self._moving_average(
                self.rewards, self.window_size
            )
            # Align the x-axis for smoothed data
            self.smoothed_line.set_data(
                range(self.window_size - 1, len(self.rewards)),
                self.smoothed_rewards,
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
        Compute the moving average using a sliding window.

        Args:
            data: The data points to average.
            window_size: The size of the moving window.

        Returns:
            The smoothed data.
        """
        return np.convolve(data, np.ones(window_size), "valid") / window_size

    def save_plot(self, filename: str) -> None:
        """
        Save the current plot to a file.

        Args:
            The path to save the plot image.
        """
        self.fig.savefig(filename)
