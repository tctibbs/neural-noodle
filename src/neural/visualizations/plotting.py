"""
Plotting Module

This module defines the `TrainingPlotter` class, which facilitates real-time
visualization of training rewards for reinforcement learning agents.
"""

import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import gaussian_filter1d


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
        self.lower_bounds = []
        self.upper_bounds = []

        # Load custom style
        plt.style.use("fivethirtyeight")

        # Initialize the plot
        plt.ion()
        self.fig, self.ax = plt.subplots(figsize=figsize)

        # Plot elements
        self.scatter_points = self.ax.scatter(
            [],
            [],
            label="Raw Rewards",
            color="blue",
            alpha=0.4,
            s=8,
            marker="d",
        )
        if self.window_size > 1:
            (self.smoothed_line,) = self.ax.plot(
                [],
                [],
                label=f"Moving Average (window={self.window_size})",
                color="red",
                linewidth=2,
            )
        (self.lower_bound_line,) = self.ax.plot(
            [],
            [],
            label="1st Quartile",
            color="blue",
            linestyle="--",
            linewidth=1.5,
        )
        (self.upper_bound_line,) = self.ax.plot(
            [],
            [],
            label="3rd Quartile",
            color="blue",
            linestyle="--",
            linewidth=1.5,
        )

        # Configure plot
        self.ax.set_title("Training Rewards Over Time")
        self.ax.set_xlabel("Episode")
        self.ax.set_ylabel("Reward")
        self.ax.grid(True, linestyle="--", alpha=0.6)
        self.ax.spines["top"].set_visible(False)
        self.ax.spines["right"].set_visible(False)

        # Add legend
        self.ax.legend(
            loc="upper right", frameon=True, framealpha=0.8, edgecolor="gray"
        )

        plt.tight_layout()
        plt.show()

    def update(self, reward: float) -> None:
        """
        Update the plot with a new reward.

        Args:
            reward: The latest reward to add.
        """
        self.rewards.append(reward)

        # Update scatter points
        self.scatter_points.set_offsets(
            np.c_[range(len(self.rewards)), self.rewards]
        )

        # Update smoothed rewards if window_size > 1
        if self.window_size > 1:
            self.smoothed_rewards = self._moving_average(
                self.rewards, self.window_size
            )
            self.smoothed_line.set_data(
                range(len(self.smoothed_rewards)), self.smoothed_rewards
            )

        # Update quartile curves
        self._update_quartile_curves()

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

    def _update_quartile_curves(self) -> None:
        """
        Compute and update the curves for 1st and 3rd quartiles using a rolling window.
        """
        if len(self.rewards) < 2:
            return

        self.lower_bounds = []
        self.upper_bounds = []

        for i in range(len(self.rewards)):
            # Define the rolling window range
            start_idx = max(0, i - self.window_size + 1)
            current_window = self.rewards[start_idx : i + 1]

            # Compute the 1st and 3rd quartiles for the rolling window
            self.lower_bounds.append(np.percentile(current_window, 25))
            self.upper_bounds.append(np.percentile(current_window, 75))

        # Apply smoothing to the quartile curves
        self.lower_bounds = gaussian_filter1d(self.lower_bounds, sigma=2)
        self.upper_bounds = gaussian_filter1d(self.upper_bounds, sigma=2)

        # Update the plot data
        x_range = range(len(self.rewards))
        self.lower_bound_line.set_data(x_range, self.lower_bounds)
        self.upper_bound_line.set_data(x_range, self.upper_bounds)

    def save_plot(self, filename: str) -> None:
        """
        Save the current plot to a file.

        Args:
            filename: The path to save the plot image.
        """
        self.fig.savefig(filename)
