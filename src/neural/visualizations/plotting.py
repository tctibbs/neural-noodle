"""
Plotting Module.

This module provides a single class, TrainingPlotter, that sets up Matplotlib
plots for real-time visualization of Snake training and updates them with new
data. No separate references to reward/length lines are needed.
"""

import matplotlib.pyplot as plt


class TrainingPlotter:
    """
    A self-contained class for setting up and updating the Snake training plots.

    Example usage:
        plotter = TrainingPlotter()
        ...
        plotter.update(episode_rewards, episode_lengths)
    """

    def __init__(self) -> None:
        """Initialize the figure, axes, and plot lines."""
        plt.ion()  # Interactive mode on
        self.fig, (self.reward_ax, self.length_ax) = plt.subplots(
            1, 2, figsize=(12, 6)
        )

        # Create line objects for rewards and lengths
        (self.reward_line,) = self.reward_ax.plot(
            [], [], label="Episode Rewards"
        )
        (self.length_line,) = self.length_ax.plot(
            [], [], label="Episode Lengths"
        )

        # Configure reward axis
        self.reward_ax.set_xlabel("Episodes")
        self.reward_ax.set_ylabel("Rewards")
        self.reward_ax.set_title("Episode Rewards Over Time")
        self.reward_ax.legend()

        # Configure length axis
        self.length_ax.set_xlabel("Episodes")
        self.length_ax.set_ylabel("Episode Lengths")
        self.length_ax.set_title("Episode Lengths Over Time")
        self.length_ax.legend()

        plt.tight_layout()

    def update(
        self, episode_rewards: list[float], episode_lengths: list[int]
    ) -> None:
        """
        Updates the training plots in real-time with new episode data.

        :param episode_rewards: A list of cumulative rewards per episode.
        :param episode_lengths: A list of lengths (number of steps) per episode.
        """
        self.reward_line.set_xdata(range(len(episode_rewards)))
        self.reward_line.set_ydata(episode_rewards)

        self.length_line.set_xdata(range(len(episode_lengths)))
        self.length_line.set_ydata(episode_lengths)

        # Rescale axes to accommodate new data
        if self.reward_line.axes is not None:
            self.reward_line.axes.relim()
            self.reward_line.axes.autoscale_view()

        if self.length_line.axes is not None:
            self.length_line.axes.relim()
            self.length_line.axes.autoscale_view()

        plt.draw()
        plt.pause(0.01)
