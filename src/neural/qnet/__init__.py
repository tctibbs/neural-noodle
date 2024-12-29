"""
QNet Package.

This package provides an interface for initializing and configuring 
the components required for Q-learning, including the `Model` and `Agent` 
classes. It handles the setup of the neural network model and reinforcement 
learning agent.
"""


from . model import Model
from . agent import Agent

__all__ = ["Model", "Agent"]
