"""Module containing trainer class for training a Q-learning model. """
import torch
import torch.nn as nn
import torch.optim as optim
from noodle.entities import Direction


class QTrainer:
    """Trainer for training a Q-learning model.

    Attributes:
        model: The Q-learning model.
        lr: The learning rate for the optimizer.
        gamma: The discount factor for future rewards.
    """

    def __init__(self, model: nn.Module, lr: float, gamma: float) -> None:
        self.model = model
        self.lr = lr
        self.gamma = gamma

        self.optimizer = optim.Adam(model.parameters(), lr=self.lr)
        self.loss = nn.MSELoss()

    def train_step(self, state, action, reward, next_state, done):
        """Perform a single training step.

        Args:
            state: The current state.
            action: The action taken in the current state.
            reward (float): The reward received for taking the action.
            next_state (list or numpy.ndarray): The next state after taking the action.
            done (bool): Whether the episode is done after taking the action.

        """
        state = torch.tensor(state, dtype=torch.float)
        next_state = torch.tensor(next_state, dtype=torch.float)
        action = torch.tensor(action, dtype=torch.long)
        reward = torch.tensor(reward, dtype=torch.float)

        if len(state.shape) == 1:
            state = torch.unsqueeze(state, 0)
            next_state = torch.unsqueeze(next_state, 0)
            action = torch.unsqueeze(action, 0)
            reward = torch.unsqueeze(reward, 0)
            done = (done,)

        pred = self.model(state)
    
        target = pred.clone()
        for idx in range(len(done)):
            Q_new = reward[idx]
            if not done[idx]:
                Q_new = reward[idx] + self.gamma * torch.max(self.model(next_state[idx]))
            target[idx][torch.argmax(action[idx]).item()] = Q_new

        self.optimizer.zero_grad()
        loss = self.loss(target, pred)
        loss.backward()
        self.optimizer.step()
