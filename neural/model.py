"""Module defining the neural network architecture."""
import torch
import torch.nn as nn
import torch.nn.functional as F
import os


class Linear_QNet(nn.Module):
    """Linear Q-Network.
    
    Attributes:
        input_size: The input size.
        hidden_size: The hidden size.
        output_size: The output size.
    """
    def __init__(self, input_size: int, hidden_size: int, output_size: int) -> None:
        super().__init__()
        self.linear1 = nn.Linear(input_size, hidden_size)
        self.linear2 = nn.Linear(hidden_size, output_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        x = F.relu(self.linear1(x))
        x = self.linear2(x)
        # print(x)
        return x

    def save(self, file_name: str = 'model.pth') -> None:
        """Saves the model."""
        model_folder_path = './model'
        if not os.path.exists(model_folder_path):
            os.makedirs(model_folder_path)

        file_name = os.path.join(model_folder_path, file_name)
        torch.save(self.state_dict(), file_name)
    