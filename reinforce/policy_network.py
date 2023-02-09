import torch
import torch.nn as nn

class PolicyNetwork(nn.Module):
    """Parametrized Policy Network."""

    def __init__(self, obs_space_dims: int, action_space_dims: int):
        """Initializes a neural network that estimates the mean and standard deviation
         of a normal distribution from which an action is sampled from.

        Args:
            obs_space_dims: Dimension of the observation space
            action_space_dims: Dimension of the action space
        """
        super().__init__()

        hidden_space1 = 16  # Nothing special with 16
        hidden_space2 = 32  # Nothing special with 32
        hidden_space3 = 32  # Nothing special with 32
        hidden_space4 = 16  # Nothing special with 16

        # Shared Network
        self.shared_net = nn.Sequential(
            nn.Linear(obs_space_dims, hidden_space1),
            nn.ReLU(),
            nn.Linear(hidden_space1, hidden_space2),
            nn.ReLU(),
            nn.Linear(hidden_space2, hidden_space3),
            nn.ReLU(),
            nn.Linear(hidden_space3, hidden_space4),
            nn.ReLU(),
            nn.Linear(hidden_space4, action_space_dims), 
            nn.Softmax(dim=1)
        )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Conditioned on the observation, returns the mean and standard deviation
         of a normal distribution from which an action is sampled from.

        Args:
            x: Observation from the environment

        Returns:
            action_prob: Probability of the action
        """
        return self.shared_net(x.float())