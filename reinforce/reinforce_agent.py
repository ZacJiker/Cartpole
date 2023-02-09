import torch 

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from datetime import datetime
from reinforce.policy_network import PolicyNetwork
from torch.distributions import Categorical

class ReinforceAgent(object):

    def __init__(self, obs_space_dims: int, action_space_dims: int):
        """Initializes an agent that learns a policy via REINFORCE algorithm [1]
        to solve the task at hand (Inverted Pendulum v4).

        Args:
            obs_space_dims: Dimension of the observation space
            action_space_dims: Dimension of the action space
        """

        # Hyperparameters
        self.learning_rate = 1e-4  # Learning rate for policy optimization
        self.gamma = 0.99  # Discount factor
        self.eps = 1e-6  # small number for mathematical stability

        self.probs = []  # Stores probability values of the sampled action
        self.rewards = []  # Stores the corresponding rewards

        self.net = PolicyNetwork(obs_space_dims, action_space_dims)
        self.optimizer = torch.optim.AdamW(self.net.parameters(), lr=self.learning_rate)

    def _discount_rewards(self):
        """Calculates the discounted rewards.

        Returns:
            discounted_rewards: Discounted rewards
        """
        discounted_rewards = []
        cumulative_reward = 0
        for reward in self.rewards[::-1]:
            cumulative_reward = reward + self.gamma * cumulative_reward
            discounted_rewards.insert(0, cumulative_reward)
        return discounted_rewards

    def _calculate_loss(self, discounted_rewards: list[float]):
        """Calculates the loss.

        Returns:
            loss: Loss
        """
        # Convert the lists to tensors
        discounted_rewards = torch.tensor(discounted_rewards)
        probs = torch.stack(self.probs)
        # Calculate the loss
        loss = torch.sum(-probs * discounted_rewards)
        return loss

    def sample_action(self, state: np.ndarray) -> float:
        """Returns an action, conditioned on the policy and observation.

        Args:
            state: Observation from the environment

        Returns:
            action: Action to be performed
        """
        # Convert the state to a tensor
        state = torch.tensor(np.array([state]))
        # Get the mean and standard deviation of the normal distribution
        action_prob = self.net(state)
        # Sample an action from the normal distributio
        action = Categorical(action_prob).sample()
        # Calculate the log probability of the sampled action
        action_prob = torch.log(action_prob.squeeze(0)[action] + self.eps)
        # Store the probability and the corresponding reward
        self.probs.append(action_prob)
        # Return the sampled action
        return action.item()

    def update(self):
        """Updates the policy network."""
        # Calculate the discounted rewards
        discounted_rewards = self._discount_rewards()
        # Calculate the loss
        loss = self._calculate_loss(discounted_rewards)
        # Update the policy network
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        # Reset the lists
        self.probs = []
        self.rewards = []

    def plot_reward_per_episode(self, rewards: list[float]):
        """Plots the reward per episode.
        
        Args:
            rewards: List of rewards per episode
        """
        df = pd.DataFrame(rewards, columns=["reward"])
        df["episode"] = df.index
        df["cumulative_reward"] = df["reward"].cumsum()
        df["cumulative_mean_reward"] = df["cumulative_reward"] / (df["episode"] + 1)
        sns.lineplot(x="episode", y="cumulative_mean_reward", data=df).set_title("Cumulative Mean Reward per Episode")
        plt.show()
        
    def save_model(self, path: str = None):
        """Saves the policy network.

        Args:
            path: Path to save the model
        """
        if path is None:
            path = "models/model_{}.h5".format(datetime.now().strftime('%Y%m%d_%H%M%S'))
        torch.save(self.net.state_dict(), path)

    def load_model(self, path: str):
        """Loads the policy network.

        Args:
            path: Path to load the model
        """
        self.net.load_state_dict(torch.load(path))