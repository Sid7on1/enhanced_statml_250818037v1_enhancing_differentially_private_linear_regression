import torch
import numpy as np
from typing import List, Dict, Tuple, Optional
from torch import nn
from torch.nn import functional as F

class PolicyNetwork(nn.Module):
    """
    Policy Network class for the enhanced differentially private linear regression agent.

    This network implements the policy gradient algorithm to update the agent's policy
    based on rewards received from the environment. It uses a feedforward neural network
    architecture with multiple layers to approximate the policy function.

    ...

    Attributes
    ----------
    input_size : int
        The dimension of the input feature vector.
    output_size : int
        The dimension of the output action vector.
    hidden_sizes : List[int]
        A list of integers representing the number of neurons in each hidden layer.
    device : torch.device
        The device on which the network is stored and computations are performed.
    pi : torch.distributions.Distribution
        A probability distribution object representing the current policy.
    optimizer : torch.optim.Optimizer
        The optimizer used to update the network weights during training.

    Methods
    -------
    forward(x)
        Perform forward pass through the network.
    select_action(state)
        Select an action based on the current state and the policy network.
    update_policy(states, actions, rewards)
        Update the policy network weights based on the given states, actions, and rewards.
    log_action_prob(action, log_prob)
        Store the log probability of the given action for use in policy gradient update.
    reset()
        Reset the policy network and clear any stored log probabilities.

    """

    def __init__(self, input_size: int, output_size: int, hidden_sizes: List[int], device: str = 'cpu'):
        """
        Initialize the PolicyNetwork module.

        Parameters
        ----------
        input_size : int
            The dimension of the input feature vector.
        output_size : int
            The dimension of the output action vector.
        hidden_sizes : List[int]
            A list of integers representing the number of neurons in each hidden layer.
        device : str, optional
            The device on which to store the network and perform computations, by default 'cpu'.

        """
        super(PolicyNetwork, self).__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.hidden_sizes = hidden_sizes
        self.device = torch.device(device)

        self.layers = nn.ModuleList()
        self.layers.append(nn.Linear(input_size, hidden_sizes[0]))
        for i in range(1, len(hidden_sizes)):
            self.layers.append(nn.Linear(hidden_sizes[i-1], hidden_sizes[i]))
        self.layers.append(nn.Linear(hidden_sizes[-1], output_size))

        self.pi = None
        self.optimizer = None
        self.log_probs = []

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Perform forward pass through the network.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape (batch_size, input_size).

        Returns
        -------
        torch.Tensor
            Output tensor of shape (batch_size, output_size) representing the predicted action.

        """
        for layer in self.layers:
            x = F.relu(layer(x))
        return x

    def select_action(self, state: np.ndarray) -> np.ndarray:
        """
        Select an action based on the current state and the policy network.

        Parameters
        ----------
        state : np.ndarray
            The current state of the environment, of shape (input_size,).

        Returns
        -------
        np.ndarray
            The selected action, of shape (output_size,).

        """
        state = torch.from_numpy(state).float().to(self.device)
        action_probs = self.pi.probs(self.forward(state))
        action = action_probs.multinomial(1).item()
        return action

    def update_policy(self, states: np.ndarray, actions: np.ndarray, rewards: np.ndarray) -> None:
        """
        Update the policy network weights based on the given states, actions, and rewards.

        This method implements the policy gradient algorithm to update the network weights.

        Parameters
        ----------
        states : np.ndarray
            Array of states of shape (batch_size, input_size).
        actions : np.ndarray
            Array of actions taken of shape (batch_size,).
        rewards : np.ndarray
            Array of rewards received of shape (batch_size,).

        """
        states = torch.from_numpy(states).float().to(self.device)
        actions = torch.from_numpy(actions).long().to(self.device)
        rewards = torch.from_numpy(rewards).float().to(self.device)

        action_probs = self.pi.probs(self.forward(states))
        log_probs = action_probs.gather(1, actions.unsqueeze(1)).squeeze(1)
        self.log_probs.append(log_probs)

        policy_loss = -torch.mean(log_probs * rewards)

        self.optimizer.zero_grad()
        policy_loss.backward()
        self.optimizer.step()

    def log_action_prob(self, action: int, log_prob: float) -> None:
        """
        Store the log probability of the given action for use in policy gradient update.

        Parameters
        ----------
        action : int
            The action taken.
        log_prob : float
            The log probability of the action.

        """
        self.log_probs.append(log_prob)

    def reset(self) -> None:
        """
        Reset the policy network and clear any stored log probabilities.

        """
        self.pi.reset()
        self.log_probs = []

class PolicyGradientAgent:
    """
    Policy Gradient Agent class for interacting with the environment.

    This agent uses a policy network to select actions and update its policy based on rewards.

    ...

    Attributes
    ----------
    policy : PolicyNetwork
        The policy network used to select actions and update the policy.
    device : torch.device
        The device on which the agent's computations are performed.

    Methods
    -------
    select_action(state)
        Select an action based on the current state.
    update_policy(rewards)
        Update the policy network weights based on the given rewards.
    train(states, actions, rewards)
        Train the agent on a batch of states, actions, and rewards.
    save(filepath)
        Save the agent's policy network weights to a file.
    load(filepath)
        Load the agent's policy network weights from a file.

    """

    def __init__(self, input_size: int, output_size: int, hidden_sizes: List[int], device: str = 'cpu'):
        """
        Initialize the PolicyGradientAgent class.

        Parameters
        ----------
        input_size : int
            The dimension of the input feature vector.
        output_size : int
            The dimension of the output action vector.
        hidden_sizes : List[int]
            A list of integers representing the number of neurons in each hidden layer.
        device : str, optional
            The device on which to perform computations, by default 'cpu'.

        """
        self.policy = PolicyNetwork(input_size, output_size, hidden_sizes, device)
        self.device = self.policy.device
        self.policy.pi = torch.distributions.Categorical(torch.zeros(self.policy.output_size).to(self.device))
        self.policy.optimizer = torch.optim.Adam(self.policy.parameters(), lr=0.01)

    def select_action(self, state: np.ndarray) -> np.ndarray:
        """
        Select an action based on the current state.

        Parameters
        ----------
        state : np.ndarray
            The current state of the environment, of shape (input_size,).

        Returns
        -------
        np.ndarray
            The selected action, of shape (output_size,).

        """
        return self.policy.select_action(state)

    def update_policy(self, states: np.ndarray, actions: np.ndarray, rewards: np.ndarray) -> None:
        """
        Update the policy network weights based on the given states, actions, and rewards.

        Parameters
        ----------
        states : np.ndarray
            Array of states of shape (batch_size, input_size).
        actions : np.ndarray
            Array of actions taken of shape (batch_size,).
        rewards : np.ndarray
            Array of rewards received of shape (batch_size,).

        """
        self.policy.update_policy(states, actions, rewards)

    def train(self, states: np.ndarray, actions: List[int], rewards: np.ndarray) -> None:
        """
        Train the agent on a batch of states, actions, and rewards.

        Parameters
        ----------
        states : np.ndarray
            Array of states of shape (batch_size, input_size).
        actions : List[int]
            List of actions taken of shape (batch_size,).
        rewards : np.ndarray
            Array of rewards received of shape (batch_size,).

        """
        # Convert actions list to numpy array
        actions_np = np.array(actions)

        # Update policy network weights
        self.update_policy(states, actions_np, rewards)

        # Reset policy network for next episode
        self.policy.reset()

    def save(self, filepath: str) -> None:
        """
        Save the agent's policy network weights to a file.

        Parameters
        ----------
        filepath : str
            The path to the file where the weights will be saved.

        """
        torch.save(self.policy.state_dict(), filepath)

    def load(self, filepath: str) -> None:
        """
        Load the agent's policy network weights from a file.

        Parameters
        ----------
        filepath : str
            The path to the file from which the weights will be loaded.

        """
        self.policy.load_state_dict(torch.load(filepath, map_location=self.device))

# Example usage
if __name__ == '__main__':
    input_size = 8
    output_size = 4
    hidden_sizes = [16, 32]

    agent = PolicyGradientAgent(input_size, output_size, hidden_sizes)

    # Example states, actions, and rewards
    states = np.random.random((1000, input_size))
    actions = [np.random.randint(0, output_size) for _ in range(1000)]
    rewards = np.random.random(1000)

    # Train the agent
    agent.train(states, actions, rewards)

    # Save and load agent weights
    filepath = 'policy_network_weights.pth'
    agent.save(filepath)
    agent.load(filepath)