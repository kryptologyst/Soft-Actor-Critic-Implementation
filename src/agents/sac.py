"""
Soft Actor-Critic (SAC) implementation with modern best practices.
"""

from typing import Dict, Optional, Tuple
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from gymnasium import Env

from .base_agent import BaseAgent
from src.utils.replay_buffer import ReplayBuffer


class GaussianPolicy(nn.Module):
    """
    Gaussian policy network with reparameterization trick.
    
    Args:
        state_dim: Dimension of state space
        action_dim: Dimension of action space
        hidden_dim: Hidden layer dimension
        max_action: Maximum action value
    """
    
    def __init__(
        self, 
        state_dim: int, 
        action_dim: int, 
        hidden_dim: int = 256, 
        max_action: float = 1.0
    ) -> None:
        super().__init__()
        
        self.max_action = max_action
        
        # Shared layers
        self.shared_layers = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        
        # Output layers
        self.mean_layer = nn.Linear(hidden_dim, action_dim)
        self.log_std_layer = nn.Linear(hidden_dim, action_dim)
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self) -> None:
        """Initialize network weights."""
        for layer in self.shared_layers:
            if isinstance(layer, nn.Linear):
                nn.init.orthogonal_(layer.weight, gain=np.sqrt(2))
                nn.init.constant_(layer.bias, 0)
        
        nn.init.orthogonal_(self.mean_layer.weight, gain=0.01)
        nn.init.constant_(self.mean_layer.bias, 0)
        
        nn.init.orthogonal_(self.log_std_layer.weight, gain=0.01)
        nn.init.constant_(self.log_std_layer.bias, -1)
    
    def forward(self, state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass through the policy network.
        
        Args:
            state: Input state tensor
            
        Returns:
            Tuple of (action, log_probability)
        """
        x = self.shared_layers(state)
        
        mean = self.mean_layer(x)
        log_std = torch.clamp(self.log_std_layer(x), -20, 2)
        std = torch.clamp(log_std.exp(), min=1e-6)
        
        # Check for NaN values
        if torch.isnan(mean).any() or torch.isnan(std).any():
            print(f"NaN detected in policy forward pass!")
            print(f"State: {state}")
            print(f"Mean: {mean}")
            print(f"Std: {std}")
            # Use fallback values
            mean = torch.zeros_like(mean)
            std = torch.ones_like(std)
        
        # Reparameterization trick
        normal = torch.distributions.Normal(mean, std)
        z = normal.rsample()
        action = torch.tanh(z) * self.max_action
        
        # Compute log probability with Jacobian correction
        log_prob = normal.log_prob(z) - torch.log(1 - action.pow(2) + 1e-7)
        log_prob = log_prob.sum(dim=1, keepdim=True)
        
        return action, log_prob
    
    def sample(self, state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Sample action from policy."""
        return self.forward(state)


class QNetwork(nn.Module):
    """
    Q-network for value function approximation.
    
    Args:
        state_dim: Dimension of state space
        action_dim: Dimension of action space
        hidden_dim: Hidden layer dimension
    """
    
    def __init__(
        self, 
        state_dim: int, 
        action_dim: int, 
        hidden_dim: int = 256
    ) -> None:
        super().__init__()
        
        self.q_network = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
        
        self._init_weights()
    
    def _init_weights(self) -> None:
        """Initialize network weights."""
        for layer in self.q_network:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
                nn.init.constant_(layer.bias, 0)
    
    def forward(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the Q-network.
        
        Args:
            state: State tensor
            action: Action tensor
            
        Returns:
            Q-value tensor
        """
        x = torch.cat([state, action], dim=1)
        return self.q_network(x)


class SACAgent(BaseAgent):
    """
    Soft Actor-Critic (SAC) agent implementation.
    
    SAC is an off-policy actor-critic algorithm that uses entropy regularization
    to encourage exploration and learn robust policies.
    """
    
    def __init__(
        self,
        env: Env,
        learning_rate: float = 3e-4,
        gamma: float = 0.99,
        tau: float = 0.005,
        alpha: float = 0.2,
        batch_size: int = 256,
        buffer_size: int = 1_000_000,
        hidden_dim: int = 256,
        device: str = "auto",
        seed: Optional[int] = None,
        **kwargs
    ) -> None:
        """
        Initialize SAC agent.
        
        Args:
            env: Environment to interact with
            learning_rate: Learning rate for all networks
            gamma: Discount factor
            tau: Soft update coefficient
            alpha: Entropy regularization coefficient
            batch_size: Batch size for training
            buffer_size: Size of replay buffer
            hidden_dim: Hidden layer dimension
            device: Device to run on
            seed: Random seed
        """
        super().__init__(env, device, seed, **kwargs)
        
        # Hyperparameters
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.tau = tau
        self.alpha = alpha
        self.batch_size = batch_size
        
        # Environment dimensions
        self.state_dim = env.observation_space.shape[0]
        self.action_dim = env.action_space.shape[0]
        self.max_action = float(env.action_space.high[0])
        
        # Networks
        self.actor = GaussianPolicy(
            self.state_dim, self.action_dim, hidden_dim, self.max_action
        ).to(self.device)
        
        self.q1 = QNetwork(self.state_dim, self.action_dim, hidden_dim).to(self.device)
        self.q2 = QNetwork(self.state_dim, self.action_dim, hidden_dim).to(self.device)
        
        # Target networks
        self.q1_target = QNetwork(self.state_dim, self.action_dim, hidden_dim).to(self.device)
        self.q2_target = QNetwork(self.state_dim, self.action_dim, hidden_dim).to(self.device)
        
        # Copy weights to target networks
        self.q1_target.load_state_dict(self.q1.state_dict())
        self.q2_target.load_state_dict(self.q2.state_dict())
        
        # Optimizers
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=learning_rate)
        self.q1_optimizer = optim.Adam(self.q1.parameters(), lr=learning_rate)
        self.q2_optimizer = optim.Adam(self.q2.parameters(), lr=learning_rate)
        
        # Replay buffer
        self.replay_buffer = ReplayBuffer(buffer_size, str(self.device))
        
        # Training metrics
        self.metrics = {}
    
    def select_action(
        self, 
        state: np.ndarray, 
        deterministic: bool = False
    ) -> np.ndarray:
        """
        Select action using the current policy.
        
        Args:
            state: Current state
            deterministic: Whether to select action deterministically
            
        Returns:
            Selected action
        """
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            if deterministic:
                # Use mean action for deterministic selection
                x = self.actor.shared_layers(state_tensor)
                mean = self.actor.mean_layer(x)
                action = torch.tanh(mean) * self.actor.max_action
            else:
                action, _ = self.actor.sample(state_tensor)
        
        return action.cpu().numpy().flatten()
    
    def update(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """
        Update the agent's networks.
        
        Args:
            batch: Batch of experience data
            
        Returns:
            Dictionary of training metrics
        """
        states = batch["states"]
        actions = batch["actions"]
        rewards = batch["rewards"]
        next_states = batch["next_states"]
        dones = batch["dones"]
        
        # Update Q-networks
        with torch.no_grad():
            next_actions, next_log_probs = self.actor.sample(next_states)
            target_q1 = self.q1_target(next_states, next_actions)
            target_q2 = self.q2_target(next_states, next_actions)
            target_q = torch.min(target_q1, target_q2) - self.alpha * next_log_probs
            target_q = rewards + self.gamma * (1 - dones) * target_q
        
        # Q-network losses
        current_q1 = self.q1(states, actions)
        current_q2 = self.q2(states, actions)
        
        q1_loss = F.mse_loss(current_q1, target_q)
        q2_loss = F.mse_loss(current_q2, target_q)
        
        # Update Q-networks
        self.q1_optimizer.zero_grad()
        q1_loss.backward()
        self.q1_optimizer.step()
        
        self.q2_optimizer.zero_grad()
        q2_loss.backward()
        self.q2_optimizer.step()
        
        # Update actor
        new_actions, log_probs = self.actor.sample(states)
        q1_values = self.q1(states, new_actions)
        actor_loss = (self.alpha * log_probs - q1_values).mean()
        
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()
        
        # Soft update target networks
        self._soft_update(self.q1_target, self.q1)
        self._soft_update(self.q2_target, self.q2)
        
        # Store metrics
        self.metrics = {
            "q1_loss": q1_loss.item(),
            "q2_loss": q2_loss.item(),
            "actor_loss": actor_loss.item(),
            "mean_q1": current_q1.mean().item(),
            "mean_q2": current_q2.mean().item(),
            "mean_log_prob": log_probs.mean().item()
        }
        
        return self.metrics
    
    def _soft_update(self, target: nn.Module, source: nn.Module) -> None:
        """
        Soft update target network parameters.
        
        Args:
            target: Target network
            source: Source network
        """
        for target_param, param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_(
                self.tau * param.data + (1 - self.tau) * target_param.data
            )
    
    def _store_experience(
        self, 
        state: np.ndarray, 
        action: np.ndarray, 
        reward: float, 
        next_state: np.ndarray, 
        done: bool
    ) -> None:
        """Store experience in replay buffer."""
        self.replay_buffer.add(state, action, reward, next_state, done)
    
    def _should_update(self) -> bool:
        """Check if agent should update."""
        return self.replay_buffer.is_ready(self.batch_size)
    
    def _get_batch(self) -> Dict[str, torch.Tensor]:
        """Get batch for training."""
        return self.replay_buffer.sample(self.batch_size)
    
    def save(self, filepath: str) -> None:
        """
        Save agent state.
        
        Args:
            filepath: Path to save the model
        """
        torch.save({
            "actor_state_dict": self.actor.state_dict(),
            "q1_state_dict": self.q1.state_dict(),
            "q2_state_dict": self.q2.state_dict(),
            "q1_target_state_dict": self.q1_target.state_dict(),
            "q2_target_state_dict": self.q2_target.state_dict(),
            "actor_optimizer_state_dict": self.actor_optimizer.state_dict(),
            "q1_optimizer_state_dict": self.q1_optimizer.state_dict(),
            "q2_optimizer_state_dict": self.q2_optimizer.state_dict(),
            "training_step": self.training_step,
            "episode_count": self.episode_count,
            "metrics": self.metrics
        }, filepath)
    
    def load(self, filepath: str) -> None:
        """
        Load agent state.
        
        Args:
            filepath: Path to load the model from
        """
        checkpoint = torch.load(filepath, map_location=self.device)
        
        self.actor.load_state_dict(checkpoint["actor_state_dict"])
        self.q1.load_state_dict(checkpoint["q1_state_dict"])
        self.q2.load_state_dict(checkpoint["q2_state_dict"])
        self.q1_target.load_state_dict(checkpoint["q1_target_state_dict"])
        self.q2_target.load_state_dict(checkpoint["q2_target_state_dict"])
        
        self.actor_optimizer.load_state_dict(checkpoint["actor_optimizer_state_dict"])
        self.q1_optimizer.load_state_dict(checkpoint["q1_optimizer_state_dict"])
        self.q2_optimizer.load_state_dict(checkpoint["q2_optimizer_state_dict"])
        
        self.training_step = checkpoint["training_step"]
        self.episode_count = checkpoint["episode_count"]
        self.metrics = checkpoint["metrics"]
