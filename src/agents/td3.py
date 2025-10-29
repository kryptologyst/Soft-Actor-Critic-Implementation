"""
Twin Delayed Deep Deterministic Policy Gradient (TD3) implementation.
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


class DeterministicPolicy(nn.Module):
    """
    Deterministic policy network for TD3.
    
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
        
        self.network = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
            nn.Tanh()
        )
        
        self._init_weights()
    
    def _init_weights(self) -> None:
        """Initialize network weights."""
        for layer in self.network:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
                nn.init.constant_(layer.bias, 0)
    
    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the policy network.
        
        Args:
            state: Input state tensor
            
        Returns:
            Action tensor
        """
        return self.network(state) * self.max_action


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


class TD3Agent(BaseAgent):
    """
    Twin Delayed Deep Deterministic Policy Gradient (TD3) agent implementation.
    
    TD3 is an off-policy algorithm that addresses overestimation bias in DDPG
    by using twin Q-networks and delayed policy updates.
    """
    
    def __init__(
        self,
        env: Env,
        learning_rate: float = 3e-4,
        gamma: float = 0.99,
        tau: float = 0.005,
        policy_noise: float = 0.2,
        noise_clip: float = 0.5,
        policy_delay: int = 2,
        batch_size: int = 256,
        buffer_size: int = 1_000_000,
        hidden_dim: int = 256,
        device: str = "auto",
        seed: Optional[int] = None,
        **kwargs
    ) -> None:
        """
        Initialize TD3 agent.
        
        Args:
            env: Environment to interact with
            learning_rate: Learning rate for all networks
            gamma: Discount factor
            tau: Soft update coefficient
            policy_noise: Noise added to target policy
            noise_clip: Noise clipping range
            policy_delay: Policy update delay
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
        self.policy_noise = policy_noise
        self.noise_clip = noise_clip
        self.policy_delay = policy_delay
        self.batch_size = batch_size
        
        # Environment dimensions
        self.state_dim = env.observation_space.shape[0]
        self.action_dim = env.action_space.shape[0]
        self.max_action = float(env.action_space.high[0])
        
        # Networks
        self.actor = DeterministicPolicy(
            self.state_dim, self.action_dim, hidden_dim, self.max_action
        ).to(self.device)
        
        self.critic1 = QNetwork(self.state_dim, self.action_dim, hidden_dim).to(self.device)
        self.critic2 = QNetwork(self.state_dim, self.action_dim, hidden_dim).to(self.device)
        
        # Target networks
        self.actor_target = DeterministicPolicy(
            self.state_dim, self.action_dim, hidden_dim, self.max_action
        ).to(self.device)
        self.critic1_target = QNetwork(self.state_dim, self.action_dim, hidden_dim).to(self.device)
        self.critic2_target = QNetwork(self.state_dim, self.action_dim, hidden_dim).to(self.device)
        
        # Copy weights to target networks
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.critic1_target.load_state_dict(self.critic1.state_dict())
        self.critic2_target.load_state_dict(self.critic2.state_dict())
        
        # Optimizers
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=learning_rate)
        self.critic1_optimizer = optim.Adam(self.critic1.parameters(), lr=learning_rate)
        self.critic2_optimizer = optim.Adam(self.critic2.parameters(), lr=learning_rate)
        
        # Replay buffer
        self.replay_buffer = ReplayBuffer(buffer_size, str(self.device))
        
        # Training metrics
        self.metrics = {}
        self.update_count = 0
    
    def select_action(
        self, 
        state: np.ndarray, 
        deterministic: bool = False,
        add_noise: bool = True
    ) -> np.ndarray:
        """
        Select action using the current policy.
        
        Args:
            state: Current state
            deterministic: Whether to select action deterministically
            add_noise: Whether to add exploration noise
            
        Returns:
            Selected action
        """
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            action = self.actor(state_tensor)
            
            if add_noise and not deterministic:
                noise = torch.randn_like(action) * 0.1
                action = torch.clamp(action + noise, -self.max_action, self.max_action)
        
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
        
        # Update critics
        with torch.no_grad():
            # Target policy smoothing
            noise = torch.randn_like(actions) * self.policy_noise
            noise = torch.clamp(noise, -self.noise_clip, self.noise_clip)
            
            next_actions = self.actor_target(next_states) + noise
            next_actions = torch.clamp(next_actions, -self.max_action, self.max_action)
            
            # Target Q-values
            target_q1 = self.critic1_target(next_states, next_actions)
            target_q2 = self.critic2_target(next_states, next_actions)
            target_q = torch.min(target_q1, target_q2)
            target_q = rewards + self.gamma * (1 - dones) * target_q
        
        # Current Q-values
        current_q1 = self.critic1(states, actions)
        current_q2 = self.critic2(states, actions)
        
        # Critic losses
        critic1_loss = F.mse_loss(current_q1, target_q)
        critic2_loss = F.mse_loss(current_q2, target_q)
        
        # Update critics
        self.critic1_optimizer.zero_grad()
        critic1_loss.backward()
        self.critic1_optimizer.step()
        
        self.critic2_optimizer.zero_grad()
        critic2_loss.backward()
        self.critic2_optimizer.step()
        
        # Update actor (delayed)
        actor_loss = 0
        if self.update_count % self.policy_delay == 0:
            actor_loss = -self.critic1(states, self.actor(states)).mean()
            
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()
            
            # Soft update target networks
            self._soft_update(self.actor_target, self.actor)
            self._soft_update(self.critic1_target, self.critic1)
            self._soft_update(self.critic2_target, self.critic2)
        
        self.update_count += 1
        
        # Store metrics
        self.metrics = {
            "critic1_loss": critic1_loss.item(),
            "critic2_loss": critic2_loss.item(),
            "actor_loss": actor_loss.item() if isinstance(actor_loss, torch.Tensor) else actor_loss,
            "mean_q1": current_q1.mean().item(),
            "mean_q2": current_q2.mean().item(),
            "target_q": target_q.mean().item()
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
            "critic1_state_dict": self.critic1.state_dict(),
            "critic2_state_dict": self.critic2.state_dict(),
            "actor_target_state_dict": self.actor_target.state_dict(),
            "critic1_target_state_dict": self.critic1_target.state_dict(),
            "critic2_target_state_dict": self.critic2_target.state_dict(),
            "actor_optimizer_state_dict": self.actor_optimizer.state_dict(),
            "critic1_optimizer_state_dict": self.critic1_optimizer.state_dict(),
            "critic2_optimizer_state_dict": self.critic2_optimizer.state_dict(),
            "training_step": self.training_step,
            "episode_count": self.episode_count,
            "update_count": self.update_count,
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
        self.critic1.load_state_dict(checkpoint["critic1_state_dict"])
        self.critic2.load_state_dict(checkpoint["critic2_state_dict"])
        self.actor_target.load_state_dict(checkpoint["actor_target_state_dict"])
        self.critic1_target.load_state_dict(checkpoint["critic1_target_state_dict"])
        self.critic2_target.load_state_dict(checkpoint["critic2_target_state_dict"])
        
        self.actor_optimizer.load_state_dict(checkpoint["actor_optimizer_state_dict"])
        self.critic1_optimizer.load_state_dict(checkpoint["critic1_optimizer_state_dict"])
        self.critic2_optimizer.load_state_dict(checkpoint["critic2_optimizer_state_dict"])
        
        self.training_step = checkpoint["training_step"]
        self.episode_count = checkpoint["episode_count"]
        self.update_count = checkpoint["update_count"]
        self.metrics = checkpoint["metrics"]
