"""
Proximal Policy Optimization (PPO) implementation.
"""

from typing import Dict, List, Optional, Tuple
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from gymnasium import Env

from .base_agent import BaseAgent


class PPOPolicy(nn.Module):
    """
    PPO policy network with value function.
    
    Args:
        state_dim: Dimension of state space
        action_dim: Dimension of action space
        hidden_dim: Hidden layer dimension
        continuous: Whether action space is continuous
        max_action: Maximum action value (for continuous actions)
    """
    
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_dim: int = 64,
        continuous: bool = True,
        max_action: float = 1.0
    ) -> None:
        super().__init__()
        
        self.continuous = continuous
        self.max_action = max_action
        
        # Shared layers
        self.shared_layers = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh()
        )
        
        if continuous:
            # Continuous action space
            self.mean_layer = nn.Linear(hidden_dim, action_dim)
            self.log_std_layer = nn.Linear(hidden_dim, action_dim)
        else:
            # Discrete action space
            self.action_layer = nn.Linear(hidden_dim, action_dim)
        
        # Value function
        self.value_layer = nn.Linear(hidden_dim, 1)
        
        self._init_weights()
    
    def _init_weights(self) -> None:
        """Initialize network weights."""
        for layer in self.shared_layers:
            if isinstance(layer, nn.Linear):
                nn.init.orthogonal_(layer.weight, gain=np.sqrt(2))
                nn.init.constant_(layer.bias, 0)
        
        if self.continuous:
            nn.init.orthogonal_(self.mean_layer.weight, gain=0.01)
            nn.init.constant_(self.mean_layer.bias, 0)
            nn.init.orthogonal_(self.log_std_layer.weight, gain=0.01)
            nn.init.constant_(self.log_std_layer.bias, 0)
        else:
            nn.init.orthogonal_(self.action_layer.weight, gain=0.01)
            nn.init.constant_(self.action_layer.bias, 0)
        
        nn.init.orthogonal_(self.value_layer.weight, gain=1.0)
        nn.init.constant_(self.value_layer.bias, 0)
    
    def forward(self, state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass through the policy network.
        
        Args:
            state: Input state tensor
            
        Returns:
            Tuple of (action, log_prob, value)
        """
        x = self.shared_layers(state)
        value = self.value_layer(x)
        
        if self.continuous:
            mean = self.mean_layer(x)
            log_std = torch.clamp(self.log_std_layer(x), -20, 2)
            std = log_std.exp()
            
            normal = torch.distributions.Normal(mean, std)
            action = normal.rsample()
            action = torch.tanh(action) * self.max_action
            
            log_prob = normal.log_prob(action) - torch.log(1 - action.pow(2) + 1e-7)
            log_prob = log_prob.sum(dim=1, keepdim=True)
        else:
            logits = self.action_layer(x)
            dist = torch.distributions.Categorical(logits=logits)
            action = dist.sample()
            log_prob = dist.log_prob(action).unsqueeze(1)
        
        return action, log_prob, value
    
    def evaluate(self, state: torch.Tensor, action: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Evaluate action for given state.
        
        Args:
            state: Input state tensor
            action: Action tensor
            
        Returns:
            Tuple of (log_prob, value, entropy)
        """
        x = self.shared_layers(state)
        value = self.value_layer(x)
        
        if self.continuous:
            mean = self.mean_layer(x)
            log_std = torch.clamp(self.log_std_layer(x), -20, 2)
            std = log_std.exp()
            
            normal = torch.distributions.Normal(mean, std)
            log_prob = normal.log_prob(action) - torch.log(1 - action.pow(2) + 1e-7)
            log_prob = log_prob.sum(dim=1, keepdim=True)
            entropy = normal.entropy().sum(dim=1, keepdim=True)
        else:
            logits = self.action_layer(x)
            dist = torch.distributions.Categorical(logits=logits)
            log_prob = dist.log_prob(action.squeeze()).unsqueeze(1)
            entropy = dist.entropy().unsqueeze(1)
        
        return log_prob, value, entropy


class PPOAgent(BaseAgent):
    """
    Proximal Policy Optimization (PPO) agent implementation.
    
    PPO is an on-policy algorithm that uses clipped objective to ensure
    stable policy updates.
    """
    
    def __init__(
        self,
        env: Env,
        learning_rate: float = 3e-4,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        clip_ratio: float = 0.2,
        value_coef: float = 0.5,
        entropy_coef: float = 0.01,
        max_grad_norm: float = 0.5,
        n_epochs: int = 4,
        batch_size: int = 64,
        hidden_dim: int = 64,
        device: str = "auto",
        seed: Optional[int] = None,
        **kwargs
    ) -> None:
        """
        Initialize PPO agent.
        
        Args:
            env: Environment to interact with
            learning_rate: Learning rate
            gamma: Discount factor
            gae_lambda: GAE lambda parameter
            clip_ratio: PPO clipping ratio
            value_coef: Value function loss coefficient
            entropy_coef: Entropy bonus coefficient
            max_grad_norm: Maximum gradient norm for clipping
            n_epochs: Number of epochs per update
            batch_size: Batch size for training
            hidden_dim: Hidden layer dimension
            device: Device to run on
            seed: Random seed
        """
        super().__init__(env, device, seed, **kwargs)
        
        # Hyperparameters
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_ratio = clip_ratio
        self.value_coef = value_coef
        self.entropy_coef = entropy_coef
        self.max_grad_norm = max_grad_norm
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        
        # Environment dimensions
        self.state_dim = env.observation_space.shape[0]
        self.action_dim = env.action_space.shape[0]
        self.continuous = hasattr(env.action_space, 'high')
        self.max_action = float(env.action_space.high[0]) if self.continuous else None
        
        # Policy network
        self.policy = PPOPolicy(
            self.state_dim, 
            self.action_dim, 
            hidden_dim, 
            self.continuous, 
            self.max_action
        ).to(self.device)
        
        # Optimizer
        self.optimizer = optim.Adam(self.policy.parameters(), lr=learning_rate)
        
        # Experience storage
        self.states = []
        self.actions = []
        self.rewards = []
        self.values = []
        self.log_probs = []
        self.dones = []
        
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
            if deterministic and self.continuous:
                # Use mean action for deterministic selection
                x = self.policy.shared_layers(state_tensor)
                mean = self.policy.mean_layer(x)
                action = torch.tanh(mean) * self.policy.max_action
            else:
                action, log_prob, value = self.policy(state_tensor)
        
        return action.cpu().numpy().flatten()
    
    def store_experience(
        self,
        state: np.ndarray,
        action: np.ndarray,
        reward: float,
        value: float,
        log_prob: float,
        done: bool
    ) -> None:
        """
        Store experience for PPO update.
        
        Args:
            state: Current state
            action: Action taken
            reward: Reward received
            value: Value estimate
            log_prob: Log probability of action
            done: Whether episode is done
        """
        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)
        self.values.append(value)
        self.log_probs.append(log_prob)
        self.dones.append(done)
    
    def compute_gae(self, next_value: float) -> Tuple[List[float], List[float]]:
        """
        Compute Generalized Advantage Estimation (GAE).
        
        Args:
            next_value: Value of next state
            
        Returns:
            Tuple of (advantages, returns)
        """
        advantages = []
        returns = []
        
        # Add next value for bootstrap
        values = self.values + [next_value]
        
        gae = 0
        for t in reversed(range(len(self.rewards))):
            delta = self.rewards[t] + self.gamma * values[t + 1] * (1 - self.dones[t]) - values[t]
            gae = delta + self.gamma * self.gae_lambda * (1 - self.dones[t]) * gae
            advantages.insert(0, gae)
            returns.insert(0, gae + values[t])
        
        return advantages, returns
    
    def update(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """
        Update the agent's policy.
        
        Args:
            batch: Batch of experience data (not used in PPO)
            
        Returns:
            Dictionary of training metrics
        """
        if len(self.states) == 0:
            return {}
        
        # Convert to tensors
        states = torch.FloatTensor(np.array(self.states)).to(self.device)
        actions = torch.FloatTensor(np.array(self.actions)).to(self.device)
        old_log_probs = torch.FloatTensor(np.array(self.log_probs)).to(self.device)
        old_values = torch.FloatTensor(np.array(self.values)).to(self.device)
        
        # Compute advantages and returns
        with torch.no_grad():
            _, _, next_value = self.policy(states[-1:])
            advantages, returns = self.compute_gae(next_value.item())
            advantages = torch.FloatTensor(advantages).to(self.device)
            returns = torch.FloatTensor(returns).to(self.device)
            
            # Normalize advantages
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        # Training loop
        total_policy_loss = 0
        total_value_loss = 0
        total_entropy_loss = 0
        
        for _ in range(self.n_epochs):
            # Create mini-batches
            indices = torch.randperm(len(states))
            
            for start_idx in range(0, len(states), self.batch_size):
                end_idx = min(start_idx + self.batch_size, len(states))
                batch_indices = indices[start_idx:end_idx]
                
                batch_states = states[batch_indices]
                batch_actions = actions[batch_indices]
                batch_old_log_probs = old_log_probs[batch_indices]
                batch_old_values = old_values[batch_indices]
                batch_advantages = advantages[batch_indices]
                batch_returns = returns[batch_indices]
                
                # Forward pass
                log_probs, values, entropy = self.policy.evaluate(batch_states, batch_actions)
                
                # Compute ratios
                ratio = torch.exp(log_probs - batch_old_log_probs)
                
                # Policy loss (clipped)
                surr1 = ratio * batch_advantages
                surr2 = torch.clamp(ratio, 1 - self.clip_ratio, 1 + self.clip_ratio) * batch_advantages
                policy_loss = -torch.min(surr1, surr2).mean()
                
                # Value loss
                value_loss = F.mse_loss(values, batch_returns)
                
                # Entropy loss
                entropy_loss = -entropy.mean()
                
                # Total loss
                total_loss = policy_loss + self.value_coef * value_loss + self.entropy_coef * entropy_loss
                
                # Backward pass
                self.optimizer.zero_grad()
                total_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
                self.optimizer.step()
                
                total_policy_loss += policy_loss.item()
                total_value_loss += value_loss.item()
                total_entropy_loss += entropy_loss.item()
        
        # Clear experience buffer
        self.states.clear()
        self.actions.clear()
        self.rewards.clear()
        self.values.clear()
        self.log_probs.clear()
        self.dones.clear()
        
        # Store metrics
        self.metrics = {
            "policy_loss": total_policy_loss / self.n_epochs,
            "value_loss": total_value_loss / self.n_epochs,
            "entropy_loss": total_entropy_loss / self.n_epochs,
            "mean_advantage": advantages.mean().item(),
            "mean_return": returns.mean().item()
        }
        
        return self.metrics
    
    def _store_experience(
        self, 
        state: np.ndarray, 
        action: np.ndarray, 
        reward: float, 
        next_state: np.ndarray, 
        done: bool
    ) -> None:
        """Store experience for PPO update."""
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            _, log_prob, value = self.policy(state_tensor)
        
        self.store_experience(
            state, action, reward, value.item(), log_prob.item(), done
        )
    
    def _should_update(self) -> bool:
        """Check if agent should update."""
        return len(self.states) > 0
    
    def _get_batch(self) -> Dict[str, torch.Tensor]:
        """Get batch for training."""
        return {}
    
    def save(self, filepath: str) -> None:
        """
        Save agent state.
        
        Args:
            filepath: Path to save the model
        """
        torch.save({
            "policy_state_dict": self.policy.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
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
        
        self.policy.load_state_dict(checkpoint["policy_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        
        self.training_step = checkpoint["training_step"]
        self.episode_count = checkpoint["episode_count"]
        self.metrics = checkpoint["metrics"]
