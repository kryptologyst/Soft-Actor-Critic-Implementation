"""
Base agent class for reinforcement learning algorithms.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
import torch
import torch.nn as nn
from gymnasium import Env


class BaseAgent(ABC):
    """Abstract base class for all RL agents."""
    
    def __init__(
        self,
        env: Env,
        device: str = "auto",
        seed: Optional[int] = None,
        **kwargs
    ) -> None:
        """
        Initialize the base agent.
        
        Args:
            env: The environment to interact with
            device: Device to run computations on ("cpu", "cuda", or "auto")
            seed: Random seed for reproducibility
            **kwargs: Additional agent-specific parameters
        """
        self.env = env
        self.device = self._get_device(device)
        self.seed = seed
        
        if seed is not None:
            self._set_seed(seed)
            
        self.training_step = 0
        self.episode_count = 0
        
    def _get_device(self, device: str) -> torch.device:
        """Get the appropriate device for computation."""
        if device == "auto":
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return torch.device(device)
    
    def _set_seed(self, seed: int) -> None:
        """Set random seeds for reproducibility."""
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
    
    @abstractmethod
    def select_action(
        self, 
        state: np.ndarray, 
        deterministic: bool = False
    ) -> np.ndarray:
        """
        Select an action given a state.
        
        Args:
            state: Current state
            deterministic: Whether to select action deterministically
            
        Returns:
            Selected action
        """
        pass
    
    @abstractmethod
    def update(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """
        Update the agent's parameters.
        
        Args:
            batch: Batch of experience data
            
        Returns:
            Dictionary of training metrics
        """
        pass
    
    @abstractmethod
    def save(self, filepath: str) -> None:
        """Save the agent's state."""
        pass
    
    @abstractmethod
    def load(self, filepath: str) -> None:
        """Load the agent's state."""
        pass
    
    def train(self, num_episodes: int) -> List[float]:
        """
        Train the agent for a specified number of episodes.
        
        Args:
            num_episodes: Number of episodes to train
            
        Returns:
            List of episode rewards
        """
        episode_rewards = []
        
        for episode in range(num_episodes):
            state, _ = self.env.reset()
            episode_reward = 0
            done = False
            
            while not done:
                action = self.select_action(state)
                next_state, reward, terminated, truncated, _ = self.env.step(action)
                done = terminated or truncated
                
                # Store experience (implemented by subclasses)
                self._store_experience(state, action, reward, next_state, done)
                
                # Update agent (implemented by subclasses)
                if self._should_update():
                    metrics = self.update(self._get_batch())
                    self.training_step += 1
                
                state = next_state
                episode_reward += reward
            
            episode_rewards.append(episode_reward)
            self.episode_count += 1
            
            if episode % 10 == 0:
                avg_reward = np.mean(episode_rewards[-10:])
                print(f"Episode {episode}, Average Reward: {avg_reward:.2f}")
        
        return episode_rewards
    
    def _store_experience(
        self, 
        state: np.ndarray, 
        action: np.ndarray, 
        reward: float, 
        next_state: np.ndarray, 
        done: bool
    ) -> None:
        """Store experience in replay buffer (implemented by subclasses)."""
        pass
    
    def _should_update(self) -> bool:
        """Check if agent should update (implemented by subclasses)."""
        return False
    
    def _get_batch(self) -> Dict[str, torch.Tensor]:
        """Get batch for training (implemented by subclasses)."""
        return {}
