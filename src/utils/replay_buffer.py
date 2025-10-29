"""
Replay buffer implementations for off-policy reinforcement learning.
"""

from typing import Dict, List, Optional, Tuple, Union
import numpy as np
import torch
from collections import deque
import random


class ReplayBuffer:
    """
    Standard replay buffer for storing and sampling experience tuples.
    
    Args:
        capacity: Maximum number of experiences to store
        device: Device to store tensors on
    """
    
    def __init__(self, capacity: int = 1_000_000, device: str = "cpu") -> None:
        self.capacity = capacity
        self.device = torch.device(device)
        self.buffer = deque(maxlen=capacity)
        
    def add(
        self, 
        state: np.ndarray, 
        action: np.ndarray, 
        reward: float, 
        next_state: np.ndarray, 
        done: bool
    ) -> None:
        """
        Add an experience to the buffer.
        
        Args:
            state: Current state
            action: Action taken
            reward: Reward received
            next_state: Next state
            done: Whether episode is done
        """
        self.buffer.append((state, action, reward, next_state, done))
    
    def sample(self, batch_size: int) -> Dict[str, torch.Tensor]:
        """
        Sample a batch of experiences.
        
        Args:
            batch_size: Number of experiences to sample
            
        Returns:
            Dictionary containing batched tensors
        """
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        
        return {
            "states": torch.FloatTensor(np.array(states)).to(self.device),
            "actions": torch.FloatTensor(np.array(actions)).to(self.device),
            "rewards": torch.FloatTensor(np.array(rewards)).unsqueeze(1).to(self.device),
            "next_states": torch.FloatTensor(np.array(next_states)).to(self.device),
            "dones": torch.FloatTensor(np.array(dones)).unsqueeze(1).to(self.device)
        }
    
    def __len__(self) -> int:
        """Return the current size of the buffer."""
        return len(self.buffer)
    
    def is_ready(self, batch_size: int) -> bool:
        """Check if buffer has enough samples for training."""
        return len(self.buffer) >= batch_size


class PrioritizedReplayBuffer:
    """
    Prioritized Experience Replay buffer.
    
    Args:
        capacity: Maximum number of experiences to store
        alpha: Prioritization exponent (0 = uniform, 1 = fully prioritized)
        beta: Importance sampling exponent
        device: Device to store tensors on
    """
    
    def __init__(
        self, 
        capacity: int = 1_000_000, 
        alpha: float = 0.6, 
        beta: float = 0.4,
        device: str = "cpu"
    ) -> None:
        self.capacity = capacity
        self.alpha = alpha
        self.beta = beta
        self.device = torch.device(device)
        
        self.buffer = deque(maxlen=capacity)
        self.priorities = deque(maxlen=capacity)
        self.max_priority = 1.0
        
    def add(
        self, 
        state: np.ndarray, 
        action: np.ndarray, 
        reward: float, 
        next_state: np.ndarray, 
        done: bool,
        priority: Optional[float] = None
    ) -> None:
        """
        Add an experience to the buffer.
        
        Args:
            state: Current state
            action: Action taken
            reward: Reward received
            next_state: Next state
            done: Whether episode is done
            priority: Priority of the experience (if None, uses max priority)
        """
        if priority is None:
            priority = self.max_priority
            
        self.buffer.append((state, action, reward, next_state, done))
        self.priorities.append(priority)
        self.max_priority = max(self.max_priority, priority)
    
    def sample(self, batch_size: int) -> Tuple[Dict[str, torch.Tensor], torch.Tensor, np.ndarray]:
        """
        Sample a batch of experiences with importance sampling weights.
        
        Args:
            batch_size: Number of experiences to sample
            
        Returns:
            Tuple of (batched tensors, importance weights, indices)
        """
        priorities = np.array(self.priorities)
        probabilities = priorities ** self.alpha
        probabilities /= probabilities.sum()
        
        indices = np.random.choice(len(self.buffer), batch_size, p=probabilities)
        
        batch = [self.buffer[i] for i in indices]
        states, actions, rewards, next_states, dones = zip(*batch)
        
        # Calculate importance sampling weights
        weights = (len(self.buffer) * probabilities[indices]) ** (-self.beta)
        weights /= weights.max()
        
        return {
            "states": torch.FloatTensor(np.array(states)).to(self.device),
            "actions": torch.FloatTensor(np.array(actions)).to(self.device),
            "rewards": torch.FloatTensor(np.array(rewards)).unsqueeze(1).to(self.device),
            "next_states": torch.FloatTensor(np.array(next_states)).to(self.device),
            "dones": torch.FloatTensor(np.array(dones)).unsqueeze(1).to(self.device)
        }, torch.FloatTensor(weights).to(self.device), indices
    
    def update_priorities(self, indices: np.ndarray, priorities: np.ndarray) -> None:
        """
        Update priorities for given indices.
        
        Args:
            indices: Indices to update
            priorities: New priorities
        """
        for idx, priority in zip(indices, priorities):
            self.priorities[idx] = priority
            self.max_priority = max(self.max_priority, priority)
    
    def __len__(self) -> int:
        """Return the current size of the buffer."""
        return len(self.buffer)
    
    def is_ready(self, batch_size: int) -> bool:
        """Check if buffer has enough samples for training."""
        return len(self.buffer) >= batch_size
