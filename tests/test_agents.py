"""
Unit tests for the reinforcement learning project.
"""

import pytest
import numpy as np
import torch
import gymnasium as gym
import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

from agents import SACAgent, PPOAgent, TD3Agent
from utils import ReplayBuffer, PrioritizedReplayBuffer


class TestReplayBuffer:
    """Test replay buffer functionality."""
    
    def test_replay_buffer_add_sample(self):
        """Test adding and sampling from replay buffer."""
        buffer = ReplayBuffer(capacity=100)
        
        # Add some experiences
        for i in range(50):
            state = np.random.randn(4)
            action = np.random.randn(2)
            reward = np.random.randn()
            next_state = np.random.randn(4)
            done = i % 10 == 0
            
            buffer.add(state, action, reward, next_state, done)
        
        # Test sampling
        batch = buffer.sample(32)
        
        assert len(batch["states"]) == 32
        assert len(batch["actions"]) == 32
        assert len(batch["rewards"]) == 32
        assert len(batch["next_states"]) == 32
        assert len(batch["dones"]) == 32
        
        assert batch["states"].shape == (32, 4)
        assert batch["actions"].shape == (32, 2)
        assert batch["rewards"].shape == (32, 1)
        assert batch["next_states"].shape == (32, 4)
        assert batch["dones"].shape == (32, 1)
    
    def test_prioritized_replay_buffer(self):
        """Test prioritized replay buffer."""
        buffer = PrioritizedReplayBuffer(capacity=100)
        
        # Add experiences with priorities
        for i in range(50):
            state = np.random.randn(4)
            action = np.random.randn(2)
            reward = np.random.randn()
            next_state = np.random.randn(4)
            done = i % 10 == 0
            priority = np.random.rand()
            
            buffer.add(state, action, reward, next_state, done, priority)
        
        # Test sampling
        batch, weights, indices = buffer.sample(32)
        
        assert len(batch["states"]) == 32
        assert len(weights) == 32
        assert len(indices) == 32
        
        # Test priority update
        new_priorities = np.random.rand(32)
        buffer.update_priorities(indices, new_priorities)


class TestSACAgent:
    """Test SAC agent functionality."""
    
    def test_sac_initialization(self):
        """Test SAC agent initialization."""
        env = gym.make("Pendulum-v1")
        agent = SACAgent(env, device="cpu")
        
        assert agent.state_dim == 3
        assert agent.action_dim == 1
        assert agent.max_action == 2.0
        assert agent.device.type == "cpu"
    
    def test_sac_action_selection(self):
        """Test SAC action selection."""
        env = gym.make("Pendulum-v1")
        agent = SACAgent(env, device="cpu")
        
        state = np.random.randn(3)
        action = agent.select_action(state)
        
        assert action.shape == (1,)
        assert -agent.max_action <= action[0] <= agent.max_action
    
    def test_sac_update(self):
        """Test SAC agent update."""
        env = gym.make("Pendulum-v1")
        agent = SACAgent(env, device="cpu", batch_size=32)
        
        # Add experiences to buffer
        for _ in range(100):
            state = np.random.randn(3)
            action = np.random.randn(1)
            reward = np.random.randn()
            next_state = np.random.randn(3)
            done = np.random.choice([True, False])
            
            agent._store_experience(state, action, reward, next_state, done)
        
        # Test update
        metrics = agent.update(agent._get_batch())
        
        assert "q1_loss" in metrics
        assert "q2_loss" in metrics
        assert "actor_loss" in metrics
        assert isinstance(metrics["q1_loss"], float)


class TestPPOAgent:
    """Test PPO agent functionality."""
    
    def test_ppo_initialization(self):
        """Test PPO agent initialization."""
        env = gym.make("CartPole-v1")
        agent = PPOAgent(env, device="cpu")
        
        assert agent.state_dim == 4
        assert agent.action_dim == 2
        assert agent.continuous == False
        assert agent.device.type == "cpu"
    
    def test_ppo_action_selection(self):
        """Test PPO action selection."""
        env = gym.make("CartPole-v1")
        agent = PPOAgent(env, device="cpu")
        
        state = np.random.randn(4)
        action = agent.select_action(state)
        
        assert action.shape == (1,)
        assert action[0] in [0, 1]  # Discrete actions


class TestTD3Agent:
    """Test TD3 agent functionality."""
    
    def test_td3_initialization(self):
        """Test TD3 agent initialization."""
        env = gym.make("Pendulum-v1")
        agent = TD3Agent(env, device="cpu")
        
        assert agent.state_dim == 3
        assert agent.action_dim == 1
        assert agent.max_action == 2.0
        assert agent.device.type == "cpu"
    
    def test_td3_action_selection(self):
        """Test TD3 action selection."""
        env = gym.make("Pendulum-v1")
        agent = TD3Agent(env, device="cpu")
        
        state = np.random.randn(3)
        action = agent.select_action(state)
        
        assert action.shape == (1,)
        assert -agent.max_action <= action[0] <= agent.max_action


if __name__ == "__main__":
    pytest.main([__file__])
