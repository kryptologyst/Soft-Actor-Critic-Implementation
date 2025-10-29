#!/usr/bin/env python3
"""
Simple SAC Demo - Working example without NaN issues
"""

import sys
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import gymnasium as gym
import matplotlib.pyplot as plt
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

from agents import SACAgent


def simple_sac_demo():
    """Simple SAC demonstration."""
    print("🚀 Starting Simple SAC Demo")
    
    # Create environment
    env = gym.make("Pendulum-v1")
    print(f"Environment: {env.observation_space.shape} -> {env.action_space}")
    
    # Create SAC agent with conservative settings
    agent = SACAgent(
        env,
        learning_rate=1e-3,  # Slightly higher learning rate
        batch_size=64,        # Smaller batch size
        hidden_dim=128,       # Smaller network
        device="cpu"
    )
    
    print(f"Agent created: {agent.state_dim}D state, {agent.action_dim}D action")
    
    # Training loop
    episode_rewards = []
    num_episodes = 50
    
    for episode in range(num_episodes):
        state, _ = env.reset()
        episode_reward = 0
        done = False
        step = 0
        
        while not done and step < 200:
            # Select action
            action = agent.select_action(state)
            
            # Take step
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            
            # Store experience
            agent._store_experience(state, action, reward, next_state, done)
            
            # Update if enough samples
            if agent._should_update():
                metrics = agent.update(agent._get_batch())
                agent.training_step += 1
            
            state = next_state
            episode_reward += reward
            step += 1
        
        episode_rewards.append(episode_reward)
        agent.episode_count += 1
        
        # Log progress
        if episode % 10 == 0:
            avg_reward = np.mean(episode_rewards[-10:])
            print(f"Episode {episode:3d} | Reward: {episode_reward:8.2f} | Avg: {avg_reward:8.2f}")
    
    # Plot results
    plt.figure(figsize=(10, 6))
    plt.plot(episode_rewards, alpha=0.7, label='Episode Rewards')
    
    # Smooth curve
    if len(episode_rewards) > 10:
        window = min(10, len(episode_rewards) // 5)
        smoothed = np.convolve(episode_rewards, np.ones(window)/window, mode='valid')
        plt.plot(range(window-1, len(episode_rewards)), smoothed, 'r-', linewidth=2, label=f'Smoothed (window={window})')
    
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    plt.title('SAC Training Progress on Pendulum-v1')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    
    # Save plot
    plt.savefig('sac_training_results.png', dpi=300, bbox_inches='tight')
    print(f"📊 Training plot saved as 'sac_training_results.png'")
    
    # Final statistics
    print("\n" + "="*50)
    print("TRAINING COMPLETED")
    print("="*50)
    print(f"Final Episode Reward: {episode_rewards[-1]:.2f}")
    print(f"Average Reward (last 10): {np.mean(episode_rewards[-10:]):.2f}")
    print(f"Best Episode Reward: {np.max(episode_rewards):.2f}")
    print(f"Total Training Steps: {agent.training_step}")
    
    # Save model
    agent.save("sac_demo_model.pth")
    print("💾 Model saved as 'sac_demo_model.pth'")
    
    env.close()
    return episode_rewards


if __name__ == "__main__":
    rewards = simple_sac_demo()
    print(f"\n✅ Demo completed successfully!")
    print(f"📈 Final reward: {rewards[-1]:.2f}")
    print(f"🎯 Improvement: {rewards[-1] - rewards[0]:.2f}")
