#!/usr/bin/env python3
"""
Modern SAC Training Script

This script demonstrates training a Soft Actor-Critic agent on various environments
with proper logging, visualization, and checkpointing.
"""

import argparse
import os
import sys
from typing import Dict, List, Optional
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import gymnasium as gym
import torch
import yaml
from pathlib import Path

# Add src to path for imports
sys.path.append(str(Path(__file__).parent / "src"))

from agents import SACAgent, PPOAgent, TD3Agent
from utils import ReplayBuffer


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Train RL agents")
    
    # Environment
    parser.add_argument("--env", type=str, default="Pendulum-v1", 
                       help="Environment name")
    parser.add_argument("--episodes", type=int, default=200,
                       help="Number of training episodes")
    
    # Agent
    parser.add_argument("--agent", type=str, default="sac", 
                       choices=["sac", "ppo", "td3"],
                       help="Agent algorithm to use")
    parser.add_argument("--config", type=str, default=None,
                       help="Path to config file")
    
    # Training
    parser.add_argument("--seed", type=int, default=42,
                       help="Random seed")
    parser.add_argument("--device", type=str, default="auto",
                       help="Device to use (cpu, cuda, auto)")
    
    # Logging and saving
    parser.add_argument("--save-dir", type=str, default="./checkpoints",
                       help="Directory to save checkpoints")
    parser.add_argument("--log-interval", type=int, default=10,
                       help="Logging interval")
    parser.add_argument("--save-interval", type=int, default=50,
                       help="Checkpoint saving interval")
    
    # Visualization
    parser.add_argument("--plot", action="store_true",
                       help="Plot training curves")
    parser.add_argument("--render", action="store_true",
                       help="Render environment during training")
    
    return parser.parse_args()


def load_config(config_path: Optional[str]) -> Dict:
    """Load configuration from YAML file."""
    if config_path and os.path.exists(config_path):
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)
    return {}


def create_agent(agent_type: str, env: gym.Env, config: Dict, args: argparse.Namespace):
    """Create agent based on type and configuration."""
    agent_config = {
        "env": env,
        "device": args.device,
        "seed": args.seed,
        **config.get("agent", {})
    }
    
    if agent_type == "sac":
        return SACAgent(**agent_config)
    elif agent_type == "ppo":
        return PPOAgent(**agent_config)
    elif agent_type == "td3":
        return TD3Agent(**agent_config)
    else:
        raise ValueError(f"Unknown agent type: {agent_type}")


def train_agent(agent, env: gym.Env, args: argparse.Namespace) -> List[float]:
    """Train the agent and return episode rewards."""
    episode_rewards = []
    best_reward = float('-inf')
    
    # Create save directory
    os.makedirs(args.save_dir, exist_ok=True)
    
    print(f"Training {args.agent.upper()} agent on {args.env}")
    print(f"Device: {agent.device}")
    print(f"Episodes: {args.episodes}")
    print("-" * 50)
    
    for episode in range(args.episodes):
        state, _ = env.reset()
        episode_reward = 0
        done = False
        step = 0
        
        while not done and step < 1000:  # Prevent infinite episodes
            if args.render:
                env.render()
            
            # Select action
            action = agent.select_action(state)
            
            # Take step
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            
            # Store experience and update
            agent._store_experience(state, action, reward, next_state, done)
            
            if agent._should_update():
                metrics = agent.update(agent._get_batch())
                agent.training_step += 1
            
            state = next_state
            episode_reward += reward
            step += 1
        
        episode_rewards.append(episode_reward)
        agent.episode_count += 1
        
        # Logging
        if episode % args.log_interval == 0:
            avg_reward = np.mean(episode_rewards[-args.log_interval:])
            print(f"Episode {episode:4d} | "
                  f"Reward: {episode_reward:8.2f} | "
                  f"Avg Reward: {avg_reward:8.2f} | "
                  f"Steps: {step:4d}")
            
            if hasattr(agent, 'metrics') and agent.metrics:
                for key, value in agent.metrics.items():
                    print(f"  {key}: {value:.4f}")
        
        # Save checkpoint
        if episode % args.save_interval == 0 and episode > 0:
            checkpoint_path = os.path.join(args.save_dir, f"{args.agent}_{episode}.pth")
            agent.save(checkpoint_path)
            
            if episode_reward > best_reward:
                best_reward = episode_reward
                best_path = os.path.join(args.save_dir, f"{args.agent}_best.pth")
                agent.save(best_path)
                print(f"New best model saved with reward: {best_reward:.2f}")
    
    # Save final model
    final_path = os.path.join(args.save_dir, f"{args.agent}_final.pth")
    agent.save(final_path)
    
    return episode_rewards


def plot_training_curves(rewards: List[float], agent_type: str, env_name: str) -> None:
    """Plot training curves."""
    plt.figure(figsize=(12, 8))
    
    # Plot raw rewards
    plt.subplot(2, 2, 1)
    plt.plot(rewards, alpha=0.3, color='blue')
    plt.title(f"{agent_type.upper()} - Raw Episode Rewards")
    plt.xlabel("Episode")
    plt.ylabel("Reward")
    plt.grid(True)
    
    # Plot smoothed rewards
    plt.subplot(2, 2, 2)
    window_size = min(50, len(rewards) // 10)
    if window_size > 1:
        smoothed = np.convolve(rewards, np.ones(window_size)/window_size, mode='valid')
        plt.plot(smoothed, color='red', linewidth=2)
    plt.title(f"{agent_type.upper()} - Smoothed Rewards (window={window_size})")
    plt.xlabel("Episode")
    plt.ylabel("Smoothed Reward")
    plt.grid(True)
    
    # Plot reward distribution
    plt.subplot(2, 2, 3)
    plt.hist(rewards, bins=30, alpha=0.7, color='green')
    plt.title(f"{agent_type.upper()} - Reward Distribution")
    plt.xlabel("Reward")
    plt.ylabel("Frequency")
    plt.grid(True)
    
    # Plot cumulative rewards
    plt.subplot(2, 2, 4)
    cumulative_rewards = np.cumsum(rewards)
    plt.plot(cumulative_rewards, color='purple')
    plt.title(f"{agent_type.upper()} - Cumulative Rewards")
    plt.xlabel("Episode")
    plt.ylabel("Cumulative Reward")
    plt.grid(True)
    
    plt.tight_layout()
    plt.suptitle(f"{agent_type.upper()} Training on {env_name}", y=1.02)
    
    # Save plot
    plot_path = f"{agent_type}_training_curves.png"
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"Training curves saved to {plot_path}")
    
    plt.show()


def main():
    """Main training function."""
    args = parse_args()
    
    # Set random seeds
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)
    
    # Load configuration
    config = load_config(args.config)
    
    # Create environment
    env = gym.make(args.env)
    
    # Create agent
    agent = create_agent(args.agent, env, config, args)
    
    # Train agent
    episode_rewards = train_agent(agent, env, args)
    
    # Plot results
    if args.plot:
        plot_training_curves(episode_rewards, args.agent, args.env)
    
    # Print final statistics
    print("\n" + "="*50)
    print("TRAINING COMPLETED")
    print("="*50)
    print(f"Final Episode Reward: {episode_rewards[-1]:.2f}")
    print(f"Average Reward (last 10): {np.mean(episode_rewards[-10:]):.2f}")
    print(f"Average Reward (last 50): {np.mean(episode_rewards[-50:]):.2f}")
    print(f"Best Episode Reward: {np.max(episode_rewards):.2f}")
    print(f"Total Training Steps: {agent.training_step}")
    
    env.close()


if __name__ == "__main__":
    main()
