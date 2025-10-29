# Reinforcement Learning Project

A comprehensive implementation of state-of-the-art reinforcement learning algorithms with proper software engineering practices, visualization, and extensibility.

## 🚀 Features

- **Multiple Algorithms**: SAC, PPO, TD3 implementations
- **Modern Libraries**: Uses Gymnasium, PyTorch, and current best practices
- **Type Hints**: Full type annotations for better code quality
- **Comprehensive Logging**: Training metrics and visualization
- **Checkpointing**: Save and load trained models
- **Configuration**: YAML-based configuration system
- **Extensible**: Easy to add new algorithms and environments
- **Testing**: Unit tests for core components

## 📦 Installation

### Prerequisites
- Python 3.10+
- CUDA (optional, for GPU acceleration)

### Setup
```bash
# Clone the repository
git clone https://github.com/kryptologyst/Reinforcement-Learning-Project.git
cd Reinforcement-Learning-Project

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## 🎯 Quick Start

### Basic Training
```bash
# Train SAC on Pendulum environment
python train.py --agent sac --env Pendulum-v1 --episodes 200

# Train PPO on CartPole
python train.py --agent ppo --env CartPole-v1 --episodes 300

# Train TD3 on MountainCar
python train.py --agent td3 --env MountainCar-v0 --episodes 500
```

### Advanced Usage
```bash
# Use custom configuration
python train.py --agent sac --config configs/custom.yaml --episodes 500

# Enable rendering and plotting
python train.py --agent sac --env Pendulum-v1 --render --plot

# Use specific device
python train.py --agent sac --device cuda --episodes 200
```

## 🏗️ Project Structure

```
├── src/
│   ├── agents/           # RL algorithm implementations
│   │   ├── base_agent.py # Abstract base class
│   │   ├── sac.py       # Soft Actor-Critic
│   │   ├── ppo.py       # Proximal Policy Optimization
│   │   └── td3.py       # Twin Delayed DDPG
│   └── utils/           # Utility modules
│       └── replay_buffer.py
├── configs/              # Configuration files
│   └── default.yaml
├── notebooks/            # Jupyter notebooks for analysis
├── tests/               # Unit tests
├── checkpoints/         # Saved models (created during training)
├── train.py            # Main training script
├── requirements.txt    # Dependencies
└── README.md          # This file
```

## 🧠 Algorithms

### Soft Actor-Critic (SAC)
- **Type**: Off-policy, model-free
- **Best for**: Continuous control tasks
- **Key Features**: 
  - Entropy regularization for exploration
  - Twin Q-networks to reduce overestimation
  - Automatic temperature tuning

### Proximal Policy Optimization (PPO)
- **Type**: On-policy, model-free
- **Best for**: Both discrete and continuous control
- **Key Features**:
  - Clipped objective for stable updates
  - Generalized Advantage Estimation (GAE)
  - Multiple epochs per update

### Twin Delayed Deep Deterministic Policy Gradient (TD3)
- **Type**: Off-policy, model-free
- **Best for**: Continuous control with deterministic policies
- **Key Features**:
  - Twin Q-networks
  - Delayed policy updates
  - Target policy smoothing

## 🎮 Supported Environments

The project works with any Gymnasium-compatible environment. Tested environments include:

- **Pendulum-v1**: Continuous control, inverted pendulum
- **CartPole-v1**: Discrete control, balance pole
- **MountainCar-v0**: Discrete control, sparse rewards
- **LunarLander-v2**: Continuous control, landing task
- **HalfCheetah-v4**: Continuous control, locomotion (requires MuJoCo)

## 📊 Visualization

The training script automatically generates:
- Episode reward curves
- Smoothed reward trends
- Reward distribution histograms
- Cumulative reward plots

Example output:
```
Episode   0 | Reward:   -1234.56 | Avg Reward:   -1234.56 | Steps:  200
Episode  10 | Reward:    -987.23 | Avg Reward:   -1100.89 | Steps:  200
Episode  20 | Reward:    -654.32 | Avg Reward:    -821.45 | Steps:  200
```

## ⚙️ Configuration

Create custom configurations in YAML format:

```yaml
# configs/custom.yaml
agent:
  learning_rate: 1e-4
  gamma: 0.99
  batch_size: 128
  hidden_dim: 512

training:
  episodes: 1000
  log_interval: 20
  save_interval: 100
```

## 🔧 Development

### Adding New Algorithms

1. Inherit from `BaseAgent` in `src/agents/`
2. Implement required methods: `select_action`, `update`, `save`, `load`
3. Add to `src/agents/__init__.py`
4. Update configuration schema

### Adding New Environments

Simply use any Gymnasium environment:
```python
import gymnasium as gym
env = gym.make("YourEnvironment-v1")
```

### Running Tests
```bash
pytest tests/
```

## 📈 Performance Tips

1. **GPU Acceleration**: Use `--device cuda` for faster training
2. **Batch Size**: Larger batches generally improve stability
3. **Learning Rate**: Start with default values, adjust based on performance
4. **Environment**: Choose appropriate algorithm for your task type

## 🐛 Troubleshooting

### Common Issues

1. **Import Errors**: Ensure you're in the project root directory
2. **CUDA Issues**: Install PyTorch with CUDA support if using GPU
3. **Environment Not Found**: Install additional environment packages:
   ```bash
   pip install gymnasium[atari] gymnasium[mujoco]
   ```

### Getting Help

- Check the logs for detailed error messages
- Ensure all dependencies are installed correctly
- Verify environment compatibility

## 📚 References

- [SAC Paper](https://arxiv.org/abs/1801.01290)
- [PPO Paper](https://arxiv.org/abs/1707.06347)
- [TD3 Paper](https://arxiv.org/abs/1802.09477)
- [Gymnasium Documentation](https://gymnasium.farama.org/)

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- OpenAI for the original Gym framework
- The RL community for algorithm implementations
- PyTorch team for the excellent deep learning framework
# Reinforcement-Learning-Project
