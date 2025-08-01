"""
Hybrid LEA-Neuroplasticity Recommendation System

A comprehensive recommendation system combining:
- LLM as Environment (LEA) modeling
- Neuroplasticity-inspired reward shaping
- Deep Q-Network (DQN) reinforcement learning
- Hybrid state encoding with transformer architecture
"""

__version__ = "1.0.0"
__author__ = "Your Name"
__email__ = "your.email@example.com"

from .main import HybridLEANeuroplasticRecommendationSystem
from .core.lea_environment import LLMEnvironmentModel
from .core.neuroplasticity import EnhancedNeuroplasticityRewardShaper
from .core.state_encoder import HybridLEANeuroplasticStateEncoder
from .core.dqn_agent import EnhancedDQNAgent
from .core.replay_buffer import EnhancedPrioritizedReplayBuffer

__all__ = [
    'HybridLEANeuroplasticRecommendationSystem',
    'LLMEnvironmentModel',
    'EnhancedNeuroplasticityRewardShaper',
    'HybridLEANeuroplasticStateEncoder',
    'EnhancedDQNAgent',
    'EnhancedPrioritizedReplayBuffer'
]
