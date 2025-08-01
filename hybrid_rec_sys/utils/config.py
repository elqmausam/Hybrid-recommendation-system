"""
Configuration Management for Hybrid LEA-Neuroplasticity System

This module handles configuration loading, validation, and management
for different deployment scenarios.
"""

import yaml
import json
from pathlib import Path
from typing import Dict, Any, Optional
from dataclasses import dataclass, asdict
import logging

logger = logging.getLogger(__name__)


@dataclass
class LEAConfig:
    """Configuration for LLM Environment Model"""
    embedding_dim: int = 128
    max_sequence_length: int = 50
    semantic_similarity_threshold: float = 0.7
    transition_learning_rate: float = 0.1
    intent_inference_window: int = 10


@dataclass
class NeuroplasticityConfig:
    """Configuration for Neuroplasticity Reward Shaper"""
    learning_rate: float = 0.1
    adaptation_threshold: float = 0.8
    synaptic_decay_rate: float = 0.99
    hebbian_trace_decay: float = 0.9
    novelty_bonus_weight: float = 0.5
    meta_learning_enabled: bool = True


@dataclass
class DQNConfig:
    """Configuration for DQN Agent"""
    state_dim: int = 512
    action_dim: int = 10000
    learning_rate: float = 1e-4
    gamma: float = 0.99
    epsilon_start: float = 1.0
    epsilon_end: float = 0.01
    epsilon_decay: int = 200000
    batch_size: int = 128
    target_update_freq: int = 2000
    memory_capacity: int = 100000
    prioritized_replay_alpha: float = 0.6
    prioritized_replay_beta: float = 0.4


@dataclass
class RewardConfig:
    """Configuration for Reward System"""
    clicks: float = 1.0
    carts: float = 5.0
    orders: float = 20.0
    negative_sample: float = -0.5
    lea_weight: float = 0.4
    neuroplastic_weight: float = 0.6
    position_bonus_enabled: bool = True
    temporal_decay_enabled: bool = True


@dataclass
class SystemConfig:
    """Main System Configuration"""
    max_aids: int = 100000
    state_dim: int = 512
    recommendation_size: int = 10
    lea_update_frequency: int = 50
    training_mode: bool = True
    
   
    lea_config: LEAConfig = None
    neuroplasticity_config: NeuroplasticityConfig = None
    dqn_config: DQNConfig = None
    reward_config: RewardConfig = None
    
    def __post_init__(self):
        if self.lea_config is None:
            self.lea_config = LEAConfig()
        if self.neuroplasticity_config is None:
            self.neuroplasticity_config = NeuroplasticityConfig()
        if self.dqn_config is None:
            self.dqn_config = DQNConfig(state_dim=self.state_dim)
        if self.reward_config is None:
            self.reward_config = RewardConfig()


class ConfigManager:
    """Configuration Manager for the Hybrid System"""
    
    def __init__(self, config_path: Optional[str] = None):
        self.config_path = config_path
        self.config = self._load_config()
    
    def _load_config(self) -> SystemConfig:
        """Load configuration from file or create default"""
        if self.config_path and Path(self.config_path).exists():
            return self._load_from_file(self.config_path)
        else:
            logger.info("Using default configuration")
            return SystemConfig()
    
    def _load_from_file(self, file_path: str) -> SystemConfig:
        """Load configuration from YAML or JSON file"""
        file_path = Path(file_path)
        
        try:
            with open(file_path, 'r') as f:
                if file_path.suffix.lower() in ['.yaml', '.yml']:
                    data = yaml.safe_load(f)
                elif file_path.suffix.lower() == '.json':
                    data = json.load(f)
                else:
                    raise ValueError(f"Unsupported config file format: {file_path.suffix}")
            
            return self._dict_to_config(data)
            
        except Exception as e:
            logger.error(f"Error loading config from {file_path}: {e}")
            logger.info("Falling back to default configuration")
            return SystemConfig()
    
    def _dict_to_config(self, data: Dict[str, Any]) -> SystemConfig:
        """Convert dictionary to SystemConfig object"""
        
        lea_data = data.pop('lea_config', {})
        neuroplasticity_data = data.pop('neuroplasticity_config', {})
        dqn_data = data.pop('dqn_config', {})
        reward_data = data.pop('reward_config', {})
        
        
        lea_config = LEAConfig(**lea_data)
        neuroplasticity_config = NeuroplasticityConfig(**neuroplasticity_data)
        dqn_config = DQNConfig(**dqn_data)
        reward_config = RewardConfig(**reward_data)
        
        
        system_config = SystemConfig(
            lea_config=lea_config,
            neuroplasticity_config=neuroplasticity_config,
            dqn_config=dqn_config,
            reward_config=reward_config,
            **data
        )
        
        return system_config
    
    def save_config(self, file_path: str):
        """Save current configuration to file"""
        file_path = Path(file_path)
        file_path.parent.mkdir(parents=True, exist_ok=True)
        
       
        config_dict = asdict(self.config)
        
        try:
            with open(file_path, 'w') as f:
                if file_path.suffix.lower() in ['.yaml', '.yml']:
                    yaml.dump(config_dict, f, default_flow_style=False, indent=2)
                elif file_path.suffix.lower() == '.json':
                    json.dump(config_dict, f, indent=2)
                else:
                    raise ValueError(f"Unsupported config file format: {file_path.suffix}")
            
            logger.info(f"Configuration saved to {file_path}")
            
        except Exception as e:
            logger.error(f"Error saving config to {file_path}: {e}")
    
    def update_config(self, updates: Dict[str, Any]):
        """Update configuration with new values"""
        for key, value in updates.items():
            if hasattr(self.config, key):
                setattr(self.config, key, value)
            else:
                logger.warning(f"Unknown configuration key: {key}")
    
    def get_config(self) -> SystemConfig:
        """Get current configuration"""
        return self.config
    
    def validate_config(self) -> bool:
        """Validate configuration values"""
        try:
          
            assert 0 < self.config.lea_config.embedding_dim <= 1024
            assert 0 < self.config.lea_config.max_sequence_length <= 1000
            assert 0 <= self.config.lea_config.semantic_similarity_threshold <= 1.0
            
            assert 0 < self.config.neuroplasticity_config.learning_rate <= 1.0
            assert 0 <= self.config.neuroplasticity_config.adaptation_threshold <= 1.0
            assert 0 < self.config.neuroplasticity_config.synaptic_decay_rate <= 1.0
            
            assert 0 < self.config.dqn_config.state_dim <= 2048
            assert 0 < self.config.dqn_config.learning_rate <= 1.0
            assert 0 <= self.config.dqn_config.gamma <= 1.0
            assert 0 <= self.config.dqn_config.epsilon_end <= self.config.dqn_config.epsilon_start <= 1.0
            
            assert self.config.reward_config.lea_weight + self.config.reward_config.neuroplastic_weight > 0
            
            logger.info("Configuration validation passed")
            return True
            
        except AssertionError as e:
            logger.error(f"Configuration validation failed: {e}")
            return False
    
    @classmethod
    def create_default_configs(cls, output_dir: str = "configs"):
        """Create default configuration files for different scenarios"""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        
        default_config = SystemConfig()
        default_manager = cls()
        default_manager.config = default_config
        default_manager.save_config(output_path / "default_config.yaml")
        
        
        training_config = SystemConfig(
            training_mode=True,
            neuroplasticity_config=NeuroplasticityConfig(
                learning_rate=0.2,
                novelty_bonus_weight=0.7
            ),
            dqn_config=DQNConfig(
                learning_rate=2e-4,
                epsilon_decay=100000 
            )
        )
        training_manager = cls()
        training_manager.config = training_config
        training_manager.save_config(output_path / "training_config.yaml")
        
        
        production_config = SystemConfig(
            training_mode=False,
            neuroplasticity_config=NeuroplasticityConfig(
                learning_rate=0.05,
                novelty_bonus_weight=0.3
            ),
            dqn_config=DQNConfig(
                learning_rate=5e-5,
                epsilon_start=0.1,
                epsilon_end=0.01
            )
        )
        production_manager = cls()
        production_manager.config = production_config
        production_manager.save_config(output_path / "production_config.yaml")
        
        logger.info(f"Default configurations created in {output_dir}")


def load_config(config_path: Optional[str] = None) -> SystemConfig:
    """Convenience function to load configuration"""
    manager = ConfigManager(config_path)
    return manager.get_config()


def save_config(config: SystemConfig, file_path: str):
    """Convenience function to save configuration"""
    manager = ConfigManager()
    manager.config = config
    manager.save_config(file_path)
