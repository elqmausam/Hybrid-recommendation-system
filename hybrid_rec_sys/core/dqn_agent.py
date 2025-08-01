"""
Enhanced DQN Agent

This module implements the DQN agent with Dueling architecture
optimized for recommendation systems with LEA and neuroplasticity integration.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import random
import math
from typing import List, Dict
from .replay_buffer import EnhancedPrioritizedReplayBuffer, Experience
import logging

logger = logging.getLogger(__name__)


class EnhancedDuelingDQN(nn.Module):
    """Enhanced Dueling DQN with better architecture for LEA integration"""
    
    def __init__(self, state_dim: int = 512, action_dim: int = 10000):
        super().__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        
        
        self.feature_layers = nn.Sequential(
            nn.Linear(state_dim, 1024),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.LayerNorm(1024),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.LayerNorm(512),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.LayerNorm(256)
        )
        
        
        self.value_stream = nn.Sequential(
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 1)
        )
        
       
        self.advantage_stream = nn.Sequential(
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, action_dim)
        )
        
    def forward(self, state: torch.Tensor) -> torch.Tensor:
        features = self.feature_layers(state)
        value = self.value_stream(features)
        advantage = self.advantage_stream(features)
        
        
        q_values = value + advantage - advantage.mean(dim=1, keepdim=True)
        return q_values


class EnhancedDQNAgent:
    """Enhanced DQN agent optimized for LEA + neuroplasticity integration"""
    
    def __init__(self, state_dim: int = 512, action_dim: int = 10000, 
                 lr: float = 1e-4, gamma: float = 0.99):
        
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.epsilon = 1.0
        self.epsilon_end = 0.01
        self.epsilon_decay = 200000 
        self.steps_done = 0
        
      
        self.q_network = EnhancedDuelingDQN(state_dim, action_dim)
        self.target_network = EnhancedDuelingDQN(state_dim, action_dim)
        self.target_network.load_state_dict(self.q_network.state_dict())
        
        self.optimizer = optim.AdamW(self.q_network.parameters(), lr=lr, weight_decay=1e-5)
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, patience=10000)
        
       
        self.memory = EnhancedPrioritizedReplayBuffer(capacity=100000)
        
        self.batch_size = 128
        self.target_update_freq = 2000
        self.training_step = 0
        
     
        self.action_mapper = {}
        self.reverse_action_mapper = {}
        self.current_action_id = 0
        
        
        self.loss_history = []
        
    def map_action(self, aid: int) -> int:
        """Map action ID to internal action space"""
        if aid not in self.action_mapper:
            self.action_mapper[aid] = self.current_action_id
            self.reverse_action_mapper[self.current_action_id] = aid
            self.current_action_id += 1
        return self.action_mapper[aid]
    
    def get_aid_from_action(self, action_id: int) -> int:
        """Get original action ID from internal mapping"""
        return self.reverse_action_mapper.get(action_id, 0)
    
    def select_action(self, state: torch.Tensor, available_actions: List[int]) -> int:
        """Enhanced epsilon-greedy action selection with LEA-aware exploration"""
        self.epsilon = self.epsilon_end + (1.0 - self.epsilon_end) * \
                      math.exp(-1. * self.steps_done / self.epsilon_decay)
        self.steps_done += 1
        
        if random.random() > self.epsilon:
            with torch.no_grad():
                q_values = self.q_network(state)
                available_action_ids = [self.map_action(aid) for aid in available_actions]
                
            
                masked_q_values = torch.full((self.action_dim,), float('-inf'))
                
                for aid in available_action_ids:
                    if aid < self.action_dim:
                        masked_q_values[aid] = q_values[0, aid]
                
                action_id = masked_q_values.argmax().item()
                return self.get_aid_from_action(action_id)
        else:
            return random.choice(available_actions)
    
    def store_experience(self, state: torch.Tensor, action: int, reward: float, 
                        next_state: torch.Tensor, done: bool):
        """Store experience in replay buffer"""
        action_id = self.map_action(action)
        experience = Experience(state, action_id, reward, next_state, done)
        self.memory.add(experience)
    
    def train(self):
        """Train the DQN with prioritized experience replay"""
        if len(self.memory.buffer) < self.batch_size:
            return
        
        experiences, weights, indices = self.memory.sample(self.batch_size)
        if not experiences:
            return
        
     
        self.q_network.train()
        self.target_network.eval()
        
       
        states = torch.stack([exp.state.squeeze() for exp in experiences])
        actions = torch.tensor([exp.action for exp in experiences], dtype=torch.long)
        rewards = torch.tensor([exp.reward for exp in experiences], dtype=torch.float32)
        next_states = torch.stack([exp.next_state.squeeze() for exp in experiences])
        dones = torch.tensor([exp.done for exp in experiences], dtype=torch.bool)
        
       
        current_q_values = self.q_network(states).gather(1, actions.unsqueeze(1))
        
    
        with torch.no_grad():
            next_actions = self.q_network(next_states).argmax(1, keepdim=True)
            next_q_values = self.target_network(next_states).gather(1, next_actions)
            target_q_values = rewards.unsqueeze(1) + (self.gamma * next_q_values * ~dones.unsqueeze(1))
        
      
        td_errors = torch.abs(current_q_values - target_q_values).squeeze()
        loss = (weights.unsqueeze(1) * F.huber_loss(current_q_values, target_q_values, reduction='none')).mean()
        
        
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.q_network.parameters(), 1.0)
        self.optimizer.step()
        
        
        self.memory.update_priorities(indices, td_errors + 1e-6)
        
       
        self.scheduler.step(loss)
        
       
        self.loss_history.append(loss.item())
        if len(self.loss_history) > 1000:
            self.loss_history = self.loss_history[-1000:]
        
        self.training_step += 1
        if self.training_step % self.target_update_freq == 0:
            self.target_network.load_state_dict(self.q_network.state_dict())
    
    def save_model(self, path: str):
        """Save model state"""
        torch.save({
            'q_network_state_dict': self.q_network.state_dict(),
            'target_network_state_dict': self.target_network.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'action_mapper': self.action_mapper,
            'reverse_action_mapper': self.reverse_action_mapper,
            'current_action_id': self.current_action_id,
            'training_step': self.training_step,
            'epsilon': self.epsilon
        }, path)
    
    def load_model(self, path: str):
        """Load model state"""
        checkpoint = torch.load(path)
        self.q_network.load_state_dict(checkpoint['q_network_state_dict'])
        self.target_network.load_state_dict(checkpoint['target_network_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.action_mapper = checkpoint['action_mapper']
        self.reverse_action_mapper = checkpoint['reverse_action_mapper']
        self.current_action_id = checkpoint['current_action_id']
        self.training_step = checkpoint['training_step']
        self.epsilon = checkpoint['epsilon']
    
    def get_metrics(self) -> Dict:
        """Get training metrics"""
        return {
            'epsilon': self.epsilon,
            'training_steps': self.training_step,
            'action_space_size': self.current_action_id,
            'memory_size': len(self.memory),
            'recent_loss': np.mean(self.loss_history[-100:]) if self.loss_history else 0,
            'lr': self.optimizer.param_groups[0]['lr']
        }
