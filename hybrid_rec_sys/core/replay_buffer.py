"""
Enhanced Prioritized Replay Buffer

This module implements a robust prioritized experience replay buffer
with NaN handling and importance sampling for DQN training.
"""

import numpy as np
import torch
from collections import namedtuple
from typing import List, Tuple
import logging

logger = logging.getLogger(__name__)


Experience = namedtuple('Experience', ['state', 'action', 'reward', 'next_state', 'done'])


class EnhancedPrioritizedReplayBuffer:
    """Fixed prioritized replay buffer with NaN handling"""
    
    def __init__(self, capacity: int = 100000, alpha: float = 0.6, beta: float = 0.4):
        self.capacity = capacity
        self.alpha = alpha
        self.beta = beta
        self.beta_increment = 0.0001
        
        self.buffer = []
        self.priorities = np.ones(capacity, dtype=np.float32) * 1e-6  
        self.position = 0
        self.max_priority = 1.0
        
    def add(self, experience: Experience):
        """Add experience to buffer with maximum priority"""
        if len(self.buffer) < self.capacity:
            self.buffer.append(experience)
        else:
            self.buffer[self.position] = experience
        
        self.priorities[self.position] = max(self.max_priority, 1e-6)  
        self.position = (self.position + 1) % self.capacity
    
    def sample(self, batch_size: int) -> Tuple[List[Experience], torch.Tensor, torch.Tensor]:
        """Sample batch of experiences with importance sampling weights"""
        if len(self.buffer) < batch_size:
            return [], torch.tensor([]), torch.tensor([])
        
        priorities = self.priorities[:len(self.buffer)]
        
        
        priorities = np.nan_to_num(priorities, nan=1e-6, posinf=1e-6, neginf=1e-6)
        priorities = np.maximum(priorities, 1e-6)  
        
        probs = priorities ** self.alpha
        probs_sum = probs.sum()
        
        if probs_sum <= 0 or np.isnan(probs_sum) or np.isinf(probs_sum):
            
            probs = np.ones_like(probs) / len(probs)
        else:
            probs /= probs_sum
        
       
        probs = np.nan_to_num(probs, nan=1.0/len(probs))
        probs = np.maximum(probs, 1e-8)  
        probs /= probs.sum()  
        
        try:
            indices = np.random.choice(len(self.buffer), batch_size, p=probs)
        except ValueError:
           
            indices = np.random.choice(len(self.buffer), batch_size)
        
        weights = (len(self.buffer) * probs[indices]) ** (-self.beta)
        weights = np.nan_to_num(weights, nan=1.0, posinf=1.0, neginf=1.0)
        max_weight = weights.max()
        if max_weight > 0:
            weights /= max_weight
        else:
            weights = np.ones_like(weights)
        
        experiences = [self.buffer[i] for i in indices]
        self.beta = min(1.0, self.beta + self.beta_increment)
        
        return experiences, torch.FloatTensor(weights), torch.LongTensor(indices)
    
    def update_priorities(self, indices: torch.Tensor, priorities: torch.Tensor):
        """Update priorities for given indices"""
        for idx, priority in zip(indices, priorities):
            if 0 <= idx.item() < len(self.priorities):
                
                priority_val = max(1e-6, float(priority.item()))
                if np.isfinite(priority_val):
                    self.priorities[idx.item()] = priority_val
                    self.max_priority = max(self.max_priority, priority_val)
    
    def __len__(self):
        """Return current buffer size"""
        return len(self.buffer)
    
    def get_statistics(self) -> dict:
        """Get buffer statistics for monitoring"""
        if not self.buffer:
            return {'size': 0, 'avg_priority': 0, 'max_priority': 0}
        
        valid_priorities = self.priorities[:len(self.buffer)]
        valid_priorities = valid_priorities[np.isfinite(valid_priorities)]
        
        return {
            'size': len(self.buffer),
            'capacity': self.capacity,
            'avg_priority': np.mean(valid_priorities) if len(valid_priorities) > 0 else 0,
            'max_priority': self.max_priority,
            'beta': self.beta
        }
