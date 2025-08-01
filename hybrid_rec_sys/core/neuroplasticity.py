"""
Enhanced Neuroplasticity Reward Shaper

This module implements neuroplasticity-inspired reward shaping mechanisms
that adapt based on learning patterns and LEA integration.
"""

import numpy as np
from collections import defaultdict
from typing import Dict, List
import logging

logger = logging.getLogger(__name__)


class EnhancedNeuroplasticityRewardShaper:
    """Enhanced neuroplasticity with LEA integration"""
    
    def __init__(self, learning_rate: float = 0.1, adaptation_threshold: float = 0.8):
        self.learning_rate = learning_rate
        self.adaptation_threshold = adaptation_threshold
        
        # Original neuroplasticity components
        self.synaptic_weights = defaultdict(lambda: defaultdict(float))
        self.connection_strengths = defaultdict(float)
        self.hebbian_traces = defaultdict(lambda: defaultdict(float))
        self.homeostatic_scaling = defaultdict(float)
        self.synaptic_tagging = defaultdict(lambda: defaultdict(float))
        self.meta_rewards = defaultdict(list)
        self.adaptation_history = []
        
        # LEA integration
        self.lle_reward_history = defaultdict(list)
        self.lle_state_quality = defaultdict(float)
        
    def calculate_enhanced_neuroplastic_reward(self, base_reward: float, lle_reward: float,
                                             state_action: str, context: Dict, 
                                             is_novel: bool = False) -> float:
        """Calculate neuroplastic reward with LEA enhancement"""
        
        # Combine base and LLE rewards
        combined_reward = 0.6 * base_reward + 0.4 * lle_reward
        
        # Apply original neuroplasticity
        shaped_reward = self._apply_neuroplastic_shaping(combined_reward, state_action, context, is_novel)
        
        # LEA-specific enhancements
        lle_enhancement = self._calculate_lle_enhancement(lle_reward, state_action)
        final_reward = shaped_reward * lle_enhancement
        
        # Update LEA tracking
        self.lle_reward_history[state_action].append(lle_reward)
        if len(self.lle_reward_history[state_action]) > 20:
            self.lle_reward_history[state_action] = self.lle_reward_history[state_action][-20:]
        
        return final_reward
    
    def _apply_neuroplastic_shaping(self, reward: float, state_action: str, 
                                  context: Dict, is_novel: bool) -> float:
        """Apply original neuroplastic shaping"""
        
        shaped_reward = reward
        
        # Synaptic strength modulation
        synaptic_strength = self.synaptic_weights[state_action]['strength']
        strength_modulation = 1.0 + 0.2 * np.tanh(synaptic_strength)
        shaped_reward *= strength_modulation
        
        # Novelty bonus
        if is_novel:
            novelty_bonus = min(2.0, 1.0 + 0.5 * np.exp(-abs(synaptic_strength)))
            shaped_reward *= novelty_bonus
        
        # Hebbian trace influence
        hebbian_trace = self.hebbian_traces[state_action]['trace']
        trace_modulation = 1.0 + 0.1 * hebbian_trace
        shaped_reward *= trace_modulation
        
        # Update synaptic strength
        coactivation = context.get('coactivation', 1.0)
        self.update_synaptic_strength(state_action, shaped_reward, coactivation)
        
        return shaped_reward
    
    def _calculate_lle_enhancement(self, lle_reward: float, state_action: str) -> float:
        """Calculate LLE-specific enhancement factor"""
        
        # Recent LLE reward trend
        recent_lle_rewards = self.lle_reward_history[state_action][-5:]
        
        if len(recent_lle_rewards) < 2:
            return 1.0
        
        # Trend analysis with NaN protection
        trend = 0.0
        if len(recent_lle_rewards) >= 4:
            recent_mean = np.mean(recent_lle_rewards[-2:])
            earlier_mean = np.mean(recent_lle_rewards[:-2])
            if np.isfinite(recent_mean) and np.isfinite(earlier_mean):
                trend = recent_mean - earlier_mean
        
        # Enhancement based on LLE consistency
        lle_std = np.std(recent_lle_rewards) if len(recent_lle_rewards) > 1 else 0
        if not np.isfinite(lle_std):
            lle_std = 1.0
        consistency_bonus = 1.0 + 0.1 * (1.0 / (1.0 + lle_std))
        
        # Trend-based adjustment
        trend_adjustment = 1.0 + 0.05 * np.tanh(trend)
        
        return consistency_bonus * trend_adjustment
    
    def update_synaptic_strength(self, state_action: str, reward: float, coactivation: float = 1.0):
        """Update synaptic strength with LLE integration"""
        
        # Original Hebbian learning
        if np.isfinite(reward) and np.isfinite(coactivation):
            hebbian_update = self.learning_rate * reward * coactivation
            self.synaptic_weights[state_action]['strength'] += hebbian_update
        
        # Decay
        self.synaptic_weights[state_action]['strength'] *= 0.99
        
        # Update Hebbian traces
        if np.isfinite(reward):
            self.hebbian_traces[state_action]['trace'] = (
                0.9 * self.hebbian_traces[state_action]['trace'] + 0.1 * reward
            )
    
    def apply_homeostatic_scaling(self, state_action: str, target_activity: float = 0.1):
        """Apply homeostatic scaling to maintain network stability"""
        current_strength = self.synaptic_weights[state_action]['strength']
        scaling_factor = target_activity / max(abs(current_strength), 1e-6)
        
        # Gradual scaling to avoid instability
        self.synaptic_weights[state_action]['strength'] *= (1.0 + 0.01 * (scaling_factor - 1.0))
    
    def calculate_meta_learning_bonus(self, state_action: str, recent_performance: List[float]) -> float:
        """Calculate meta-learning bonus based on adaptation patterns"""
        if len(recent_performance) < 3:
            return 0.0
        
        # Calculate learning trend
        trend = np.polyfit(range(len(recent_performance)), recent_performance, 1)[0]
        
        # Reward positive learning trends
        meta_bonus = 0.1 * np.tanh(trend * 10)  # Scale and bound the bonus
        
        # Store meta-learning signal
        self.meta_rewards[state_action].append(meta_bonus)
        if len(self.meta_rewards[state_action]) > 50:
            self.meta_rewards[state_action] = self.meta_rewards[state_action][-50:]
        
        return meta_bonus
    
    def get_neuroplasticity_features(self, state_action: str) -> np.ndarray:
        """Extract neuroplasticity features for state encoding"""
        synaptic_strength = self.synaptic_weights[state_action]['strength']
        hebbian_trace = self.hebbian_traces[state_action]['trace']
        
        # Recent meta-learning performance
        recent_meta = self.meta_rewards[state_action][-5:] if self.meta_rewards[state_action] else [0.0]
        avg_meta_performance = np.mean(recent_meta)
        
        # Adaptation indicators
        recent_lle_rewards = self.lle_reward_history[state_action][-10:]
        lle_consistency = 1.0 / (1.0 + np.std(recent_lle_rewards)) if len(recent_lle_rewards) > 1 else 0.5
        
        # Connection stability
        connection_age = len(self.lle_reward_history[state_action])
        stability_score = min(1.0, connection_age / 100.0)  # Normalize by expected max interactions
        
        features = np.array([
            synaptic_strength,
            hebbian_trace,
            avg_meta_performance,
            lle_consistency,
            stability_score
        ])
        
       
        features = np.nan_to_num(features, nan=0.0, posinf=1.0, neginf=-1.0)
        
        return features
    
    def adapt_learning_rate(self, performance_trend: float):
        """Adapt learning rate based on performance trends"""
        if performance_trend > 0:
           
            self.learning_rate = min(0.3, self.learning_rate * 1.01)
        else:
            
            self.learning_rate = max(0.01, self.learning_rate * 0.99)
        
       
        self.adaptation_history.append({
            'trend': performance_trend,
            'new_lr': self.learning_rate,
            'timestamp': len(self.adaptation_history)
        })
        
       
        if len(self.adaptation_history) > 1000:
            self.adaptation_history = self.adaptation_history[-1000:]
