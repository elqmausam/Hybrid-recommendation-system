"""
Main Hybrid LEA-Neuroplasticity Recommendation System

This module contains the main system class that orchestrates all components
for comprehensive recommendation generation and learning.
"""

import numpy as np
import pandas as pd
import json
import torch
import time
from collections import defaultdict
from typing import Dict, List, Optional
import logging

from .core.lea_environment import LLMEnvironmentModel
from .core.neuroplasticity import EnhancedNeuroplasticityRewardShaper
from .core.state_encoder import HybridLEANeuroplasticStateEncoder
from .core.dqn_agent import EnhancedDQNAgent

logger = logging.getLogger(__name__)


class HybridLEANeuroplasticRecommendationSystem:
    """Complete hybrid system combining LEA + Neuroplasticity + DQN"""
    
    def __init__(self, max_aids: int = 100000, state_dim: int = 512):
        self.max_aids = max_aids
        self.state_dim = state_dim
        
        
        self.lle_model = LLMEnvironmentModel()
        self.neuroplastic_rewarder = EnhancedNeuroplasticityRewardShaper()
        self.state_encoder = HybridLEANeuroplasticStateEncoder(max_aids, state_dim=state_dim)
        
      
        self.dqn_agent = EnhancedDQNAgent(state_dim=state_dim)
        
        
        self.item_stats = defaultdict(lambda: {'clicks': 0, 'carts': 0, 'orders': 0})
        self.active_sessions = {}
        self.session_histories = defaultdict(list)
        
       
        self.training_mode = True
        self.recommendation_size = 10
        self.lea_update_frequency = 50
        self.sessions_processed = 0
        
      
        self.reward_config = {
            'clicks': 1.0,
            'carts': 5.0,
            'orders': 20.0,
            'negative_sample': -0.5,
            'lea_weight': 0.4,
            'neuroplastic_weight': 0.6
        }
        
    def preprocess_with_lea_learning(self, df: pd.DataFrame):
        """Enhanced preprocessing with LEA learning"""
        logger.info("Starting LEA + Neuroplasticity preprocessing...")
        
      
        self._build_basic_statistics(df)
        
        
        self.lle_model.learn_from_interactions(df)
        
        logger.info(f"LEA preprocessing complete: {len(self.item_stats)} items processed")
    
    def _build_basic_statistics(self, df: pd.DataFrame):
        """Build basic item statistics"""
        for _, row in df.iterrows():
            try:
                events = json.loads(row['events'].replace("'", '"'))
            except:
                continue
            
            for event in events:
                aid = min(event['aid'], self.max_aids)
                event_type = event['type']
                self.item_stats[aid][event_type] += 1
                self.item_stats[aid]['total'] = sum(self.item_stats[aid].values()) - self.item_stats[aid].get('total', 0)
    
    def get_hybrid_recommendations(self, session_id: int, session_events: List[Dict]) -> List[int]:
        """Get recommendations using hybrid LEA + neuroplasticity + DQN approach"""
        
        
        state = self.encode_hybrid_state(session_events, session_id)
        
        
        traditional_candidates = self._get_traditional_candidates(session_events)
        lea_candidates = self.lle_model.generate_positive_actions(session_events, k=100)
        
        
        all_candidates = list(set(traditional_candidates + lea_candidates))
        
        if not all_candidates:
            
            popular_items = sorted(
                self.item_stats.keys(),
                key=lambda x: self.item_stats[x].get('total', 0),
                reverse=True
            )[:self.recommendation_size]
            return list(popular_items)
        
        
        recommendations = []
        used_candidates = set()
        
        for _ in range(min(self.recommendation_size, len(all_candidates))):
            available = [c for c in all_candidates if c not in used_candidates]
            if not available:
                break
                
            selected_aid = self.dqn_agent.select_action(state, available)
            recommendations.append(selected_aid)
            used_candidates.add(selected_aid)
        
        
        self.active_sessions[session_id] = {
            'state': state,
            'recommendations': recommendations,
            'lea_candidates': lea_candidates,
            'timestamp': time.time(),
            'hybrid_enhanced': True
        }
        
        return recommendations
    
    def _get_traditional_candidates(self, session_events: List[Dict], k: int = 200) -> List[int]:
        """Get traditional collaborative filtering candidates"""
        candidates = set()
        
        if not session_events:
            popular_items = sorted(
                self.item_stats.keys(),
                key=lambda x: self.item_stats[x].get('total', 0),
                reverse=True
            )[:k]
            return list(popular_items)
        
        recent_aids = [event['aid'] for event in session_events[-10:]]
        
        
        for aid in recent_aids:
            
            for other_aid, stats in list(self.item_stats.items())[:1000]:  
                if other_aid != aid and stats.get('total', 0) > 5:
                    candidates.add(other_aid)
        
        
        popular_items = sorted(
            self.item_stats.keys(),
            key=lambda x: self.item_stats[x].get('total', 0),
            reverse=True
        )[:100]
        candidates.update(popular_items)
        
        # Remove session items
        session_aids = set(event['aid'] for event in session_events)
        candidates -= session_aids
        
        return list(candidates)[:k]
    
    def update_with_hybrid_feedback(self, session_id: int, clicked_aid: int, 
                                  event_type: str, new_session_events: List[Dict]):
        """Update with hybrid LEA + neuroplasticity feedback"""
        
        if session_id not in self.active_sessions:
            return
        
        session_data = self.active_sessions[session_id]
        prev_state = session_data['state']
        recommendations = session_data['recommendations']
        
        
        base_reward = self.reward_config.get(event_type, 0.0)
        
        if clicked_aid in recommendations:
            position_bonus = (len(recommendations) - recommendations.index(clicked_aid)) / len(recommendations)
            traditional_reward = base_reward * (1 + position_bonus)
        else:
            traditional_reward = self.reward_config['negative_sample']
        
        
        previous_events = new_session_events[:-1] if len(new_session_events) > 1 else []
        next_event = new_session_events[-1] if new_session_events else None
        lle_reward = self.lle_model.calculate_lle_reward(previous_events, clicked_aid, next_event)
        
       
        state_action_key = f"session_{session_id}_action_{clicked_aid}"
        is_novel = clicked_aid not in [e['aid'] for e in self.session_histories[session_id]]
        
        context = {
            'coactivation': len([r for r in recommendations if r in [e['aid'] for e in new_session_events]]) / max(1, len(recommendations)),
            'session_length': len(new_session_events),
            'event_type': event_type,
            'lea_candidate': clicked_aid in session_data.get('lea_candidates', [])
        }
        
        
        final_reward = self.neuroplastic_rewarder.calculate_enhanced_neuroplastic_reward(
            traditional_reward, lle_reward, state_action_key, context, is_novel
        )
        
       
        next_state = self.encode_hybrid_state(new_session_events, session_id)
        
      
        if np.isfinite(final_reward):
            self.dqn_agent.store_experience(
                prev_state, clicked_aid, final_reward, next_state, False
            )
        
        
        for rec_aid in recommendations[:5]: 
            if rec_aid != clicked_aid:
                neg_state_action_key = f"session_{session_id}_action_{rec_aid}"
                neg_lle_reward = self.lle_model.calculate_lle_reward(previous_events, rec_aid, None)
                
                neg_reward = self.neuroplastic_rewarder.calculate_enhanced_neuroplastic_reward(
                    self.reward_config['negative_sample'], neg_lle_reward, 
                    neg_state_action_key, context, False
                )
                
                if np.isfinite(neg_reward):
                    self.dqn_agent.store_experience(
                        prev_state, rec_aid, neg_reward, next_state, False
                    )
        
       
        if self.training_mode and self.sessions_processed % 10 == 0:  
            self.dqn_agent.train()
        
        self.session_histories[session_id].extend(new_session_events)
        
     
        self.sessions_processed += 1
        if self.sessions_processed % self.lea_update_frequency == 0:
            self._update_lea_insights()
    
    def _update_lea_insights(self):
        """Periodically update LEA insights"""
        logger.info(f"Updating LEA insights after {self.sessions_processed} sessions...")
        
        
        for aid in list(self.item_stats.keys())[:1000]:  
            total = self.item_stats[aid].get('total', 1)
            self.lle_model.item_features[aid]['total_interactions'] = total
            for event_type in ['clicks', 'carts', 'orders']:
                self.lle_model.item_features[aid][f'{event_type}_count'] = self.item_stats[aid].get(event_type, 0)
        
       
        limited_items = dict(list(self.lle_model.item_features.items())[:1000])
        original_features = self.lle_model.item_features
        self.lle_model.item_features = defaultdict(dict, limited_items)
        self.lle_model._generate_semantic_embeddings()
        self.lle_model.item_features = original_features
        
        logger.info("LEA insights updated successfully")
    
    def encode_hybrid_state(self, session_events: List[Dict], session_id: int) -> torch.Tensor:
        """Encode state using hybrid LEA + neuroplasticity approach"""
        
        if not session_events:
            return torch.zeros(1, self.state_dim)
        
       
        self.state_encoder.eval()
        
       
        recent_events = session_events[-30:]
        max_len = 30
        sequence = torch.zeros(1, max_len, 4)
        
        for i, event in enumerate(recent_events):
            if i >= max_len:
                break
            sequence[0, i, 0] = min(event['aid'], self.max_aids)
            sequence[0, i, 1] = {'clicks': 1, 'carts': 2, 'orders': 3}.get(event['type'], 1)
            sequence[0, i, 2] = event['ts'] / 1e12
            sequence[0, i, 3] = i
        
        
        lea_states = self.lle_model.generate_enhanced_state(session_events, session_id)
        
       
        lea_tensor_states = {
            'sequential_features': torch.tensor(lea_states['sequential_features'], dtype=torch.float32).unsqueeze(0),
            'contextual_features': torch.tensor(lea_states['contextual_features'], dtype=torch.float32).unsqueeze(0),
            'intent_features': torch.tensor(lea_states['intent_features'], dtype=torch.float32).unsqueeze(0),
            'temporal_features': torch.tensor(lea_states['temporal_features'], dtype=torch.float32).unsqueeze(0)
        }
        
       
        state_action_key = f"session_{session_id}_state"
        neuroplastic_features = self.neuroplastic_rewarder.get_neuroplasticity_features(state_action_key)
        neuroplastic_tensor = torch.tensor([neuroplastic_features], dtype=torch.float32)
        
      
        with torch.no_grad():
            state = self.state_encoder(sequence, lea_tensor_states, neuroplastic_tensor)
        
        return state
    
    def get_comprehensive_analytics(self) -> Dict:
        """Get comprehensive system analytics"""
        analytics = {
            'basic_stats': {
                'total_items': len(self.item_stats),
                'active_sessions': len(self.active_sessions),
                'sessions_processed': self.sessions_processed,
                'training_mode': self.training_mode
            },
            'dqn_metrics': self.dqn_agent.get_metrics(),
            'lea_metrics': {
                'user_patterns': len(self.lle_model.user_patterns),
                'item_features': len(self.lle_model.item_features),
                'semantic_embeddings': len(self.lle_model.item_semantic_embeddings),
                'transition_patterns': len(self.lle_model.transition_probabilities),
                'positive_action_pool': sum(len(actions) for actions in self.lle_model.positive_action_pool.values())
            },
            'neuroplasticity_metrics': {
                'synaptic_connections': len(self.neuroplastic_rewarder.synaptic_weights),
                'hebbian_traces': len(self.neuroplastic_rewarder.hebbian_traces),
                'adaptation_events': len(self.neuroplastic_rewarder.adaptation_history),
                'lle_reward_tracking': len(self.neuroplastic_rewarder.lle_reward_history)
            }
        }
        
    
        if self.neuroplastic_rewarder.synaptic_weights:
            strengths = []
            for data in self.neuroplastic_rewarder.synaptic_weights.values():
                strength = data.get('strength', 0)
                if np.isfinite(strength):
                    strengths.append(strength)
            
            if strengths:
                analytics['neuroplasticity_metrics']['avg_synaptic_strength'] = np.mean(strengths)
                analytics['neuroplasticity_metrics']['synaptic_strength_std'] = np.std(strengths)
        
     
        all_lle_rewards = []
        for rewards in self.neuroplastic_rewarder.lle_reward_history.values():
            for reward in rewards:
                if np.isfinite(reward):
                    all_lle_rewards.append(reward)
        
        if all_lle_rewards:
            analytics['lea_metrics']['avg_lle_reward'] = np.mean(all_lle_rewards)
            analytics['lea_metrics']['lle_reward_std'] = np.std(all_lle_rewards)
        
        return analytics
    
    def save_system(self, path: str):
        """Save the complete system state"""
        import joblib
        
       
        dqn_path = path.replace('.joblib', '_dqn.pth')
        self.dqn_agent.save_model(dqn_path)
        
       
        system_state = {
            'lle_model': self.lle_model,
            'neuroplastic_rewarder': self.neuroplastic_rewarder,
            'item_stats': dict(self.item_stats),
            'session_histories': dict(self.session_histories),
            'reward_config': self.reward_config,
            'sessions_processed': self.sessions_processed,
            'max_aids': self.max_aids,
            'state_dim': self.state_dim
        }
        
        joblib.dump(system_state, path)
        logger.info(f"System saved to {path} and {dqn_path}")
    
    def load_system(self, path: str):
        """Load the complete system state"""
        import joblib
        
      
        system_state = joblib.load(path)
        
        self.lle_model = system_state['lle_model']
        self.neuroplastic_rewarder = system_state['neuroplastic_rewarder']
        self.item_stats = defaultdict(lambda: {'clicks': 0, 'carts': 0, 'orders': 0}, system_state['item_stats'])
        self.session_histories = defaultdict(list, system_state['session_histories'])
        self.reward_config = system_state['reward_config']
        self.sessions_processed = system_state['sessions_processed']
        
       
        dqn_path = path.replace('.joblib', '_dqn.pth')
        try:
            self.dqn_agent.load_model(dqn_path)
            logger.info(f"System loaded from {path} and {dqn_path}")
        except FileNotFoundError:
            logger.warning(f"DQN model not found at {dqn_path}, using fresh DQN")
    
    def set_training_mode(self, mode: bool):
        """Set training mode on/off"""
        self.training_mode = mode
        if mode:
            self.state_encoder.train()
        else:
            self.state_encoder.eval()
        
        logger.info(f"Training mode set to: {mode}")
    
    def get_session_insights(self, session_id: int, session_events: List[Dict]) -> Dict:
        """Get detailed insights for a specific session"""
        if not session_events:
            return {}
        
        
        lea_states = self.lle_model.generate_enhanced_state(session_events, session_id)
        

        intent_features = lea_states['intent_features']
        purchase_likelihood = intent_features[4]
        
       
        session_aids = [e['aid'] for e in session_events]
        session_types = [e['type'] for e in session_events]
        
        insights = {
            'session_summary': {
                'length': len(session_events),
                'unique_items': len(set(session_aids)),
                'purchase_likelihood': float(purchase_likelihood),
                'session_quality': float(intent_features[5]),
            },
            'behavior_analysis': {
                'exploration_score': float(intent_features[0]),
                'consideration_score': float(intent_features[1]),
                'purchase_score': float(intent_features[2]),
                'diversity_score': float(intent_features[3]),
            },
            'temporal_patterns': {
                'session_duration': float(lea_states['temporal_features'][0]),
                'avg_time_between_actions': float(lea_states['temporal_features'][1]),
                'quick_transitions': int(lea_states['temporal_features'][2]),
                'events_per_minute': float(lea_states['temporal_features'][3]),
            }
        }
        
        return insights
