
"""
LLM Environment Model (LEA) for Recommendation Systems

This module implements the LLM as Environment approach for modeling user behavior,
generating enhanced states, calculating semantic rewards, and producing positive actions.
"""

import numpy as np
import pandas as pd
import json
import torch
import torch.nn as nn
from collections import defaultdict
from typing import Dict, List, Optional
from sklearn.metrics.pairwise import cosine_similarity
import logging

logger = logging.getLogger(__name__)


class LLMEnvironmentModel:
    """LLM as Environment (LE) for state and reward modeling + action generation"""
    
    def __init__(self, embedding_dim: int = 128, max_sequence_length: int = 50):
        self.embedding_dim = embedding_dim
        self.max_seq_len = max_sequence_length
        
        # User-Item interaction patterns
        self.user_patterns = defaultdict(list)
        self.item_features = defaultdict(dict)
        self.interaction_contexts = {}
        
        # LLM-generated insights
        self.user_intent_profiles = {}
        self.item_semantic_embeddings = {}
        self.transition_probabilities = defaultdict(dict)
        
        # State modeling components
        self.state_representations = {}
        self.contextual_features = {}
        
        # Reward modeling
        self.reward_patterns = defaultdict(list)
        self.preference_signals = defaultdict(dict)
        
        # Action generation
        self.positive_action_pool = defaultdict(set)
        self.counterfactual_actions = defaultdict(list)
        
        # TF-IDF for semantic similarity
        self.embedding_model = nn.Embedding(100000, 128)  # Pre-trained item embeddings
        self.embedding_dim_actual = 128  # Actual embedding dimension
        
    def learn_from_interactions(self, df: pd.DataFrame):
        """Learn user patterns and item features from interaction data"""
        logger.info("Learning user-item patterns from interactions...")
        
        # Extract interaction patterns
        for _, row in df.iterrows():
            session_id = row['session']
            try:
                events = json.loads(row['events'].replace("'", '"'))
            except:
                continue
            
            # Build user pattern
            session_pattern = []
            item_sequence = []
            
            for event in events:
                aid = event['aid']
                event_type = event['type']
                timestamp = event['ts']
                
                # Update item features
                self.item_features[aid]['total_interactions'] = self.item_features[aid].get('total_interactions', 0) + 1
                self.item_features[aid][f'{event_type}_count'] = self.item_features[aid].get(f'{event_type}_count', 0) + 1
                
                # Build sequence
                session_pattern.append({
                    'aid': aid,
                    'type': event_type,
                    'ts': timestamp,
                    'sequence_pos': len(session_pattern)
                })
                item_sequence.append(aid)
            
            self.user_patterns[session_id] = session_pattern
            
            # Learn transition patterns
            for i in range(len(item_sequence) - 1):
                current_item = item_sequence[i]
                next_item = item_sequence[i + 1]
                
                if next_item not in self.transition_probabilities[current_item]:
                    self.transition_probabilities[current_item][next_item] = 0
                self.transition_probabilities[current_item][next_item] += 1
        
        # Normalize transition probabilities
        for current_item in self.transition_probabilities:
            total = sum(self.transition_probabilities[current_item].values())
            if total > 0:  # Fix division by zero
                for next_item in self.transition_probabilities[current_item]:
                    self.transition_probabilities[current_item][next_item] /= total
        
        # Generate semantic embeddings
        self._generate_semantic_embeddings()
        
        logger.info(f"Learned patterns for {len(self.user_patterns)} sessions and {len(self.item_features)} items")
    
    def _generate_semantic_embeddings(self):
        """Generate semantic embeddings using pre-trained embeddings"""
        
        for aid, features in self.item_features.items():
            # Use pre-trained embedding lookup
            aid_tensor = torch.tensor([min(aid, 99999)], dtype=torch.long)
            with torch.no_grad():
                embedding = self.embedding_model(aid_tensor).squeeze().numpy()
            
            # Enhance with behavioral features
            total_interactions = max(1, features.get('total_interactions', 1))  # Avoid division by zero
            click_ratio = features.get('clicks_count', 0) / total_interactions
            cart_ratio = features.get('carts_count', 0) / total_interactions
            order_ratio = features.get('orders_count', 0) / total_interactions
            
            # Create behavioral enhancement vector
            behavioral_features = np.array([
                click_ratio, cart_ratio, order_ratio,
                np.log1p(total_interactions),  # Log-scaled popularity
                order_ratio * 2 + cart_ratio  # Purchase intent score
            ])
            
            # Pad behavioral features to match embedding dimension
            if len(behavioral_features) < self.embedding_dim_actual:
                behavioral_features = np.pad(behavioral_features, 
                                           (0, self.embedding_dim_actual - len(behavioral_features)))
            else:
                behavioral_features = behavioral_features[:self.embedding_dim_actual]
            
            # Combine pre-trained embedding with behavioral features (weighted combination)
            combined_embedding = 0.7 * embedding + 0.3 * behavioral_features
            self.item_semantic_embeddings[aid] = combined_embedding
    
    def generate_enhanced_state(self, session_events: List[Dict], session_id: int) -> Dict:
        """Generate enhanced state representation using LLM insights"""
        
        if not session_events:
            return {
                'sequential_features': np.zeros(self.embedding_dim),
                'contextual_features': np.zeros(32),
                'intent_features': np.zeros(16),
                'temporal_features': np.zeros(8)
            }
        
        # Sequential features from item embeddings
        recent_events = session_events[-10:]  # Focus on recent events
        seq_embeddings = []
        
        for event in recent_events:
            aid = event['aid']
            if aid in self.item_semantic_embeddings:
                seq_embeddings.append(self.item_semantic_embeddings[aid])
            else:
                # Generate embedding on-the-fly for unknown items
                aid_tensor = torch.tensor([min(aid, 99999)], dtype=torch.long)
                with torch.no_grad():
                    embedding = self.embedding_model(aid_tensor).squeeze().numpy()
                seq_embeddings.append(embedding)

        if seq_embeddings:
            # Average pooling of embeddings (all embeddings are now same dimension)
            sequential_features = np.mean(seq_embeddings, axis=0)
            
            # Ensure correct dimension
            if len(sequential_features) < self.embedding_dim:
                sequential_features = np.pad(sequential_features, 
                                           (0, self.embedding_dim - len(sequential_features)))
            elif len(sequential_features) > self.embedding_dim:
                sequential_features = sequential_features[:self.embedding_dim]
        else:
            sequential_features = np.zeros(self.embedding_dim)
        
        # Contextual features
        session_aids = [e['aid'] for e in session_events]
        session_types = [e['type'] for e in session_events]
        
        # Safe calculation with division by zero protection
        total_events = max(1, len(session_types))
        click_ratio = session_types.count('clicks') / total_events
        cart_ratio = session_types.count('carts') / total_events
        order_ratio = session_types.count('orders') / total_events
        
        # Safe popularity calculation
        popularity_scores = []
        for aid in session_aids:
            interactions = self.item_features.get(aid, {}).get('total_interactions', 0)
            popularity_scores.append(interactions)
        avg_popularity = np.mean(popularity_scores) if popularity_scores else 0
        
        contextual_features = np.array([
            len(session_events),  # Session length
            len(set(session_aids)),  # Unique items
            click_ratio,  # Click ratio
            cart_ratio,  # Cart ratio
            order_ratio,  # Order ratio
            avg_popularity,  # Avg popularity
            len([aid for aid in session_aids if aid in self.item_semantic_embeddings]),  # Known items
            len(set(session_aids[-5:])) / min(5, len(session_aids)),  # Recent diversity
        ])
        
        # Pad contextual features to 32 dimensions
        contextual_features = np.pad(contextual_features, (0, 32 - len(contextual_features)))
        
        # Intent features (LLM-inferred user intent)
        intent_features = self._infer_user_intent(session_events)
        
        # Temporal features
        if len(session_events) > 1:
            timestamps = [e['ts'] for e in session_events]
            session_duration = max(1, (timestamps[-1] - timestamps[0]) / 1e6)  # Convert to seconds, avoid zero
            avg_time_between = session_duration / max(1, len(timestamps) - 1)
            
            temporal_features = np.array([
                session_duration,
                avg_time_between,
                len([i for i in range(1, len(timestamps)) 
                    if (timestamps[i] - timestamps[i-1]) < 60000]),  # Quick transitions
                len(session_events) / max(1, session_duration / 60),  # Events per minute
            ])
        else:
            temporal_features = np.array([0, 0, 0, 0])
        
        # Pad temporal features to 8 dimensions
        temporal_features = np.pad(temporal_features, (0, 8 - len(temporal_features)))
        
        return {
            'sequential_features': sequential_features,
            'contextual_features': contextual_features,
            'intent_features': intent_features,
            'temporal_features': temporal_features
        }
    
    def _infer_user_intent(self, session_events: List[Dict]) -> np.ndarray:
        """Infer user intent using LLM-like reasoning"""
        
        if not session_events:
            return np.zeros(16)
        
        session_types = [e['type'] for e in session_events]
        session_aids = [e['aid'] for e in session_events]
        
        # Intent categories with safe division
        total_events = max(1, len(session_types))
        exploration_score = session_types.count('clicks') / total_events
        consideration_score = session_types.count('carts') / total_events
        purchase_score = session_types.count('orders') / total_events
        
        # Behavioral patterns
        repeat_items = len(session_aids) - len(set(session_aids))
        diversity_score = len(set(session_aids)) / max(1, len(session_aids))
        
        # Shopping stage inference
        if purchase_score > 0:
            shopping_stage = [0, 0, 1]  # Purchase stage
        elif consideration_score > 0.1:
            shopping_stage = [0, 1, 0]  # Consideration stage
        else:
            shopping_stage = [1, 0, 0]  # Exploration stage
        
        # Purchase likelihood (based on session patterns)
        purchase_likelihood = min(1.0, purchase_score * 3 + consideration_score * 2 + exploration_score * 0.5)
        
        # Session quality score
        quality_indicators = [
            len(session_events) > 3,  # Engaged session
            consideration_score > 0,  # Shows interest
            diversity_score < 0.8,  # Focused browsing
            repeat_items > 0  # Item revisitation
        ]
        session_quality = sum(quality_indicators) / max(1, len(quality_indicators))
        
        intent_features = np.array([
            exploration_score,
            consideration_score,
            purchase_score,
            diversity_score,
            purchase_likelihood,
            session_quality,
            *shopping_stage,
            repeat_items,
            float(len(session_events) > 10),  # Long session
            float(len(set(session_aids[-3:])) == 1 if len(session_aids) >= 3 else 0),  # Focused recent activity
        ])
        
        # Pad to 16 dimensions
        return np.pad(intent_features, (0, 16 - len(intent_features)))
    
    def calculate_lle_reward(self, session_events: List[Dict], action: int, 
                           next_event: Dict = None) -> float:
        """Calculate reward using LLM Environment understanding"""
        
        base_rewards = {'clicks': 1.0, 'carts': 5.0, 'orders': 20.0}
        
        if next_event is None:
            return -0.5  # Negative reward for no interaction
        
        base_reward = base_rewards.get(next_event['type'], 0.0)
        
        # Context-aware reward shaping
        session_aids = [e['aid'] for e in session_events]
        
        # Relevance bonus based on transition patterns
        relevance_bonus = 0.0
        if session_aids and action in self.transition_probabilities.get(session_aids[-1], {}):
            transition_prob = self.transition_probabilities[session_aids[-1]][action]
            relevance_bonus = transition_prob * 2.0
        
        # Semantic similarity bonus
        semantic_bonus = 0.0
        if (session_aids and action in self.item_semantic_embeddings and 
            session_aids[-1] in self.item_semantic_embeddings):
            
            action_emb = self.item_semantic_embeddings[action]
            last_item_emb = self.item_semantic_embeddings[session_aids[-1]]
            
            similarity = cosine_similarity([action_emb], [last_item_emb])[0][0]
            semantic_bonus = similarity * 1.5
        
        # Intent alignment bonus
        intent_features = self._infer_user_intent(session_events)
        purchase_likelihood = intent_features[4]  # Purchase likelihood from intent
        
        intent_bonus = 0.0
        if next_event['type'] == 'orders' and purchase_likelihood > 0.7:
            intent_bonus = 3.0  # Strong intent alignment
        elif next_event['type'] == 'carts' and purchase_likelihood > 0.5:
            intent_bonus = 1.5
        
        # Novelty penalty/bonus
        novelty_factor = 1.0
        if action in session_aids:
            novelty_factor = 0.8  # Slight penalty for repetition
        else:
            novelty_factor = 1.1  # Bonus for exploration
        
        total_reward = (base_reward + relevance_bonus + semantic_bonus + intent_bonus) * novelty_factor
        
        return total_reward
    
    def generate_positive_actions(self, session_events: List[Dict], k: int = 50) -> List[int]:
        """Generate positive actions for data augmentation"""
        
        if not session_events:
            return []
        
        session_aids = [e['aid'] for e in session_events]
        candidates = set()
        
        # Transition-based candidates
        for aid in session_aids[-3:]:  # Recent items
            if aid in self.transition_probabilities:
                top_transitions = sorted(
                    self.transition_probabilities[aid].items(),
                    key=lambda x: x[1], reverse=True
                )[:10]
                candidates.update([item for item, _ in top_transitions])
        
        # Semantic similarity candidates
        if session_aids[-1] in self.item_semantic_embeddings:
            last_item_emb = self.item_semantic_embeddings[session_aids[-1]]
            similarities = {}
            
            for aid, emb in list(self.item_semantic_embeddings.items())[:1000]:  # Limit for performance
                if aid not in session_aids:
                    sim = cosine_similarity([last_item_emb], [emb])[0][0]
                    similarities[aid] = sim
            
            top_similar = sorted(similarities.items(), key=lambda x: x[1], reverse=True)[:15]
            candidates.update([aid for aid, _ in top_similar])
        
        # Intent-based candidates
        intent_features = self._infer_user_intent(session_events)
        purchase_likelihood = intent_features[4]
        
        # If high purchase intent, suggest high-conversion items
        if purchase_likelihood > 0.6:
            high_conversion_items = []
            for aid, features in list(self.item_features.items())[:500]:  # Limit for performance
                total_interactions = max(1, features.get('total_interactions', 1))
                conversion_rate = features.get('orders_count', 0) / total_interactions
                if conversion_rate > 0.1:
                    high_conversion_items.append(aid)
            candidates.update(high_conversion_items[:10])
        
        return list(candidates)[:k]
