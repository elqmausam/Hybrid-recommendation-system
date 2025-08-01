"""
Hybrid LEA-Neuroplasticity State Encoder

This module implements the hybrid state encoder that combines LEA states
with neuroplasticity features using transformer architecture.
"""

import torch
import torch.nn as nn
from typing import Dict
import logging

logger = logging.getLogger(__name__)


class HybridLEANeuroplasticStateEncoder(nn.Module):
    """Enhanced state encoder combining LEA states with neuroplasticity"""
    
    def __init__(self, max_aids: int = 100000, embedding_dim: int = 128, 
                 state_dim: int = 512, max_sequence_length: int = 50):
        super().__init__()
        self.max_aids = max_aids
        self.embedding_dim = embedding_dim
        self.state_dim = state_dim
        self.max_seq_len = max_sequence_length
        
       
        self.aid_embedding = nn.Embedding(max_aids + 1, embedding_dim, padding_idx=0)
        self.type_embedding = nn.Embedding(4, 32)
        self.position_embedding = nn.Embedding(max_sequence_length, 32)
        
       
        self.lea_sequential_encoder = nn.Linear(embedding_dim, 128)
        self.lea_contextual_encoder = nn.Linear(32, 64)
        self.lea_intent_encoder = nn.Linear(16, 32)
        self.lea_temporal_encoder = nn.Linear(8, 16)
        
        
        self.neuroplastic_encoder = nn.Sequential(
            nn.Linear(5, 32),  
            nn.ReLU(),
            nn.Dropout(0.1)
        )
        
        
        total_feature_dim = embedding_dim + 32 + 32 + 128 + 64 + 32 + 16 + 32
        
        self.attention_fusion = nn.MultiheadAttention(
            embed_dim=total_feature_dim,
            num_heads=8,
            batch_first=True,
            dropout=0.1
        )
        
        self.transformer_encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=total_feature_dim,
                nhead=8,
                dim_feedforward=1024,
                dropout=0.1,
                batch_first=True
            ),
            num_layers=4
        )
        
        
        self.state_projector = nn.Sequential(
            nn.Linear(total_feature_dim, 1024),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, state_dim),
            nn.LayerNorm(state_dim)
        )
        
    def forward(self, session_sequence: torch.Tensor, lea_states: Dict[str, torch.Tensor],
                neuroplastic_features: torch.Tensor) -> torch.Tensor:
        """Enhanced forward pass with LEA and neuroplasticity integration"""
        
        batch_size, seq_len = session_sequence.shape[:2]
        
        
        aids = session_sequence[:, :, 0].long()
        types = session_sequence[:, :, 1].long()
        positions = torch.arange(seq_len, device=session_sequence.device).unsqueeze(0).repeat(batch_size, 1)
        
        aid_emb = self.aid_embedding(aids)
        type_emb = self.type_embedding(types)
        pos_emb = self.position_embedding(positions)
        
        
        lea_sequential = self.lea_sequential_encoder(lea_states['sequential_features'])
        lea_contextual = self.lea_contextual_encoder(lea_states['contextual_features'])
        lea_intent = self.lea_intent_encoder(lea_states['intent_features'])
        lea_temporal = self.lea_temporal_encoder(lea_states['temporal_features'])
        
        
        lea_sequential_exp = lea_sequential.unsqueeze(1).repeat(1, seq_len, 1)
        lea_contextual_exp = lea_contextual.unsqueeze(1).repeat(1, seq_len, 1)
        lea_intent_exp = lea_intent.unsqueeze(1).repeat(1, seq_len, 1)
        lea_temporal_exp = lea_temporal.unsqueeze(1).repeat(1, seq_len, 1)
        
       
        neuroplastic_emb = self.neuroplastic_encoder(neuroplastic_features)
        neuroplastic_exp = neuroplastic_emb.unsqueeze(1).repeat(1, seq_len, 1)
        
        
        combined_features = torch.cat([
            aid_emb, type_emb, pos_emb,
            lea_sequential_exp, lea_contextual_exp, lea_intent_exp, lea_temporal_exp,
            neuroplastic_exp
        ], dim=-1)
        
        
        mask = (aids != 0)
        encoded_sequence = self.transformer_encoder(
            combined_features,
            src_key_padding_mask=~mask
        )
        
    
        mask_expanded = mask.unsqueeze(-1).float()
        pooled_sequence = (encoded_sequence * mask_expanded).sum(dim=1) / (mask_expanded.sum(dim=1) + 1e-8)
        
     
        final_state = self.state_projector(pooled_sequence)
        
        return final_state
    
    def get_attention_weights(self, session_sequence: torch.Tensor, lea_states: Dict[str, torch.Tensor],
                            neuroplastic_features: torch.Tensor) -> torch.Tensor:
        """Extract attention weights for interpretability"""
        batch_size, seq_len = session_sequence.shape[:2]
        
       
        aids = session_sequence[:, :, 0].long()
        types = session_sequence[:, :, 1].long()
        positions = torch.arange(seq_len, device=session_sequence.device).unsqueeze(0).repeat(batch_size, 1)
        
        aid_emb = self.aid_embedding(aids)
        type_emb = self.type_embedding(types)
        pos_emb = self.position_embedding(positions)
        
       
        lea_sequential = self.lea_sequential_encoder(lea_states['sequential_features'])
        lea_contextual = self.lea_contextual_encoder(lea_states['contextual_features'])
        lea_intent = self.lea_intent_encoder(lea_states['intent_features'])
        lea_temporal = self.lea_temporal_encoder(lea_states['temporal_features'])
        
        lea_sequential_exp = lea_sequential.unsqueeze(1).repeat(1, seq_len, 1)
        lea_contextual_exp = lea_contextual.unsqueeze(1).repeat(1, seq_len, 1)
        lea_intent_exp = lea_intent.unsqueeze(1).repeat(1, seq_len, 1)
        lea_temporal_exp = lea_temporal.unsqueeze(1).repeat(1, seq_len, 1)
        
        neuroplastic_emb = self.neuroplastic_encoder(neuroplastic_features)
        neuroplastic_exp = neuroplastic_emb.unsqueeze(1).repeat(1, seq_len, 1)
        
        combined_features = torch.cat([
            aid_emb, type_emb, pos_emb,
            lea_sequential_exp, lea_contextual_exp, lea_intent_exp, lea_temporal_exp,
            neuroplastic_exp
        ], dim=-1)
        
      
        _, attention_weights = self.attention_fusion(
            combined_features, combined_features, combined_features,
            need_weights=True
        )
        
        return attention_weights
