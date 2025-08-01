#!/usr/bin/env python3
"""
Training Script for Hybrid LEA-Neuroplasticity Recommendation System

This script handles the complete training pipeline including data loading,
preprocessing, training, and model saving.
"""

import argparse
import pandas as pd
import json
import numpy as np
import logging
from pathlib import Path
import time
from typing import Dict, List

from hybrid_recsys import HybridLEANeuroplasticRecommendationSystem


logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description="Train Hybrid LEA-Neuroplasticity Recommendation System"
    )
    
    parser.add_argument(
        "--data_path", 
        type=str, 
        required=True,
        help="Path to training dataset (CSV format)"
    )
    
    parser.add_argument(
        "--output_path", 
        type=str, 
        default="models/trained_hybrid_system.joblib",
        help="Path to save trained model"
    )
    
    parser.add_argument(
        "--max_sessions", 
        type=int, 
        default=10000,
        help="Maximum number of sessions to use for training"
    )
    
    parser.add_argument(
        "--max_aids", 
        type=int, 
        default=100000,
        help="Maximum number of unique item IDs"
    )
    
    parser.add_argument(
        "--state_dim", 
        type=int, 
        default=512,
        help="State embedding dimension"
    )
    
    parser.add_argument(
        "--batch_size", 
        type=int, 
        default=128,
        help="Training batch size"
    )
    
    parser.add_argument(
        "--validation_split", 
        type=float, 
        default=0.2,
        help="Fraction of data to use for validation"
    )
    
    parser.add_argument(
        "--verbose", 
        action="store_true",
        help="Enable verbose logging"
    )
    
    return parser.parse_args()


def load_and_validate_data(data_path: str, max_sessions: int = None) -> pd.DataFrame:
    """Load and validate training data"""
    logger.info(f"Loading data from {data_path}")
    
    try:
        df = pd.read_csv(data_path)
        logger.info(f"Loaded {len(df)} sessions")
        
        
        required_columns = ['session', 'events']
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            raise ValueError(f"Missing required columns: {missing_columns}")
        
       
        if max_sessions and len(df) > max_sessions:
            df = df.head(max_sessions)
            logger.info(f"Limited to {max_sessions} sessions for training")
        
     
        valid_sessions = []
        for idx, row in df.iterrows():
            try:
                events = json.loads(row['events'].replace("'", '"'))
                if isinstance(events, list) and len(events) > 0:
                    valid_sessions.append(idx)
            except:
                continue
        
        df = df.loc[valid_sessions]
        logger.info(f"Validated {len(df)} sessions with proper event format")
        
        return df
        
    except FileNotFoundError:
        raise FileNotFoundError(f"Data file not found: {data_path}")
    except Exception as e:
        raise Exception(f"Error loading data: {str(e)}")


def split_data(df: pd.DataFrame, validation_split: float) -> tuple:
    """Split data into training and validation sets"""
    n_train = int(len(df) * (1 - validation_split))
    
   
    df_shuffled = df.sample(frac=1, random_state=42).reset_index(drop=True)
    
    train_df = df_shuffled[:n_train]
    val_df = df_shuffled[n_train:]
    
    logger.info(f"Split data: {len(train_df)} training, {len(val_df)} validation sessions")
    
    return train_df, val_df


def train_system(system: HybridLEANeuroplasticRecommendationSystem, 
                train_df: pd.DataFrame, val_df: pd.DataFrame,
                verbose: bool = False) -> Dict:
    """Train the hybrid system"""
    logger.info("Starting training process...")
    
   
    system.preprocess_with_lea_learning(train_df)
    
   
    training_metrics = {
        'sessions_processed': 0,
        'total_rewards': [],
        'lea_rewards': [],
        'validation_scores': []
    }
    
    
    start_time = time.time()
    
    for session_idx, (_, row) in enumerate(train_df.iterrows()):
        session_id = row['session']
        
        try:
            events = json.loads(row['events'].replace("'", '"'))
        except:
            continue
        
        if len(events) < 2:
            continue
        
       
        current_events = []
        
        for event_idx, event in enumerate(events):
            current_events.append(event)
            
            
            if len(current_events) < 2:
                continue
            
          
            try:
                recommendations = system.get_hybrid_recommendations(
                    session_id, current_events[:-1]
                )
                
               
                system.update_with_hybrid_feedback(
                    session_id, event['aid'], event['type'], current_events
                )
                
                training_metrics['sessions_processed'] += 1
                
            except Exception as e:
                if verbose:
                    logger.warning(f"Error processing session {session_id}: {e}")
                continue
        
      
        if (session_idx + 1) % 100 == 0:
            elapsed_time = time.time() - start_time
            sessions_per_sec = (session_idx + 1) / elapsed_time
            
            logger.info(
                f"Processed {session_idx + 1}/{len(train_df)} sessions "
                f"({sessions_per_sec:.2f} sessions/sec)"
            )
            
            
            analytics = system.get_comprehensive_analytics()
            if verbose:
                logger.info(f"DQN Epsilon: {analytics['dqn_metrics']['epsilon']:.4f}")
                logger.info(f"Memory Size: {analytics['dqn_metrics']['memory_size']}")
        
        
        if (session_idx + 1) % 1000 == 0 and not val_df.empty:
            val_score = evaluate_system(system, val_df.head(100))  
            training_metrics['validation_scores'].append(val_score)
            logger.info(f"Validation Score: {val_score:.4f}")
    
    total_time = time.time() - start_time
    logger.info(f"Training completed in {total_time:.2f} seconds")
    
    return training_metrics


def evaluate_system(system: HybridLEANeuroplasticRecommendationSystem, 
                   val_df: pd.DataFrame) -> float:
    """Evaluate system performance on validation data"""
    total_score = 0.0
    valid_sessions = 0
    
    
    original_mode = system.training_mode
    system.set_training_mode(False)
    
    try:
        for _, row in val_df.iterrows():
            session_id = row['session'] + 1000000  
            
            try:
                events = json.loads(row['events'].replace("'", '"'))
            except:
                continue
            
            if len(events) < 3:
                continue
            
           
            split_point = len(events) // 2
            context_events = events[:split_point]
            target_events = events[split_point:]
            
            
            recommendations = system.get_hybrid_recommendations(session_id, context_events)
            
           
            target_aids = set(event['aid'] for event in target_events)
            recommendation_set = set(recommendations)
            
            hits = len(target_aids.intersection(recommendation_set))
            score = hits / len(target_aids) if target_aids else 0.0
            
            total_score += score
            valid_sessions += 1
    
    finally:
        
        system.set_training_mode(original_mode)
    
    return total_score / max(valid_sessions, 1)


def save_training_report(system: HybridLEANeuroplasticRecommendationSystem,
                        training_metrics: Dict, output_path: str):
    """Save training report and final analytics"""
    
   
    final_analytics = system.get_comprehensive_analytics()
    
    
    report = {
        'training_summary': {
            'sessions_processed': training_metrics['sessions_processed'],
            'validation_scores': training_metrics['validation_scores'],
            'final_validation_score': training_metrics['validation_scores'][-1] if training_metrics['validation_scores'] else None
        },
        'final_system_state': final_analytics
    }
    
   
    report_path = output_path.replace('.joblib', '_training_report.json')
    with open(report_path, 'w') as f:
        json.dump(report, f, indent=2, default=str)
    
    logger.info(f"Training report saved to {report_path}")


def main():
    """Main training function"""
    args = parse_arguments()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
   
    Path(args.output_path).parent.mkdir(parents=True, exist_ok=True)
    
    
    logger.info("Loading and preparing data...")
    df = load_and_validate_data(args.data_path, args.max_sessions)
    train_df, val_df = split_data(df, args.validation_split)
    

    logger.info("Initializing system...")
    system = HybridLEANeuroplasticRecommendationSystem(
        max_aids=args.max_aids,
        state_dim=args.state_dim
    )
    
    
    training_metrics = train_system(system, train_df, val_df, args.verbose)
    
   
    logger.info(f"Saving trained system to {args.output_path}")
    system.save_system(args.output_path)
    
   
    save_training_report(system, training_metrics, args.output_path)
    

    final_analytics = system.get_comprehensive_analytics()
    logger.info("\n" + "="*50)
    logger.info("TRAINING COMPLETED SUCCESSFULLY")
    logger.info("="*50)
    logger.info(f"Sessions Processed: {training_metrics['sessions_processed']}")
    logger.info(f"Final Action Space: {final_analytics['dqn_metrics']['action_space_size']}")
    logger.info(f"LEA Patterns Learned: {final_analytics['lea_metrics']['user_patterns']}")
    logger.info(f"Neuroplastic Connections: {final_analytics['neuroplasticity_metrics']['synaptic_connections']}")
    
    if training_metrics['validation_scores']:
        logger.info(f"Final Validation Score: {training_metrics['validation_scores'][-1]:.4f}")
    
    logger.info(f"Model saved to: {args.output_path}")


if __name__ == "__main__":
    main()
