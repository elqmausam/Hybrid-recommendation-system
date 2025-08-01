"""
Basic Usage Example for Hybrid LEA-Neuroplasticity Recommendation System

This example demonstrates how to use the recommendation system for basic tasks.
"""

import pandas as pd
import json
from hybrid_recsys import HybridLEANeuroplasticRecommendationSystem


def main():
    
    

    system = HybridLEANeuroplasticRecommendationSystem(
        max_aids=100000,
        state_dim=512
    )
    
   
    data_path = "dataset.csv"
    try:
        df = pd.read_csv(data_path)
        print(f"✓ Loaded dataset with {len(df)} sessions")
        
        
        system.preprocess_with_lea_learning(df)
        print("✓ LEA learning completed")
        
    except FileNotFoundError:
        print("⚠️  Dataset not found, using synthetic data for demo")
        
        df = create_synthetic_data()
        system.preprocess_with_lea_learning(df)
    
  
    #test_session_events = ...
    
    session_id = 999999
    

    print("\n🎯 Getting Recommendations...")
    recommendations = system.get_hybrid_recommendations(session_id, test_session_events)
    print(f"Recommendations: {recommendations}")
    
  
    if recommendations:
        clicked_item = recommendations[0]
        new_event = {'aid': clicked_item, 'ts': 1661724040000, 'type': 'clicks'}
        updated_events = test_session_events + [new_event]
        
     
        system.update_with_hybrid_feedback(
            session_id, clicked_item, 'clicks', updated_events
        )
        print(f"✓ System updated with user feedback on item {clicked_item}")
    
   
    print("\n📊 Session Insights:")
    insights = system.get_session_insights(session_id, test_session_events)
    if insights:
        print(f"Purchase Likelihood: {insights['session_summary']['purchase_likelihood']:.3f}")
        print(f"Session Quality: {insights['session_summary']['session_quality']:.3f}")
        print(f"Exploration Score: {insights['behavior_analysis']['exploration_score']:.3f}")
        print(f"Consideration Score: {insights['behavior_analysis']['consideration_score']:.3f}")
    
    
    print("\n📈 System Analytics:")
    analytics = system.get_comprehensive_analytics()
    print(f"Total Items: {analytics['basic_stats']['total_items']}")
    print(f"LEA Patterns: {analytics['lea_metrics']['user_patterns']}")
    print(f"Neuroplastic Connections: {analytics['neuroplasticity_metrics']['synaptic_connections']}")
    print(f"DQN Action Space: {analytics['dqn_metrics']['action_space_size']}")
    
    
    system.save_system("trained_hybrid_system.joblib")
    print("\n💾 System saved successfully!")
    
    print("\n✅ Basic usage example completed!")


def create_synthetic_data():
    """Create synthetic data for demonstration"""
    synthetic_sessions = []
    
    for session_id in range(100):
        events = []
       
        num_events = np.random.randint(3, 15)
        base_time = 1661724000000
        
        for i in range(num_events):
            aid = np.random.randint(1000, 10000)
            event_type = np.random.choice(['clicks', 'carts', 'orders'], 
                                        p=[0.7, 0.2, 0.1])
            timestamp = base_time + i * 10000
            
            events.append({
                'aid': aid,
                'type': event_type,
                'ts': timestamp
            })
        
        synthetic_sessions.append({
            'session': session_id,
            'events': str(events)
        })
    
    return pd.DataFrame(synthetic_sessions)


if __name__ == "__main__":
    import numpy as np
    main()
