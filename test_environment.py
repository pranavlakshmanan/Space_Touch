#!/usr/bin/env python3
"""
Test the actual environment to see what's happening
"""
import numpy as np
import sys
sys.path.append("/home/ubuntu/workspace/Space_Touch/Code_Pranav/RL Code")

from v7_6_sc1 import V76Environment

def test_env():
    print("="*60)
    print("ENVIRONMENT TEST")
    print("="*60)
    
    env = V76Environment(vis=False, max_steps=100)
    obs = env.reset()
    
    print(f"\nInitial observation shape: {obs.shape}")
    
    # Get initial state by doing a dummy step
    dummy_action = np.array([[0.0, 0, 0, 0, 0, 0, 0, 0, 0, 0]])  # 2D array for VecEnv
    obs, reward, done, info = env.step(dummy_action)
    ri = info[0].get('reward_info', {})
    print(f"Initial palm distance: {ri.get('palm_to_target_distance', 0)*100:.1f}cm")
    print(f"Initial overlap: {ri.get('overlap_volume', 0)*1e6:.2f}cm³")
    
    # Take 10 steps moving "forward" in X (action[0] = +1)
    print("\n--- Taking 10 steps with action [+1,0,0,...] (forward X) ---")
    forward_action = np.array([[1.0, 0, 0, 0, 0, 0, 0, 0, 0, 0]])
    
    for i in range(10):
        obs, reward, done, info = env.step(forward_action)
        ri = info[0].get('reward_info', {})
        print(f"Step {i+1}: reward={reward[0]:+.2f}, "
              f"palm_dist={ri.get('palm_to_target_distance', 0)*100:.1f}cm, "
              f"dist_deriv={ri.get('distance_derivative_reward', 0):+.2f}")
    
    # Reset and try opposite direction
    print("\n--- Reset, taking 10 steps with action [-1,0,0,...] (backward X) ---")
    env.reset()
    # Dummy step to initialize
    env.step(np.array([[0.0, 0, 0, 0, 0, 0, 0, 0, 0, 0]]))
    
    backward_action = np.array([[-1.0, 0, 0, 0, 0, 0, 0, 0, 0, 0]])
    
    for i in range(10):
        obs, reward, done, info = env.step(backward_action)
        ri = info[0].get('reward_info', {})
        print(f"Step {i+1}: reward={reward[0]:+.2f}, "
              f"palm_dist={ri.get('palm_to_target_distance', 0)*100:.1f}cm, "
              f"dist_deriv={ri.get('distance_derivative_reward', 0):+.2f}")
    
    env.close()
    print("\n" + "="*60)

if __name__ == "__main__":
    test_env()
