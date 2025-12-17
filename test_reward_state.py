#!/usr/bin/env python3
"""
Debug: Check if reward calculator state is corrupted
"""
import numpy as np
import sys
sys.path.append("/home/ubuntu/workspace/Space_Touch")

from reward_functions.v7_6_reward import V76RewardCalculator

def test_state():
    print("="*60)
    print("REWARD CALCULATOR STATE DEBUG")
    print("="*60)
    
    calc = V76RewardCalculator()
    object_pos = np.array([0.25, 0.15, 0.35])
    
    # Simulate what happens in the environment
    palm = np.array([0.20, 0.15, 0.40])  # ~7cm from target
    fingers = np.array([
        [0.22, 0.17, 0.42],
        [0.22, 0.13, 0.42],
        [0.22, 0.17, 0.38],
        [0.22, 0.13, 0.38],
    ])
    finger_bases = fingers - np.array([[0.02, 0, 0]] * 4)
    
    obs = {
        'finger_positions': fingers,
        'finger_bases': finger_bases,
        'palm_position': palm,
        'object_pos': object_pos,
        'binary_contact': np.array([0,0,0,0]),
    }
    
    # Call multiple times WITHOUT reset
    print("\n5 calls WITHOUT reset (same position):")
    for i in range(5):
        reward, info = calc.calculate_reward(obs)
        print(f"  Call {i+1}: prev_dist={calc.previous_distance:.4f}, "
              f"reported={info['distance_to_target']*100:.1f}cm, "
              f"dist_deriv={info['distance_derivative_reward']:+.2f}")
    
    # Now call reset and try again
    print("\n--- RESET ---")
    calc.reset()
    
    print("\n5 calls AFTER reset (same position):")
    for i in range(5):
        reward, info = calc.calculate_reward(obs)
        print(f"  Call {i+1}: prev_dist={calc.previous_distance:.4f}, "
              f"reported={info['distance_to_target']*100:.1f}cm, "
              f"dist_deriv={info['distance_derivative_reward']:+.2f}")
    
    # Now simulate episode reset but forgot to call calc.reset()
    print("\n--- Simulating forgotten reset ---")
    # Move far away first
    palm_far = np.array([0.50, 0.15, 0.35])  # 25cm away
    obs_far = {
        'finger_positions': fingers + np.array([[0.30, 0, 0]] * 4),
        'finger_bases': finger_bases + np.array([[0.30, 0, 0]] * 4),
        'palm_position': palm_far,
        'object_pos': object_pos,
        'binary_contact': np.array([0,0,0,0]),
    }
    reward, info = calc.calculate_reward(obs_far)
    print(f"Far position: dist={info['distance_to_target']*100:.1f}cm, prev={calc.previous_distance:.4f}")
    
    # Now "new episode" starts close but NO reset called
    print("\nNew episode starts close (NO reset called):")
    reward, info = calc.calculate_reward(obs)  # Back to close position
    print(f"  Close position: reported={info['distance_to_target']*100:.1f}cm, "
          f"dist_deriv={info['distance_derivative_reward']:+.2f}")
    print(f"  EXPECTED: dist_deriv should be LARGE POSITIVE (moved from 25cm to 7cm)")

if __name__ == "__main__":
    test_state()
