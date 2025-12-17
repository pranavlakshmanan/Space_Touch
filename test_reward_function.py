#!/usr/bin/env python3
"""
V7.6 Reward Function Validation Script
"""

import numpy as np
import sys
import os

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from reward_functions.v7_6_reward import V76RewardCalculator


def create_observation(finger_positions, palm_position, object_pos, contacts=[0,0,0,0]):
    finger_bases = finger_positions - np.array([[0.02, 0, 0]] * 4)
    return {
        'finger_positions': finger_positions,
        'finger_bases': finger_bases,
        'palm_position': palm_position,
        'object_pos': object_pos,
        'binary_contact': np.array(contacts),
    }


def test_critical():
    print("\n" + "="*60)
    print("CRITICAL TEST: Approach vs Retreat")
    print("="*60)
    
    object_pos = np.array([0.0, 0.0, 0.0])
    
    # APPROACH: 15cm -> 4cm
    calc = V76RewardCalculator()
    approach_rewards = []
    print("\nAPPROACHING (15cm -> 4cm):")
    for dist in [0.15, 0.12, 0.09, 0.06, 0.04]:
        palm = np.array([dist, 0.0, 0.0])
        fingers = np.array([[dist+0.02, 0.03, 0.03], [dist+0.02, -0.03, 0.03],
                           [dist+0.02, 0.03, -0.03], [dist+0.02, -0.03, -0.03]])
        obs = create_observation(fingers, palm, object_pos)
        reward, info = calc.calculate_reward(obs)
        approach_rewards.append(reward)
        print(f"  Dist={dist*100:5.1f}cm | Reward={reward:+8.2f} | "
              f"dist_deriv={info['distance_derivative_reward']:+7.2f} | "
              f"overlap_deriv={info['overlap_derivative_reward']:+7.2f} | "
              f"sustain={info['sustain_bonus']:.1f}")
    
    # RETREAT: 4cm -> 15cm
    calc2 = V76RewardCalculator()
    retreat_rewards = []
    print("\nRETREATING (4cm -> 15cm):")
    for dist in [0.04, 0.06, 0.09, 0.12, 0.15]:
        palm = np.array([dist, 0.0, 0.0])
        fingers = np.array([[dist+0.02, 0.03, 0.03], [dist+0.02, -0.03, 0.03],
                           [dist+0.02, 0.03, -0.03], [dist+0.02, -0.03, -0.03]])
        obs = create_observation(fingers, palm, object_pos)
        reward, info = calc2.calculate_reward(obs)
        retreat_rewards.append(reward)
        print(f"  Dist={dist*100:5.1f}cm | Reward={reward:+8.2f} | "
              f"dist_deriv={info['distance_derivative_reward']:+7.2f} | "
              f"overlap_deriv={info['overlap_derivative_reward']:+7.2f} | "
              f"sustain={info['sustain_bonus']:.1f}")
    
    approach_total = sum(approach_rewards)
    retreat_total = sum(retreat_rewards)
    
    print("\n" + "="*60)
    print(f"APPROACH total: {approach_total:+.2f}")
    print(f"RETREAT total:  {retreat_total:+.2f}")
    print(f"DIFFERENCE:     {approach_total - retreat_total:+.2f}")
    print("="*60)
    
    if approach_total > retreat_total:
        print("✓ PASS: Approaching beats retreating")
    else:
        print("✗ FAIL: Retreating is equal or better than approaching!")
        print("  THIS IS THE BUG - agent learns to retreat")


if __name__ == "__main__":
    test_critical()
