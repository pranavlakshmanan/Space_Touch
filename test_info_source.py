#!/usr/bin/env python3
"""
Debug: Where is the wrong distance coming from?
"""
import numpy as np
import pybullet as p
import sys
sys.path.append("/home/ubuntu/workspace/Space_Touch/Code_Pranav/RL Code")

from v7_6_sc1 import V76Environment

def test_info():
    print("="*60)
    print("INFO SOURCE DEBUG")
    print("="*60)
    
    env = V76Environment(vis=False, max_steps=100)
    env.reset()
    
    action = np.array([[0.0, 0, 0, 0, 0, 0, 0, 0, 0, 0]])
    
    for i in range(5):
        # Manually calculate palm position and distance
        palm_pos = env._get_palm_position()
        target = env.target_pos
        manual_dist = np.linalg.norm(palm_pos - target)
        
        # Step
        obs, reward, done, info = env.step(action)
        
        # Get info from step
        ri = info[0].get('reward_info', {})
        reported_palm_dist = ri.get('palm_to_target_distance', -1)
        reported_dist_to_target = ri.get('distance_to_target', -1)
        
        print(f"\nStep {i+1}:")
        print(f"  Manual palm dist:      {manual_dist*100:.1f}cm")
        print(f"  palm_to_target_dist:   {reported_palm_dist*100:.1f}cm")
        print(f"  distance_to_target:    {reported_dist_to_target*100:.1f}cm")
        print(f"  done={done[0]}")
        
        if done[0]:
            print("  >>> EPISODE TERMINATED - will reset next step")
    
    env.close()

if __name__ == "__main__":
    test_info()
