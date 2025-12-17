#!/usr/bin/env python3
"""
Debug: Check actual hand base position vs palm calculation
"""
import numpy as np
import pybullet as p
import sys
sys.path.append("/home/ubuntu/workspace/Space_Touch/Code_Pranav/RL Code")

from v7_6_sc1 import V76Environment

def test_positions():
    print("="*60)
    print("POSITION DEBUG TEST")
    print("="*60)
    
    env = V76Environment(vis=False, max_steps=100)
    env.reset()
    
    action = np.array([[0.0, 0, 0, 0, 0, 0, 0, 0, 0, 0]])
    
    print("\nTaking 10 steps with ZERO action:\n")
    
    for i in range(10):
        # Get hand base position BEFORE step
        base_pos_before, base_orn = p.getBasePositionAndOrientation(env.hand_id)
        
        obs, reward, done, info = env.step(action)
        
        # Get hand base position AFTER step
        base_pos_after, _ = p.getBasePositionAndOrientation(env.hand_id)
        
        # Get palm position from environment
        palm_pos = env._get_palm_position()
        
        # Get target position
        target = env.target_pos
        
        ri = info[0].get('reward_info', {})
        reported_dist = ri.get('palm_to_target_distance', 0)
        
        # Calculate actual distance
        actual_dist = np.linalg.norm(palm_pos - target)
        base_to_target = np.linalg.norm(np.array(base_pos_after) - target)
        
        print(f"Step {i+1}:")
        print(f"  Base pos BEFORE: [{base_pos_before[0]:.3f}, {base_pos_before[1]:.3f}, {base_pos_before[2]:.3f}]")
        print(f"  Base pos AFTER:  [{base_pos_after[0]:.3f}, {base_pos_after[1]:.3f}, {base_pos_after[2]:.3f}]")
        print(f"  Palm pos:        [{palm_pos[0]:.3f}, {palm_pos[1]:.3f}, {palm_pos[2]:.3f}]")
        print(f"  Target:          [{target[0]:.3f}, {target[1]:.3f}, {target[2]:.3f}]")
        print(f"  Base-to-target:  {base_to_target*100:.1f}cm")
        print(f"  Palm-to-target:  {actual_dist*100:.1f}cm")
        print(f"  Reported dist:   {reported_dist*100:.1f}cm")
        print()
    
    env.close()

if __name__ == "__main__":
    test_positions()
