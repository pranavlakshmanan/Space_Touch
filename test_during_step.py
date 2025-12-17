#!/usr/bin/env python3
"""
Debug: What happens DURING a step?
"""
import numpy as np
import pybullet as p
import sys
sys.path.append("/home/ubuntu/workspace/Space_Touch/Code_Pranav/RL Code")

from v7_6_sc1 import V76Environment

def test_during():
    print("="*60)
    print("DURING STEP DEBUG")
    print("="*60)
    
    env = V76Environment(vis=False, max_steps=100)
    env.reset()
    
    action = np.array([[0.0, 0, 0, 0, 0, 0, 0, 0, 0, 0]])
    
    print("\nStep-by-step breakdown of one env.step():\n")
    
    # Get state BEFORE
    palm_before = env._get_palm_position()
    base_before, _ = p.getBasePositionAndOrientation(env.hand_id)
    print(f"1. BEFORE _apply_action:")
    print(f"   Base: [{base_before[0]:.3f}, {base_before[1]:.3f}, {base_before[2]:.3f}]")
    print(f"   Palm: [{palm_before[0]:.3f}, {palm_before[1]:.3f}, {palm_before[2]:.3f}]")
    
    # Apply action (manually call the internal method)
    env._apply_action(action[0])
    
    palm_after_action = env._get_palm_position()
    base_after_action, _ = p.getBasePositionAndOrientation(env.hand_id)
    print(f"\n2. AFTER _apply_action, BEFORE stepSimulation:")
    print(f"   Base: [{base_after_action[0]:.3f}, {base_after_action[1]:.3f}, {base_after_action[2]:.3f}]")
    print(f"   Palm: [{palm_after_action[0]:.3f}, {palm_after_action[1]:.3f}, {palm_after_action[2]:.3f}]")
    
    # Step simulation
    p.stepSimulation()
    
    palm_after_sim = env._get_palm_position()
    base_after_sim, _ = p.getBasePositionAndOrientation(env.hand_id)
    print(f"\n3. AFTER stepSimulation:")
    print(f"   Base: [{base_after_sim[0]:.3f}, {base_after_sim[1]:.3f}, {base_after_sim[2]:.3f}]")
    print(f"   Palm: [{palm_after_sim[0]:.3f}, {palm_after_sim[1]:.3f}, {palm_after_sim[2]:.3f}]")
    
    # Check finger positions
    print(f"\n4. Finger positions after step:")
    fingers = env._get_finger_positions()
    for i, f in enumerate(fingers):
        print(f"   Finger {i}: [{f[0]:.3f}, {f[1]:.3f}, {f[2]:.3f}]")
    
    # Check finger bases
    print(f"\n5. Finger BASE positions after step:")
    bases = env._get_finger_base_positions()
    for i, b in enumerate(bases):
        print(f"   Base {i}: [{b[0]:.3f}, {b[1]:.3f}, {b[2]:.3f}]")
    
    # Palm is calculated from finger bases
    calculated_palm = np.mean(bases, axis=0)
    print(f"\n6. Palm = mean(finger_bases):")
    print(f"   Calculated: [{calculated_palm[0]:.3f}, {calculated_palm[1]:.3f}, {calculated_palm[2]:.3f}]")
    print(f"   From _get_palm_position: [{palm_after_sim[0]:.3f}, {palm_after_sim[1]:.3f}, {palm_after_sim[2]:.3f}]")
    
    env.close()

if __name__ == "__main__":
    test_during()
