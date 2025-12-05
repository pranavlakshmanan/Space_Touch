#!/usr/bin/env python3
"""
Quick test script to verify finger position synchronization with hand base
"""

import numpy as np
import sys
import pybullet as p
import os

# Add the correct paths
sys.path.append('/home/pralak/Space_Touch')
sys.path.append('/home/pralak/Space_Touch/Code_Pranav/RL Code')

# Import the environment
from V5_ConvexHull_Overlap_Training import ConvexHullOverlapEnv

def test_finger_synchronization():
    """Test that fingers move with the hand base"""

    print("🔍 Testing Finger-Hand Synchronization...")

    # Create environment (no visualization for faster testing)
    env = ConvexHullOverlapEnv(num_envs=1, vis=False, max_steps=100)

    print(f"✅ Environment created")

    # Reset and get initial state
    obs = env.reset()
    print(f"✅ Environment reset, observation shape: {obs.shape}")

    # Extract initial positions
    base_pos_initial = obs[0, :3]  # VecEnv format [batch, obs]
    finger_positions_initial = obs[0, 12:24].reshape(4, 3)

    print(f"\n📊 Initial State:")
    print(f"   Hand base: {base_pos_initial}")
    print(f"   Finger positions:")
    for i, fp in enumerate(finger_positions_initial):
        dist = np.linalg.norm(fp - base_pos_initial)
        print(f"     Finger {i}: {fp} (distance from base: {dist:.3f}m)")

    initial_distances = [np.linalg.norm(fp - base_pos_initial) for fp in finger_positions_initial]
    avg_initial_distance = np.mean(initial_distances)
    print(f"   Average finger-base distance: {avg_initial_distance:.3f}m")

    # Test 1: Apply movement action and check if fingers follow
    print(f"\n🧪 Test 1: Hand Movement Synchronization")

    # Apply a movement action (move hand in +X direction)
    movement_action = np.zeros(10)
    movement_action[0] = 1.0  # Move in +X direction

    print(f"   Applying action: {movement_action[:6]} (base movement)")

    # Step environment
    obs, reward, done, info = env.step([movement_action])

    # Extract new positions
    base_pos_new = obs[0, :3]
    finger_positions_new = obs[0, 12:24].reshape(4, 3)

    print(f"\n📊 After Movement:")
    print(f"   Hand base moved from {base_pos_initial} to {base_pos_new}")

    base_movement = np.linalg.norm(base_pos_new - base_pos_initial)
    print(f"   Base movement distance: {base_movement:.6f}m")

    # CRITICAL CHECK: Is movement now meaningful?
    if base_movement > 0.008:  # More than 8mm (close to 10mm target)
        print(f"   🎯 EXCELLENT: Base moved {base_movement*1000:.1f}mm (direct position control working!)")
    elif base_movement > 0.005:  # 5-8mm
        print(f"   ✅ GOOD: Base moved {base_movement*1000:.1f}mm (getting close to 10mm target)")
    elif base_movement > 0.001:  # 1-5mm
        print(f"   ⚠️  SLOW: Base moved {base_movement*1000:.1f}mm (still too slow)")
    else:
        print(f"   ❌ BAD: Base moved {base_movement*1000:.1f}mm (barely moving)")

    print(f"   New finger positions:")
    new_distances = []
    for i, fp in enumerate(finger_positions_new):
        dist = np.linalg.norm(fp - base_pos_new)
        new_distances.append(dist)
        old_fp = finger_positions_initial[i]
        finger_movement = np.linalg.norm(fp - old_fp)
        print(f"     Finger {i}: {fp}")
        print(f"       Distance from new base: {dist:.3f}m")
        print(f"       Finger movement: {finger_movement:.6f}m ({finger_movement*1000:.1f}mm)")

    avg_new_distance = np.mean(new_distances)
    print(f"   Average finger-base distance: {avg_new_distance:.3f}m")

    # Analysis
    print(f"\n🔍 Synchronization Analysis:")

    # Check 1: Did fingers maintain relative distance to base?
    distance_change = abs(avg_new_distance - avg_initial_distance)
    print(f"   Distance consistency: {distance_change:.6f}m change (should be <0.01m)")

    if distance_change < 0.01:
        print("   ✅ PASS: Fingers maintained relative distance to base")
        sync_test_1 = True
    else:
        print("   ❌ FAIL: Fingers did not maintain relative distance to base")
        sync_test_1 = False

    # Check 2: Did base actually move with DIRECT POSITION CONTROL speed?
    if base_movement > 0.008:  # More than 8mm (close to 10mm target)
        print(f"   🎯 EXCELLENT PASS: Base moved at direct position control speed ({base_movement:.6f}m = {base_movement*1000:.1f}mm)")
        movement_test = True
    elif base_movement > 0.005:  # 5-8mm
        print(f"   ✅ PASS: Base moved at good speed ({base_movement:.6f}m = {base_movement*1000:.1f}mm)")
        movement_test = True  # Accept this as passing now
    elif base_movement > 0.001:  # 1-5mm
        print(f"   ⚠️  BORDERLINE: Base moved slowly ({base_movement:.6f}m = {base_movement*1000:.1f}mm)")
        movement_test = False  # Still fail - too slow for effective learning
    else:
        print(f"   ❌ FAIL: Base barely moved ({base_movement:.6f}m = {base_movement*1000:.1f}mm)")
        movement_test = False

    # Check 3: Are finger movements consistent with base movement?
    finger_movements = [np.linalg.norm(finger_positions_new[i] - finger_positions_initial[i])
                       for i in range(4)]
    avg_finger_movement = np.mean(finger_movements)
    movement_ratio = abs(avg_finger_movement - base_movement) / max(base_movement, 1e-6)

    print(f"   Base movement: {base_movement:.6f}m")
    print(f"   Average finger movement: {avg_finger_movement:.6f}m")
    print(f"   Movement ratio difference: {movement_ratio:.3f} (should be <0.5)")

    if movement_ratio < 0.5:
        print("   ✅ PASS: Finger movements consistent with base movement")
        sync_test_2 = True
    else:
        print("   ❌ FAIL: Finger movements inconsistent with base movement")
        sync_test_2 = False

    # Test 2: Multiple steps to check continued synchronization
    print(f"\n🧪 Test 2: Continued Synchronization (10 steps)")

    desync_detected = False
    for step in range(10):
        # Random small movements
        action = np.random.uniform(-0.5, 0.5, 10)
        obs, reward, done, info = env.step([action])

        base_pos = obs[0, :3]
        finger_positions = obs[0, 12:24].reshape(4, 3)

        distances = [np.linalg.norm(fp - base_pos) for fp in finger_positions]
        max_distance = max(distances)

        if max_distance > 0.5:  # More than 50cm is suspicious
            print(f"   Step {step+1}: ⚠️  Large finger-base distance detected: {max_distance:.3f}m")
            desync_detected = True
        elif step % 3 == 0:  # Log every 3rd step
            avg_dist = np.mean(distances)
            print(f"   Step {step+1}: Average finger-base distance: {avg_dist:.3f}m")

    if not desync_detected:
        print("   ✅ PASS: No desynchronization detected in 10 steps")
        sync_test_3 = True
    else:
        print("   ❌ FAIL: Desynchronization detected during multi-step test")
        sync_test_3 = False

    # Final verdict
    print(f"\n🎯 Final Test Results:")
    print(f"   Distance Consistency: {'✅ PASS' if sync_test_1 else '❌ FAIL'}")
    print(f"   Base Movement: {'✅ PASS' if movement_test else '❌ FAIL'}")
    print(f"   Movement Synchronization: {'✅ PASS' if sync_test_2 else '❌ FAIL'}")
    print(f"   Continued Synchronization: {'✅ PASS' if sync_test_3 else '❌ FAIL'}")

    all_pass = sync_test_1 and movement_test and sync_test_2 and sync_test_3

    if all_pass:
        print(f"\n🎉 ALL TESTS PASSED: Finger synchronization is working!")
        print(f"   The finger desync bug has been fixed!")
        print(f"   You should now see non-zero overlap volumes during training.")
    else:
        print(f"\n🚨 TESTS FAILED: Finger synchronization still has issues!")
        print(f"   The finger desync bug persists.")
        print(f"   Need to investigate further:")

        if not sync_test_1:
            print(f"     - Fingers not maintaining relative distance to base")
        if not movement_test:
            print(f"     - Base not moving (action scaling too low?)")
        if not sync_test_2:
            print(f"     - Finger movements not matching base movements")
        if not sync_test_3:
            print(f"     - Desync occurring during multi-step simulation")

    env.close()
    return all_pass

if __name__ == "__main__":
    success = test_finger_synchronization()
    exit(0 if success else 1)