#!/usr/bin/env python3
"""
Quick test script for V5 Convex Hull Overlap Training
Tests the environment and model creation without full training
"""

import sys
import numpy as np
sys.path.append('/home/pralak/Space_Touch')

# Set matplotlib backend for headless testing
import matplotlib
matplotlib.use('Agg')

def test_v5_environment():
    """Test the V5 environment creation and basic functionality"""

    print("🧪 Testing V5 Convex Hull Overlap Environment")

    try:
        # Import the fixed training components
        from reward_functions.convex_hull_overlap_reward import ConvexHullOverlapReward

        # Test reward function
        print("✅ Reward function import successful")

        reward_func = ConvexHullOverlapReward()
        print(f"✅ Reward function created: {reward_func}")

        # Test observation and reward calculation
        test_obs = {
            'finger_positions': np.array([[0.3, 0.2, 0.4], [0.3, 0.15, 0.4], [0.3, 0.1, 0.4], [0.25, 0.15, 0.4]]),
            'palm_position': np.array([0.25, 0.15, 0.35]),
            'object_pos': np.array([0.25, 0.15, 0.35]),
            'binary_contact': np.array([0, 0, 0, 0]),
            'episode_step': 100,
        }

        reward, info = reward_func.calculate_reward(test_obs)
        print(f"✅ Reward calculation successful: {reward:.3f}")

        # Now test the full environment (simplified version)
        print("\n🔧 Testing Environment Creation...")

        # Import environment class
        exec(open('/home/pralak/Space_Touch/Code_Pranav/RL Code/V5_ConvexHull_Overlap_Training.py').read())

        return True

    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_v5_environment()
    if success:
        print("\n✅ V5 Environment Test Passed!")
        print("Ready to run full training with:")
        print("   python 'Code_Pranav/RL Code/V5_ConvexHull_Overlap_Training.py'")
    else:
        print("\n❌ V5 Environment Test Failed!")
        print("Please check the errors above before running full training.")