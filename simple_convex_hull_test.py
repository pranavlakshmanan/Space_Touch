#!/usr/bin/env python3
"""
Simple test to isolate the convex hull integration issue
"""

import sys
import importlib.util
import numpy as np

# Set up path
sys.path.append('/home/pralak/Space_Touch')

def simple_test():
    """Simple test focusing on a single step"""

    print("🔍 SIMPLE CONVEX HULL INTEGRATION TEST")
    print("=" * 50)

    try:
        # Import environment
        script_path = '/home/pralak/Space_Touch/Code_Pranav/RL Code/V2_SC-1_Fixed_V3.py'
        spec = importlib.util.spec_from_file_location("V2_SC_1_Fixed_V3", script_path)
        v2_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(v2_module)
        V2AllegroReachingEnvFixed = v2_module.V2AllegroReachingEnvFixed

        print("✅ Environment imported")

        # Create environment
        env = V2AllegroReachingEnvFixed(vis=False, num_envs=1)
        print("✅ Environment created")

        # Reset
        obs = env.reset()
        print(f"✅ Environment reset - obs shape: {obs.shape}")

        # Test action - VecEnv expects array of actions for each environment
        action = np.array([0.1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.5, 0.5, 0.5, 0.5])  # Simple action
        actions = np.array([action])  # Wrap in array for VecEnv (1 environment)

        print(f"🎯 Testing step with action: {action}")
        print(f"   VecEnv actions shape: {actions.shape}")

        # Execute step
        obs_new, reward, done, info = env.step(actions)

        print(f"📊 Results:")
        print(f"   Reward: {reward}")
        print(f"   Done: {done}")
        print(f"   Info keys: {list(info[0].keys())}")

        # Check if convex hull info is present
        expected_keys = ['hull_formation_reward', 'proximity_reward', 'envelopment_reward',
                        'sustained_envelopment_reward', 'hull_valid', 'is_enveloped']

        missing_keys = []
        present_keys = []
        for key in expected_keys:
            if key in info[0]:
                present_keys.append(key)
            else:
                missing_keys.append(key)

        if present_keys:
            print(f"   ✅ Present convex hull keys: {present_keys}")
            for key in present_keys:
                print(f"      {key}: {info[0][key]}")

        if missing_keys:
            print(f"   ❌ Missing convex hull keys: {missing_keys}")

        # Check if reward is positive (expected for convex hull system)
        if isinstance(reward, np.ndarray):
            reward_val = reward[0]
        else:
            reward_val = reward

        if reward_val >= 0.0:
            print("   ✅ Reward is non-negative (expected for convex hull system)")
            return True
        else:
            print(f"   ⚠️  Reward is negative: {reward_val}")
            if 'error' in info[0]:
                print(f"   Error info: {info[0]['error']}")
            return False

    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = simple_test()
    if success:
        print("\n🎉 Simple test completed - basic integration working!")
        print("   Issue may be in test assertions rather than core functionality")
    else:
        print("\n❌ Simple test failed - core integration issue exists")
        print("   Need to debug further")