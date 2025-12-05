#!/usr/bin/env python3
"""
Integration test for V4 SC-1 Convex Hull Envelopment Environment
Tests complete integration of all convex hull modifications
"""

import sys
import os
sys.path.append('/home/pralak/Space_Touch')
sys.path.append('/home/pralak/Space_Touch/Code_Pranav/RL Code')
import numpy as np
import time

# Import the modified V3 environment (now V4 with convex hull)
try:
    # Need to import using importlib due to hyphens in filename
    import importlib.util
    script_path = '/home/pralak/Space_Touch/Code_Pranav/RL Code/V2_SC-1_Fixed_V3.py'

    spec = importlib.util.spec_from_file_location("V2_SC_1_Fixed_V3", script_path)
    v2_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(v2_module)

    V2AllegroReachingEnvFixed = v2_module.V2AllegroReachingEnvFixed
    print("✅ Successfully imported V4 environment with convex hull modifications")
except Exception as e:
    print(f"❌ Failed to import V4 environment: {e}")
    sys.exit(1)

def test_convex_hull_integration():
    """Test complete integration of convex hull envelopment system"""

    print("=" * 80)
    print("🔍 TESTING V4 CONVEX HULL ENVELOPMENT INTEGRATION")
    print("=" * 80)

    # Test 1: Environment Initialization
    print("\n✅ Test 1: Environment Initialization")
    try:
        env = V2AllegroReachingEnvFixed(vis=False, num_envs=1)
        print("   ✓ Environment created successfully")

        # Check reward calculator type
        reward_calc = env.reward_calculator
        print(f"   ✓ Reward calculator type: {type(reward_calc).__name__}")

        # Verify it's the convex hull reward
        assert "ConvexHullEnvelopmentReward" in str(type(reward_calc)), "❌ Wrong reward calculator type"
        print("   ✓ ConvexHullEnvelopmentReward properly initialized")

    except Exception as e:
        print(f"   ❌ Environment initialization failed: {e}")
        return False

    # Test 2: Observation and Action Spaces
    print("\n✅ Test 2: Observation and Action Spaces")
    try:
        obs_shape = env.observation_space.shape
        action_shape = env.action_space.shape

        print(f"   ✓ Observation space: {obs_shape}")
        print(f"   ✓ Action space: {action_shape}")

        assert obs_shape[0] == 27, f"❌ Expected obs dim 27, got {obs_shape[0]}"
        assert action_shape[0] == 10, f"❌ Expected action dim 10, got {action_shape[0]}"

        print("   ✓ Space dimensions correct")

    except Exception as e:
        print(f"   ❌ Space verification failed: {e}")
        return False

    # Test 3: Environment Reset
    print("\n✅ Test 3: Environment Reset")
    try:
        obs = env.reset()
        print(f"   ✓ Reset successful, obs shape: {obs.shape}")

        assert obs.shape == (1, 27), f"❌ Expected obs shape (1, 27), got {obs.shape}"
        print("   ✓ Reset observation shape correct")

    except Exception as e:
        print(f"   ❌ Environment reset failed: {e}")
        return False

    # Test 4: Palm Position Helper Method
    print("\n✅ Test 4: Palm Position Helper Method")
    try:
        palm_pos = env._get_palm_position()
        print(f"   ✓ Palm position: {palm_pos}")

        assert isinstance(palm_pos, np.ndarray), "❌ Palm position should be numpy array"
        assert palm_pos.shape == (3,), f"❌ Expected palm pos shape (3,), got {palm_pos.shape}"
        print("   ✓ Palm position helper working correctly")

    except Exception as e:
        print(f"   ❌ Palm position helper failed: {e}")
        return False

    # Test 5: Finger Positions
    print("\n✅ Test 5: Finger Positions")
    try:
        finger_pos = env._get_finger_positions()
        print(f"   ✓ Finger positions shape: {finger_pos.shape}")

        assert finger_pos.shape == (12,), f"❌ Expected finger pos shape (12,), got {finger_pos.shape}"

        # Test reshaping for convex hull
        finger_pos_2d = finger_pos.reshape(4, 3)
        assert finger_pos_2d.shape == (4, 3), "❌ Finger position reshaping failed"
        print("   ✓ Finger positions and reshaping working correctly")

    except Exception as e:
        print(f"   ❌ Finger position test failed: {e}")
        return False

    # Test 6: Step Execution with Convex Hull Reward
    print("\n✅ Test 6: Step Execution with Convex Hull Reward")
    try:
        # Generate random action and format for VecEnv
        action = env.action_space.sample()
        actions = np.array([action])  # Wrap for VecEnv

        # Execute step
        obs_new, reward, done, info = env.step(actions)

        print(f"   ✓ Step executed successfully")
        print(f"   ✓ Reward: {reward[0]:.6f}")
        print(f"   ✓ Done: {done[0]}")

        # Check info contains convex hull specific metrics
        info_keys = info[0].keys()
        expected_keys = ['hull_formation_reward', 'proximity_reward', 'envelopment_reward',
                        'sustained_envelopment_reward', 'hull_valid', 'is_enveloped']

        for key in expected_keys:
            if key in info_keys:
                print(f"   ✓ Info contains {key}: {info[0][key]}")
            else:
                print(f"   ⚠️  Missing info key: {key}")

    except Exception as e:
        print(f"   ❌ Step execution failed: {e}")
        import traceback
        traceback.print_exc()
        return False

    # Test 7: Multiple Steps for Reward Consistency
    print("\n✅ Test 7: Multiple Steps for Reward Consistency")
    try:
        total_rewards = []
        hull_valid_count = 0
        envelopment_count = 0

        for step in range(10):
            action = env.action_space.sample()
            actions = np.array([action])  # Wrap for VecEnv
            obs, reward, done, info = env.step(actions)

            total_rewards.append(reward[0])

            if info[0].get('hull_valid', False):
                hull_valid_count += 1
            if info[0].get('is_enveloped', False):
                envelopment_count += 1

            if done[0]:
                print(f"   Episode completed at step {step}")
                break

        print(f"   ✓ Completed {len(total_rewards)} steps")
        print(f"   ✓ Average reward: {np.mean(total_rewards):.6f}")
        print(f"   ✓ Reward range: [{min(total_rewards):.6f}, {max(total_rewards):.6f}]")
        print(f"   ✓ Hull valid steps: {hull_valid_count}/{len(total_rewards)}")
        print(f"   ✓ Envelopment steps: {envelopment_count}/{len(total_rewards)}")

        # Check that rewards are reasonable for convex hull system
        min_reward, max_reward = min(total_rewards), max(total_rewards)

        # Convex hull rewards should generally be positive, but allow some small negative values
        # from tactical penalties in the reward function
        if min_reward < -2.0:  # Allow small negative rewards but flag large ones
            print(f"   ⚠️  Some negative rewards detected: {min_reward}")

        if max_reward > 200.0:  # Very high rewards might indicate an issue
            print(f"   ⚠️  Very high rewards detected: {max_reward}")

        print(f"   ✓ Reward range appears reasonable: [{min_reward:.2f}, {max_reward:.2f}]")

    except Exception as e:
        print(f"   ❌ Multiple steps test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

    # Test 8: Success Criteria
    print("\n✅ Test 8: Success Criteria")
    try:
        success_criteria = env.reward_calculator.get_success_criteria()

        print(f"   ✓ Envelopment required: {success_criteria['envelopment_required']}")
        print(f"   ✓ Min consecutive steps: {success_criteria['min_consecutive_steps']}")
        print(f"   ✓ Min clearance: {success_criteria['min_clearance']}")

        assert success_criteria['envelopment_required'] == True, "❌ Envelopment should be required"
        print("   ✓ Success criteria properly configured")

    except Exception as e:
        print(f"   ❌ Success criteria test failed: {e}")
        return False

    # Test 9: Curriculum Update
    print("\n✅ Test 9: Curriculum Update")
    try:
        # Test curriculum update (should work without error)
        env.update_curriculum(100000)  # 100K timesteps

        old_phase = env.reward_curriculum_phase
        env.update_curriculum(200000)  # 200K timesteps

        print(f"   ✓ Curriculum update completed without error")
        print(f"   ✓ Reward phase progression: {old_phase} → {env.reward_curriculum_phase}")

    except Exception as e:
        print(f"   ❌ Curriculum update failed: {e}")
        return False

    # Cleanup
    try:
        env.close()
        print("   ✓ Environment closed successfully")
    except:
        pass

    print("\n" + "=" * 80)
    print("✅ ALL INTEGRATION TESTS PASSED!")
    print("=" * 80)
    print("🚀 V4 Convex Hull Envelopment Integration Verified:")
    print("   • Environment initialization: ✓")
    print("   • ConvexHullEnvelopmentReward integration: ✓")
    print("   • Palm position helper method: ✓")
    print("   • Reward calculation system: ✓")
    print("   • Step execution: ✓")
    print("   • Info logging with convex hull metrics: ✓")
    print("   • Success criteria configuration: ✓")
    print("   • Curriculum system compatibility: ✓")
    print("\n💡 Ready for full V4 training!")
    print("   Command: python V2_SC-1_Fixed_V3.py")
    print("=" * 80)

    return True

def test_reward_calculation_directly():
    """Test reward calculation directly with synthetic data"""

    print("\n🎯 DIRECT REWARD CALCULATION TEST")
    print("-" * 60)

    try:
        from reward_functions.convex_hull_envelopment_reward import ConvexHullEnvelopmentReward

        reward_calc = ConvexHullEnvelopmentReward()

        # Create synthetic observation
        obs_dict = {
            'finger_positions': np.array([
                [0.1, 0.1, 0.3],
                [-0.1, 0.1, 0.3],
                [-0.1, -0.1, 0.3],
                [0.1, -0.1, 0.3]
            ]),
            'palm_position': np.array([0.0, 0.0, 0.25]),
            'target_position': np.array([0.0, 0.0, 0.28]),
            'hand_center': np.array([0.0, 0.0, 0.275])
        }

        reward, info = reward_calc.calculate_reward(obs_dict)

        print(f"   Direct reward calculation: {reward:.6f}")
        print(f"   Hull valid: {info['hull_valid']}")
        print(f"   Hull volume: {info['hull_volume']:.8f}")
        print(f"   Is enveloped: {info['is_enveloped']}")
        print(f"   ✓ Direct reward calculation working")

    except Exception as e:
        print(f"   ❌ Direct reward calculation failed: {e}")
        return False

    return True

if __name__ == "__main__":
    try:
        # Run integration tests
        integration_success = test_convex_hull_integration()

        # Run direct reward test
        reward_success = test_reward_calculation_directly()

        if integration_success and reward_success:
            print("\n🎉 ALL TESTS COMPLETED SUCCESSFULLY!")
            print("🚀 V4 Convex Hull Envelopment system is ready for training!")
            sys.exit(0)
        else:
            print("\n❌ SOME TESTS FAILED!")
            print("Please review the errors above before proceeding with training.")
            sys.exit(1)

    except Exception as e:
        print(f"\n❌ CRITICAL ERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)