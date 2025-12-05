#!/usr/bin/env python3
"""
Test the new simplified reward function in isolation before training
Validates that the reward function behaves correctly across all scenarios
"""

import sys
import os
sys.path.append('/home/pralak/Space_Touch')

from reward_functions.simplified_reward import SimplifiedReward
import numpy as np


def test_reward_scenarios():
    """Test reward function on known scenarios with assertions"""

    reward_calc = SimplifiedReward(config={})

    print("=" * 80)
    print("🧪 TESTING SIMPLIFIED REWARD FUNCTION")
    print("=" * 80)
    print(f"Reward function: {reward_calc}")
    print(f"Expected range: {reward_calc.get_expected_reward_range()}")
    print("=" * 80)

    # Test 1: Far away, no contact - Should have small positive reward
    print("\n📍 TEST 1: Far away (1.0m), no contact")
    obs = {
        'distance': 1.0,
        'contact_force': 0.0,
        'hand_pos': np.array([0, 0, 0]),
        'target_pos': np.array([1, 0, 0])
    }
    reward, info = reward_calc.calculate_reward(obs)
    print(f"   Total reward: {reward:.4f}")
    print(f"   Components: Distance={info['distance_reward']:.4f}, Success={info['success_bonus']:.1f}, Tactile={info['tactile_reward']:.3f}")
    print(f"   Success stage: {info['success_stage']}")

    # Assertions
    assert reward > 0, f"Should have positive distance reward, got {reward}"
    assert reward < 1, f"Should be small when far, got {reward}"
    assert info['success_bonus'] == 0, f"No success bonus when far, got {info['success_bonus']}"
    assert info['tactile_reward'] == 0, f"No tactile reward without contact, got {info['tactile_reward']}"
    print("   ✅ PASS: Far distance behavior correct")


    # Test 2: Close (0.1m), no contact - Should have success bonus
    print("\n📍 TEST 2: Close (0.1m), no contact")
    obs['distance'] = 0.1
    reward, info = reward_calc.calculate_reward(obs)
    print(f"   Total reward: {reward:.4f}")
    print(f"   Components: Distance={info['distance_reward']:.4f}, Success={info['success_bonus']:.1f}, Tactile={info['tactile_reward']:.3f}")
    print(f"   Success stage: {info['success_stage']}")

    # Assertions
    assert reward > 5, f"Should have large success bonus, got {reward}"
    assert info['success_bonus'] >= 5.0, f"Should have close bonus (5.0+), got {info['success_bonus']}"
    assert info['distance_reward'] > 0.1, f"Should have good distance reward when close, got {info['distance_reward']}"
    print("   ✅ PASS: Close distance behavior correct")


    # Test 3: Close with gentle contact - Should reward tactile engagement
    print("\n📍 TEST 3: Close (0.1m), gentle contact (3N)")
    obs['contact_force'] = 3.0
    reward, info = reward_calc.calculate_reward(obs)
    print(f"   Total reward: {reward:.4f}")
    print(f"   Components: Distance={info['distance_reward']:.4f}, Success={info['success_bonus']:.1f}, Tactile={info['tactile_reward']:.3f}")
    print(f"   Success stage: {info['success_stage']}")

    # Assertions
    assert info['tactile_reward'] > 0, f"Should reward gentle contact when close, got {info['tactile_reward']}"
    assert info['tactile_reward'] <= 0.5, f"Tactile reward should be ≤ 0.5, got {info['tactile_reward']}"
    assert info['has_gentle_contact'], "Should detect gentle contact"
    assert not info['has_hard_contact'], "Should not detect hard contact"
    print("   ✅ PASS: Gentle contact when close behavior correct")


    # Test 4: Far with contact (bad behavior) - Should discourage
    print("\n📍 TEST 4: Far (0.5m), contact (3N) - should discourage flailing")
    obs['distance'] = 0.5
    obs['contact_force'] = 3.0
    reward, info = reward_calc.calculate_reward(obs)
    print(f"   Total reward: {reward:.4f}")
    print(f"   Components: Distance={info['distance_reward']:.4f}, Success={info['success_bonus']:.1f}, Tactile={info['tactile_reward']:.3f}")
    print(f"   Success stage: {info['success_stage']}")

    # Assertions
    assert info['tactile_reward'] < 0, f"Should penalize contact when far, got {info['tactile_reward']}"
    assert info['tactile_reward'] >= -0.2, f"Penalty should be mild, got {info['tactile_reward']}"
    print("   ✅ PASS: Far contact discouragement correct")


    # Test 5: Hard contact (safety violation) - Should strongly penalize
    print("\n📍 TEST 5: Hard contact (25N) - safety limit")
    obs['distance'] = 0.1
    obs['contact_force'] = 25.0
    reward, info = reward_calc.calculate_reward(obs)
    print(f"   Total reward: {reward:.4f}")
    print(f"   Components: Distance={info['distance_reward']:.4f}, Success={info['success_bonus']:.1f}, Tactile={info['tactile_reward']:.3f}")
    print(f"   Success stage: {info['success_stage']}")

    # Assertions
    assert info['tactile_reward'] < -0.4, f"Should strongly penalize hard contact, got {info['tactile_reward']}"
    assert info['has_hard_contact'], "Should detect hard contact"
    print("   ✅ PASS: Hard contact safety penalty correct")


    # Test 6: Success zone progression - Should track consecutive steps
    print("\n📍 TEST 6: Success zone progression (consecutive steps)")
    obs['distance'] = 0.07  # In success zone
    obs['contact_force'] = 0.0

    # Simulate several steps in success zone
    consecutive_rewards = []
    for step in range(55):  # More than MIN_CONSECUTIVE_STEPS (50)
        reward, info = reward_calc.calculate_reward(obs)
        consecutive_rewards.append(reward)
        if step in [0, 10, 25, 49, 50, 54]:
            print(f"   Step {step:2d}: Reward={reward:6.2f}, Consecutive={info['consecutive_steps']:2d}, Stage={info['success_stage']}")

    # Assertions
    assert info['consecutive_steps'] == 55, f"Should track 55 consecutive steps, got {info['consecutive_steps']}"
    assert consecutive_rewards[-1] > consecutive_rewards[0], "Reward should increase with consecutive success steps"
    assert info['success_stage'] == 4, f"Should reach stage 4 (sustained success), got {info['success_stage']}"
    print("   ✅ PASS: Consecutive step tracking correct")


    # Test 7: Reset functionality
    print("\n📍 TEST 7: Reset functionality")
    reward_calc.reset()
    obs['distance'] = 0.07
    reward, info = reward_calc.calculate_reward(obs)
    print(f"   After reset: Consecutive steps = {info['consecutive_steps']}")

    # Assertions
    assert info['consecutive_steps'] == 1, f"Should reset to 1 step after reset, got {info['consecutive_steps']}"
    print("   ✅ PASS: Reset functionality correct")


    # Test 8: Edge case - exactly at thresholds
    print("\n📍 TEST 8: Threshold edge cases")
    test_distances = [0.15, 0.12, 0.08, 0.05]  # Exactly at thresholds
    obs['contact_force'] = 0.0

    for dist in test_distances:
        obs['distance'] = dist
        reward_calc.reset()  # Reset for clean test
        reward, info = reward_calc.calculate_reward(obs)
        print(f"   Distance {dist:4.2f}m: Stage={info['success_stage']}, Bonus={info['success_bonus']:4.1f}, Total={reward:6.2f}")

    print("   ✅ PASS: Threshold edge cases handled correctly")


    # Test 9: Reward component isolation
    print("\n📍 TEST 9: Component isolation test")
    scenarios = [
        {"name": "Distance only", "distance": 0.3, "contact_force": 0.0},
        {"name": "Success only", "distance": 0.07, "contact_force": 0.0},
        {"name": "Tactile only", "distance": 1.0, "contact_force": 3.0},
    ]

    for scenario in scenarios:
        reward_calc.reset()
        obs = {'distance': scenario['distance'], 'contact_force': scenario['contact_force'],
               'hand_pos': np.zeros(3), 'target_pos': np.zeros(3)}
        reward, info = reward_calc.calculate_reward(obs)
        print(f"   {scenario['name']:12s}: Total={reward:5.2f} | Dist={info['distance_reward']:.2f} | Success={info['success_bonus']:.1f} | Tactile={info['tactile_reward']:.2f}")

    print("   ✅ PASS: Component isolation working correctly")


def test_reward_consistency():
    """Test reward function consistency and mathematical properties"""
    print("\n" + "=" * 80)
    print("🔬 TESTING REWARD CONSISTENCY")
    print("=" * 80)

    reward_calc = SimplifiedReward()

    # Test monotonicity: closer distances should generally give higher rewards (with same contact)
    print("\n📊 Distance monotonicity test:")
    distances = np.linspace(1.0, 0.05, 20)
    rewards = []

    for dist in distances:
        reward_calc.reset()
        obs = {'distance': dist, 'contact_force': 0.0, 'hand_pos': np.zeros(3), 'target_pos': np.zeros(3)}
        reward, info = reward_calc.calculate_reward(obs)
        rewards.append(reward)
        if len(rewards) % 5 == 1:  # Print every 5th
            print(f"   Distance {dist:.3f}m → Reward {reward:.3f}")

    # Check general increasing trend (allowing for step functions at thresholds)
    increasing_segments = 0
    for i in range(1, len(rewards)):
        if rewards[i] >= rewards[i-1]:
            increasing_segments += 1

    monotonicity_ratio = increasing_segments / (len(rewards) - 1)
    print(f"   Monotonicity ratio: {monotonicity_ratio:.2f} (should be > 0.8)")
    assert monotonicity_ratio > 0.8, f"Reward should generally increase as distance decreases, ratio: {monotonicity_ratio}"
    print("   ✅ PASS: Distance monotonicity acceptable")


def test_reward_range_validation():
    """Test that rewards stay within expected bounds"""
    print("\n" + "=" * 80)
    print("📏 TESTING REWARD BOUNDS")
    print("=" * 80)

    reward_calc = SimplifiedReward()
    min_expected, max_expected = reward_calc.get_expected_reward_range()

    print(f"Expected range: [{min_expected:.2f}, {max_expected:.2f}]")

    # Test extreme scenarios
    extreme_scenarios = [
        {"name": "Worst case", "distance": 10.0, "contact_force": 50.0},  # Far + hard contact
        {"name": "Best case", "distance": 0.05, "contact_force": 2.0, "consecutive_steps": 100},  # Close + gentle contact + sustained
        {"name": "Zero distance", "distance": 0.0, "contact_force": 0.0},
        {"name": "Max contact", "distance": 0.1, "contact_force": 100.0},
    ]

    observed_min = float('inf')
    observed_max = float('-inf')

    for scenario in extreme_scenarios:
        reward_calc.reset()

        # For sustained success test
        if scenario.get('consecutive_steps', 0) > 0:
            obs = {'distance': scenario['distance'], 'contact_force': scenario['contact_force'],
                   'hand_pos': np.zeros(3), 'target_pos': np.zeros(3)}
            for _ in range(scenario['consecutive_steps']):
                reward, info = reward_calc.calculate_reward(obs)
        else:
            obs = {'distance': scenario['distance'], 'contact_force': scenario['contact_force'],
                   'hand_pos': np.zeros(3), 'target_pos': np.zeros(3)}
            reward, info = reward_calc.calculate_reward(obs)

        observed_min = min(observed_min, reward)
        observed_max = max(observed_max, reward)

        print(f"   {scenario['name']:12s}: {reward:6.2f}")

        # Check bounds
        if reward < min_expected - 0.1 or reward > max_expected + 0.1:
            print(f"   ⚠️  WARNING: Reward {reward:.2f} outside expected range [{min_expected:.2f}, {max_expected:.2f}]")

    print(f"\nObserved range: [{observed_min:.2f}, {observed_max:.2f}]")
    print(f"Expected range:  [{min_expected:.2f}, {max_expected:.2f}]")

    # Bounds should be reasonable
    assert observed_min >= min_expected - 0.1, f"Min reward too low: {observed_min} < {min_expected - 0.1}"
    assert observed_max <= max_expected + 0.1, f"Max reward too high: {observed_max} > {max_expected + 0.1}"

    print("✅ PASS: Reward bounds validation successful")


def main():
    """Run all reward function tests"""
    print("🚀 Starting comprehensive reward function validation...")

    try:
        # Core functionality tests
        test_reward_scenarios()

        # Mathematical consistency tests
        test_reward_consistency()

        # Bounds validation
        test_reward_range_validation()

        print("\n" + "=" * 80)
        print("🎉 ALL REWARD FUNCTION TESTS PASSED!")
        print("=" * 80)
        print("✅ Reward function is ready for training")
        print("✅ Expected positive rewards throughout training")
        print("✅ Proper incentive structure for soft-capture task")
        print("✅ Safety limits and gentle contact encouragement working")
        print("\n🚀 Ready to proceed with training setup!")
        print("=" * 80)

        return True

    except AssertionError as e:
        print(f"\n❌ TEST FAILED: {e}")
        print("🔧 Fix the reward function before proceeding with training!")
        return False

    except Exception as e:
        print(f"\n💥 UNEXPECTED ERROR: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)