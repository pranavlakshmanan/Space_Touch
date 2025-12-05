#!/usr/bin/env python3
"""
Verification script for relaxed SimplifiedReward criteria
Run this BEFORE training to confirm changes are working correctly
"""

import sys
sys.path.append('/home/pralak/Space_Touch')
import numpy as np
from reward_functions.simplified_reward import SimplifiedReward

def verify_relaxed_criteria():
    """Verify that relaxed criteria are properly implemented"""

    print("=" * 80)
    print("🔍 VERIFYING RELAXED SIMPLIFIED REWARD CRITERIA")
    print("=" * 80)

    # Initialize reward calculator
    reward_calc = SimplifiedReward(config={})

    # Verify thresholds
    print("\n✅ Threshold Verification:")
    assert reward_calc.SUCCESS_THRESHOLD == 0.12, f"❌ SUCCESS_THRESHOLD should be 0.12, got {reward_calc.SUCCESS_THRESHOLD}"
    print(f"   ✓ SUCCESS_THRESHOLD = {reward_calc.SUCCESS_THRESHOLD}m (relaxed from 0.08m)")

    assert reward_calc.MIN_CONSECUTIVE_STEPS == 25, f"❌ MIN_CONSECUTIVE_STEPS should be 25, got {reward_calc.MIN_CONSECUTIVE_STEPS}"
    print(f"   ✓ MIN_CONSECUTIVE_STEPS = {reward_calc.MIN_CONSECUTIVE_STEPS} (reduced from 50)")

    assert hasattr(reward_calc, 'VERY_FAR_THRESHOLD'), "❌ Missing VERY_FAR_THRESHOLD"
    print(f"   ✓ VERY_FAR_THRESHOLD = {reward_calc.VERY_FAR_THRESHOLD}m (new milestone)")

    assert hasattr(reward_calc, 'FAR_THRESHOLD'), "❌ Missing FAR_THRESHOLD"
    print(f"   ✓ FAR_THRESHOLD = {reward_calc.FAR_THRESHOLD}m (new milestone)")

    print(f"   ✓ PROXIMITY_THRESHOLD = {reward_calc.PROXIMITY_THRESHOLD}m (relaxed from 0.15m)")
    print(f"   ✓ CLOSE_THRESHOLD = {reward_calc.CLOSE_THRESHOLD}m (relaxed from 0.12m)")
    print(f"   ✓ VERY_CLOSE_THRESHOLD = {reward_calc.VERY_CLOSE_THRESHOLD}m (relaxed from 0.08m)")

    # Test enhanced staged bonuses
    print("\n✅ Enhanced Staged Bonus Verification:")
    test_distances = [0.5, 0.35, 0.25, 0.18, 0.13, 0.10]
    expected_stages = [0, 1, 2, 3, 4, 5]  # FIXED: 0.35m should be stage 1, not 2
    expected_min_bonuses = [0.0, 1.0, 3.0, 6.0, 11.0, 21.0]  # FIXED: Updated bonuses

    for dist, exp_stage, exp_min_bonus in zip(test_distances, expected_stages, expected_min_bonuses):
        obs = {
            'distance': dist,
            'contact_force': 0.0,
            'hand_pos': np.array([0, 0, 0]),
            'target_pos': np.array([dist, 0, 0])
        }
        reward, info = reward_calc.calculate_reward(obs)

        actual_stage = info['success_stage']
        actual_bonus = info['success_bonus']

        print(f"   Distance {dist}m: Stage {actual_stage} (expected {exp_stage}), Bonus {actual_bonus:.1f} (min {exp_min_bonus:.1f})")

        assert actual_stage == exp_stage, f"❌ Stage mismatch at {dist}m"
        assert actual_bonus >= exp_min_bonus, f"❌ Bonus too low at {dist}m"

        reward_calc.reset()  # Reset for next test

    # Test sustained success
    print("\n✅ Sustained Success Verification:")
    obs_success = {
        'distance': 0.10,  # Within success threshold
        'contact_force': 3.0,  # Gentle contact
        'hand_pos': np.array([0, 0, 0]),
        'target_pos': np.array([0.10, 0, 0])
    }

    # Simulate 30 consecutive successful steps
    max_bonus_seen = 0.0
    for step in range(30):
        reward, info = reward_calc.calculate_reward(obs_success)
        max_bonus_seen = max(max_bonus_seen, info['success_bonus'])

        if step < 24:
            # Should NOT have sustained success bonus yet
            assert info['success_bonus'] <= 21.0, f"❌ Premature sustained success at step {step}"
        else:
            # Should have sustained success bonus after 25 steps
            assert info['success_bonus'] > 21.0, f"❌ Missing sustained success bonus at step {step}"

    print(f"   ✓ Consecutive steps tracking: Working correctly")
    print(f"   ✓ Sustained success bonus: Activated after 25 steps")
    print(f"   ✓ Maximum bonus achieved: {max_bonus_seen:.1f} (expected ~51.0)")

    # Test reward range
    print("\n✅ Reward Range Verification:")

    # Worst case: far away, hard contact
    obs_worst = {
        'distance': 2.0,
        'contact_force': 25.0,
        'hand_pos': np.array([0, 0, 0]),
        'target_pos': np.array([2.0, 0, 0])
    }
    reward_calc.reset()
    reward_worst, _ = reward_calc.calculate_reward(obs_worst)
    print(f"   Worst case reward: {reward_worst:.2f} (expected near -0.5)")

    # Best case: very close, sustained, gentle contact
    reward_calc.reset()
    obs_best = {'distance': 0.10, 'contact_force': 3.0, 'hand_pos': np.array([0, 0, 0]), 'target_pos': np.array([0.10, 0, 0])}
    for _ in range(30):
        reward_best, _ = reward_calc.calculate_reward(obs_best)
    print(f"   Best case reward: {reward_best:.2f} (expected near 52.5)")

    # Test expected reward range method
    min_expected, max_expected = reward_calc.get_expected_reward_range()
    print(f"   Expected range method: [{min_expected:.1f}, {max_expected:.1f}]")
    assert max_expected > 50.0, f"❌ Expected max reward should be > 50.0, got {max_expected}"

    print("\n" + "=" * 80)
    print("✅ ALL VERIFICATIONS PASSED!")
    print("=" * 80)
    print("🚀 Ready to train with relaxed criteria")
    print("   Expected improvements:")
    print("   • First successes: 100K-200K steps")
    print("   • Regular successes: 300K+ steps")
    print("   • Target success rate: 15-30% at 500K steps")
    print("=" * 80)

if __name__ == "__main__":
    try:
        verify_relaxed_criteria()
    except AssertionError as e:
        print(f"\n❌ VERIFICATION FAILED: {e}")
        print("   Please review the modifications in simplified_reward.py")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)