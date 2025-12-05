#!/usr/bin/env python3
"""
Quick test to verify hand reset positioning fix
"""

import numpy as np
import sys
sys.path.append('/home/pralak/Space_Touch')
sys.path.append('/home/pralak/Space_Touch/Code_Pranav/RL Code')

from V5_ConvexHull_Overlap_Training import ConvexHullOverlapEnv

def test_hand_reset_positioning():
    """Test that hand resets close to target"""

    print("🔍 Testing Hand Reset Positioning Fix...")

    # Create environment
    env = ConvexHullOverlapEnv(num_envs=1, vis=False, max_steps=100)

    print("✅ Environment created")

    failures = 0

    for i in range(10):
        print(f"\n🧪 Reset Test {i+1}/10:")

        # Reset environment
        obs = env.reset()

        # Extract positions
        hand_pos = obs[0, :3]  # VecEnv format [batch, obs]
        target_pos = obs[0, 3:6]
        distance = np.linalg.norm(hand_pos - target_pos)

        print(f"   Hand:   [{hand_pos[0]:.3f}, {hand_pos[1]:.3f}, {hand_pos[2]:.3f}]")
        print(f"   Target: [{target_pos[0]:.3f}, {target_pos[1]:.3f}, {target_pos[2]:.3f}]")
        print(f"   Distance: {distance:.3f}m ({distance*1000:.1f}mm)")

        # Check if positioning is correct
        if distance > 0.15:
            print(f"   ❌ TOO FAR! Should be <150mm for overlap learning")
            failures += 1
        elif distance > 0.10:
            print(f"   ⚠️  BORDERLINE: Within range but could be closer")
        else:
            print(f"   ✅ EXCELLENT: Close enough for overlap detection")

        # Check if hand is on correct side of world
        if hand_pos[0] < 0 and target_pos[0] > 0:
            print(f"   🚨 CRITICAL: Hand on wrong side of world! (negative X vs positive X)")
            failures += 1
        elif abs(hand_pos[0] - target_pos[0]) > 0.2:
            print(f"   🚨 CRITICAL: X coordinates too far apart!")
            failures += 1

    env.close()

    # Summary
    print(f"\n{'='*60}")
    print(f"🎯 HAND RESET TEST RESULTS:")
    print(f"{'='*60}")
    print(f"   Tests run: 10")
    print(f"   Failures: {failures}")
    print(f"   Success rate: {(10-failures)/10*100:.0f}%")

    if failures == 0:
        print(f"\n🎉 ALL TESTS PASSED!")
        print(f"   ✅ Hand consistently spawns close to target")
        print(f"   ✅ No positioning failures detected")
        print(f"   ✅ Ready to restart training!")
        return True
    elif failures <= 2:
        print(f"\n⚠️  MOSTLY SUCCESSFUL ({failures} minor issues)")
        print(f"   ✅ Should work much better than before")
        print(f"   ✅ Acceptable to restart training")
        return True
    else:
        print(f"\n❌ TOO MANY FAILURES ({failures}/10)")
        print(f"   ❌ Hand reset fix not working properly")
        print(f"   ❌ DO NOT restart training yet - debug further")
        return False

if __name__ == "__main__":
    success = test_hand_reset_positioning()
    exit(0 if success else 1)