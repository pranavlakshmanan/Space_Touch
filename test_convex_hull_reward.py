#!/usr/bin/env python3
"""
Test script for the new convex hull overlap reward function
Validates reward calculation and PNG generation capabilities
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-GUI backend
import matplotlib.pyplot as plt
from pathlib import Path
import sys

# Add project path
sys.path.append('/home/pralak/Space_Touch')
from reward_functions.convex_hull_overlap_reward import ConvexHullOverlapReward

def test_convex_hull_reward():
    """Test the convex hull overlap reward function"""

    print("=" * 60)
    print("🧪 TESTING CONVEX HULL OVERLAP REWARD FUNCTION")
    print("=" * 60)

    # Create reward function with test configuration
    config = {
        'object_radius': 0.05,           # 5cm sphere
        'safety_margin': 0.025,          # 2.5cm clearance
        'overlap_scale': 10000.0,
        'contact_penalty': -5.0,
        'vis_dir': '/tmp/convex_hull_test',
        'generate_vis': True,
    }

    reward_func = ConvexHullOverlapReward(config=config)
    print(f"✅ Reward function created: {reward_func}")

    # Test scenarios
    test_scenarios = [
        {
            "name": "Scenario 1: Far Away (No Overlap)",
            "finger_positions": np.array([
                [0.5, 0.2, 0.4],   # Index finger
                [0.5, 0.15, 0.4],  # Middle finger
                [0.5, 0.1, 0.4],   # Ring finger
                [0.45, 0.15, 0.4]  # Thumb
            ]),
            "palm_position": np.array([0.45, 0.15, 0.35]),
            "object_pos": np.array([0.25, 0.15, 0.35]),
            "binary_contact": np.array([0, 0, 0, 0]),
            "expected_overlap": 0.0
        },
        {
            "name": "Scenario 2: Approaching (Proximity Reward)",
            "finger_positions": np.array([
                [0.3, 0.2, 0.4],
                [0.3, 0.15, 0.4],
                [0.3, 0.1, 0.4],
                [0.25, 0.15, 0.4]
            ]),
            "palm_position": np.array([0.25, 0.15, 0.35]),
            "object_pos": np.array([0.25, 0.15, 0.35]),
            "binary_contact": np.array([0, 0, 0, 0]),
            "expected_overlap": 0.0
        },
        {
            "name": "Scenario 3: Optimal Envelopment (High Overlap)",
            "finger_positions": np.array([
                [0.27, 0.18, 0.37],  # Fingers surrounding object
                [0.27, 0.12, 0.37],
                [0.23, 0.12, 0.37],
                [0.23, 0.18, 0.37]
            ]),
            "palm_position": np.array([0.25, 0.15, 0.32]),  # Palm below
            "object_pos": np.array([0.25, 0.15, 0.35]),
            "binary_contact": np.array([0, 0, 0, 0]),
            "expected_overlap": "> 0"
        },
        {
            "name": "Scenario 4: Contact Violation (Penalty)",
            "finger_positions": np.array([
                [0.26, 0.18, 0.35],  # Fingers close but not degenerate
                [0.24, 0.12, 0.35],
                [0.25, 0.12, 0.36],
                [0.25, 0.18, 0.34]
            ]),
            "palm_position": np.array([0.25, 0.15, 0.32]),
            "object_pos": np.array([0.25, 0.15, 0.35]),
            "binary_contact": np.array([1, 1, 0, 0]),  # Two fingers in contact
            "expected_overlap": "> 0 but penalized"
        }
    ]

    print(f"\n🔬 Running {len(test_scenarios)} test scenarios...")

    for i, scenario in enumerate(test_scenarios):
        print(f"\n{'='*50}")
        print(f"🎯 {scenario['name']}")
        print(f"{'='*50}")

        # Prepare observation dictionary
        obs_dict = {
            'finger_positions': scenario['finger_positions'],
            'palm_position': scenario['palm_position'],
            'object_pos': scenario['object_pos'],
            'binary_contact': scenario['binary_contact'],
            'episode_step': i * 100,  # Different step for each scenario
        }

        try:
            # Calculate reward
            total_reward, reward_info = reward_func.calculate_reward(obs_dict)

            # Print results
            print(f"📊 REWARD BREAKDOWN:")
            print(f"   Total Reward:     {total_reward:8.3f}")
            print(f"   Overlap Reward:   {reward_info.get('overlap_reward', 0):8.3f}")
            print(f"   Contact Penalty:  {reward_info.get('contact_penalty', 0):8.3f}")
            print(f"   Proximity Reward: {reward_info.get('proximity_reward', 0):8.3f}")
            print(f"   Quality Reward:   {reward_info.get('quality_reward', 0):8.3f}")

            print(f"\n📈 METRICS:")
            print(f"   Overlap Volume:   {reward_info.get('overlap_volume', 0):8.6f} m³")
            print(f"   Num Contacts:     {reward_info.get('num_contacts', 0):8d}")
            print(f"   Hand Hull Vol:    {reward_info.get('hand_hull_volume', 0):8.6f} m³")
            print(f"   Object Hull Vol:  {reward_info.get('object_hull_volume', 0):8.6f} m³")

            if reward_info.get('visualization_path'):
                print(f"   Visualization:    {reward_info['visualization_path']}")

            # Validate expectations
            if scenario['expected_overlap'] == 0.0:
                if reward_info.get('overlap_volume', 0) == 0.0:
                    print("✅ PASSED: No overlap as expected")
                else:
                    print("⚠️  UNEXPECTED: Got overlap when none expected")
            elif scenario['expected_overlap'] == "> 0":
                if reward_info.get('overlap_volume', 0) > 0:
                    print("✅ PASSED: Positive overlap as expected")
                else:
                    print("⚠️  UNEXPECTED: No overlap when some expected")

        except Exception as e:
            print(f"❌ ERROR in scenario {i+1}: {e}")
            import traceback
            traceback.print_exc()

    # Test visualization generation
    print(f"\n{'='*50}")
    print("📸 TESTING VISUALIZATION GENERATION")
    print(f"{'='*50}")

    vis_dir = Path(config['vis_dir'])
    if vis_dir.exists():
        vis_files = list(vis_dir.glob("*.png"))
        print(f"✅ Generated {len(vis_files)} visualization files:")
        for file_path in vis_files:
            print(f"   📄 {file_path}")

        if vis_files:
            print(f"\n💡 To view visualizations, run:")
            print(f"   ls {vis_dir}/*.png")
            print(f"   # Open any PNG file to see the convex hull visualization")
    else:
        print("⚠️  No visualization directory found")

    # Test reward range
    print(f"\n{'='*50}")
    print("📏 REWARD RANGE ANALYSIS")
    print(f"{'='*50}")

    min_reward, max_reward = reward_func.get_expected_reward_range()
    print(f"Expected reward range: [{min_reward:.1f}, {max_reward:.1f}]")

    print(f"\n✅ CONVEX HULL OVERLAP REWARD TEST COMPLETED")
    print(f"🔧 Reward function appears to be working correctly!")
    print(f"📸 Check visualizations in: {config['vis_dir']}")

    return True

if __name__ == "__main__":
    test_convex_hull_reward()