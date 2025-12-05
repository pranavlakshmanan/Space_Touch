#!/usr/bin/env python3
"""
Debug script to test convex hull generation and identify issues
"""

import numpy as np
import sys
sys.path.append('/home/pralak/Space_Touch')
from reward_functions.convex_hull_envelopment_reward import ConvexHullEnvelopmentReward
from scipy.spatial import ConvexHull

def test_hull_generation():
    """Test hull generation with realistic hand positions"""

    print("🔍 Testing Convex Hull Generation...")

    # Initialize reward calculator
    reward_calc = ConvexHullEnvelopmentReward()

    # Test 1: Object hull generation
    print("\n📦 Testing Object Hull Generation:")
    object_pos = np.array([0.25, 0.15, 0.35])
    object_hull_points = reward_calc.generate_object_hull_points(object_pos)
    print(f"   Object hull points shape: {object_hull_points.shape}")
    print(f"   Object center: {object_pos}")
    print(f"   Hull points sample:\n{object_hull_points[:3]}")

    # Validate object hull
    obj_valid, obj_volume, obj_error = reward_calc.validate_hull(object_hull_points, "object")
    print(f"   Object hull valid: {obj_valid}")
    print(f"   Object hull volume: {obj_volume}")
    if obj_error:
        print(f"   Object hull error: {obj_error}")

    # Test 2: Hand hull generation (realistic positions)
    print("\n✋ Testing Hand Hull Generation:")

    # Simulate realistic finger positions around target
    base_pos = np.array([0.25, 0.15, 0.35])
    finger_positions = np.array([
        [0.27, 0.18, 0.37],  # Index finger
        [0.27, 0.12, 0.37],  # Middle finger
        [0.23, 0.12, 0.37],  # Ring finger
        [0.23, 0.18, 0.37],  # Thumb
    ])
    palm_position = np.array([0.25, 0.15, 0.32])  # Palm behind fingers

    print(f"   Finger positions shape: {finger_positions.shape}")
    print(f"   Palm position: {palm_position}")
    print(f"   Finger positions:\n{finger_positions}")

    # Create hand hull points
    hand_hull_points = np.vstack([finger_positions, palm_position.reshape(1, 3)])
    print(f"   Hand hull points shape: {hand_hull_points.shape}")

    # Validate hand hull
    hand_valid, hand_volume, hand_error = reward_calc.validate_hull(hand_hull_points, "hand")
    print(f"   Hand hull valid: {hand_valid}")
    print(f"   Hand hull volume: {hand_volume}")
    if hand_error:
        print(f"   Hand hull error: {hand_error}")

    # Test 3: Check for common issues
    print("\n🔍 Detailed Diagnostics:")

    # Check point uniqueness
    unique_points = np.unique(hand_hull_points, axis=0)
    print(f"   Unique points: {len(unique_points)}/{len(hand_hull_points)}")

    # Check distances between points
    distances = []
    for i in range(len(hand_hull_points)):
        for j in range(i+1, len(hand_hull_points)):
            dist = np.linalg.norm(hand_hull_points[i] - hand_hull_points[j])
            distances.append(dist)

    print(f"   Min distance between points: {min(distances):.6f}")
    print(f"   Max distance between points: {max(distances):.6f}")
    print(f"   Mean distance: {np.mean(distances):.6f}")

    # Check if points are roughly coplanar
    if len(hand_hull_points) >= 4:
        # Fit plane to first 3 points
        p0, p1, p2 = hand_hull_points[:3]
        v1 = p1 - p0
        v2 = p2 - p0
        normal = np.cross(v1, v2)
        normal = normal / (np.linalg.norm(normal) + 1e-10)

        # Check distance of remaining points to plane
        plane_distances = []
        for i in range(3, len(hand_hull_points)):
            v3 = hand_hull_points[i] - p0
            dist_to_plane = abs(np.dot(v3, normal))
            plane_distances.append(dist_to_plane)

        print(f"   Max distance from plane: {max(plane_distances):.6f}")
        print(f"   Coplanar threshold: 1e-6")
        print(f"   Points are coplanar: {max(plane_distances) <= 1e-6}")

    # Test 4: Try direct scipy ConvexHull
    print("\n🧮 Direct SciPy ConvexHull Test:")
    try:
        direct_hull = ConvexHull(hand_hull_points)
        print(f"   Direct hull volume: {direct_hull.volume}")
        print(f"   Direct hull area: {direct_hull.area}")
        print(f"   Number of vertices: {len(direct_hull.vertices)}")
        print(f"   Number of simplices: {len(direct_hull.simplices)}")
    except Exception as e:
        print(f"   Direct hull failed: {e}")

    # Test 5: Test with slightly perturbed points
    print("\n🎲 Testing with Perturbed Points:")
    perturbed_points = hand_hull_points + np.random.normal(0, 0.001, hand_hull_points.shape)
    perturb_valid, perturb_volume, perturb_error = reward_calc.validate_hull(perturbed_points, "perturbed")
    print(f"   Perturbed hull valid: {perturb_valid}")
    print(f"   Perturbed hull volume: {perturb_volume}")
    if perturb_error:
        print(f"   Perturbed hull error: {perturb_error}")

    # Test 6: Test reward calculation
    print("\n🎯 Testing Reward Calculation:")
    obs_dict = {
        'finger_positions': finger_positions,
        'palm_position': palm_position,
        'object_pos': object_pos,
        'binary_contact': np.array([0, 0, 0, 0]),
        'episode_step': 1,
    }

    try:
        total_reward, reward_info = reward_calc.calculate_reward(obs_dict)
        print(f"   Total reward: {total_reward}")
        print(f"   Reward info: {reward_info}")
    except Exception as e:
        print(f"   Reward calculation failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_hull_generation()