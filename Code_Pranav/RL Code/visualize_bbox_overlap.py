#!/usr/bin/env python3
"""
Visualize Bounding Box Overlap Computation
Shows exactly how V7 computes hand-object overlap using AABB method
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection


def generate_sphere_points(center, radius, n_points=32):
    """Generate points on sphere using Fibonacci lattice (same as V7)"""
    points = []
    phi = np.pi * (3.0 - np.sqrt(5.0))  # Golden angle

    for i in range(n_points):
        y = 1 - (i / float(n_points - 1)) * 2
        r = np.sqrt(1 - y * y)
        theta = phi * i

        x = np.cos(theta) * r
        z = np.sin(theta) * r
        points.append([x * radius, y * radius, z * radius])

    return np.array(points) + center


def generate_hand_points(base_pos, finger_extension=0.10, spread=0.04):
    """Generate realistic 9-point hand configuration

    Allegro hand specs:
    - Palm width: ~8-10cm
    - Finger length: ~10-12cm
    - Total span when spread: ~15-20cm
    """
    # 4 finger bases spread around palm (realistic Allegro hand spacing)
    finger_bases = np.array([
        [-spread, -spread, 0.0],   # Ring base (back-left)
        [-spread,  spread, 0.0],   # Middle base (back-right)
        [ spread, -spread, 0.0],   # Index base (front-left)
        [ spread,  spread, 0.0],   # Thumb base (front-right)
    ]) + base_pos

    # 4 fingertips extended from bases (realistic finger directions)
    finger_tips = finger_bases + np.array([
        [finger_extension, -0.02, 0.02],         # Ring: forward + slight outward
        [finger_extension,  0.02, 0.03],         # Middle: forward + slight outward + up
        [finger_extension, -0.02, 0.04],         # Index: forward + slight outward + more up
        [finger_extension * 0.8, 0.03, 0.06],    # Thumb: shorter + outward + highest
    ])

    # Palm center (between finger bases)
    palm = np.mean(finger_bases, axis=0).reshape(1, 3)

    # Combine: 4 tips + 4 bases + 1 palm = 9 points
    return np.vstack([finger_tips, finger_bases, palm])


def compute_bbox(points):
    """Compute axis-aligned bounding box (same as V7)"""
    mins = np.min(points, axis=0)
    maxs = np.max(points, axis=0)
    return mins, maxs


def compute_bbox_volume(mins, maxs, correction_factor=0.5):
    """Compute bounding box volume with correction (same as V7)"""
    dims = maxs - mins
    volume = np.prod(dims) * correction_factor
    return volume


def compute_intersection(bbox1_min, bbox1_max, bbox2_min, bbox2_max):
    """Compute bounding box intersection (same as V7)"""
    inter_min = np.maximum(bbox1_min, bbox2_min)
    inter_max = np.minimum(bbox1_max, bbox2_max)

    # Check if there's any overlap
    if np.any(inter_min >= inter_max):
        return None, None, 0.0

    # Compute overlap volume with correction factor
    overlap_volume = np.prod(inter_max - inter_min) * 0.3

    return inter_min, inter_max, overlap_volume


def draw_bbox(ax, mins, maxs, color='blue', alpha=0.2, label='', linewidth=2):
    """Draw a 3D bounding box"""
    # Define the 8 corners of the box
    corners = np.array([
        [mins[0], mins[1], mins[2]],
        [maxs[0], mins[1], mins[2]],
        [maxs[0], maxs[1], mins[2]],
        [mins[0], maxs[1], mins[2]],
        [mins[0], mins[1], maxs[2]],
        [maxs[0], mins[1], maxs[2]],
        [maxs[0], maxs[1], maxs[2]],
        [mins[0], maxs[1], maxs[2]],
    ])

    # Define the 6 faces
    faces = [
        [corners[0], corners[1], corners[2], corners[3]],  # Bottom
        [corners[4], corners[5], corners[6], corners[7]],  # Top
        [corners[0], corners[1], corners[5], corners[4]],  # Front
        [corners[2], corners[3], corners[7], corners[6]],  # Back
        [corners[0], corners[3], corners[7], corners[4]],  # Left
        [corners[1], corners[2], corners[6], corners[5]],  # Right
    ]

    # Draw faces
    face_collection = Poly3DCollection(faces, alpha=alpha, facecolor=color,
                                      edgecolor='black', linewidth=linewidth)
    ax.add_collection3d(face_collection)

    # Add label at center
    center = (mins + maxs) / 2
    ax.text(center[0], center[1], center[2], label, fontsize=10,
           color='black', weight='bold', ha='center')


def visualize_scenario(hand_pos, object_pos, object_radius, title, ax):
    """Visualize a specific hand-object configuration"""

    # Generate points
    hand_points = generate_hand_points(hand_pos)
    object_points = generate_sphere_points(object_pos, object_radius, n_points=32)

    # Compute bounding boxes
    hand_min, hand_max = compute_bbox(hand_points)
    obj_min, obj_max = compute_bbox(object_points)

    # Compute volumes
    hand_vol = compute_bbox_volume(hand_min, hand_max, 0.5)
    obj_vol = compute_bbox_volume(obj_min, obj_max, 0.5)

    # Compute intersection
    inter_min, inter_max, overlap_vol = compute_intersection(
        hand_min, hand_max, obj_min, obj_max
    )

    # Draw points
    ax.scatter(hand_points[:, 0], hand_points[:, 1], hand_points[:, 2],
              c='blue', s=80, marker='o', label='Hand points (9)', alpha=0.8, edgecolor='black')
    ax.scatter(object_points[:, 0], object_points[:, 1], object_points[:, 2],
              c='red', s=30, marker='^', label='Object points (32)', alpha=0.6)

    # Draw bounding boxes
    draw_bbox(ax, hand_min, hand_max, color='blue', alpha=0.15,
             label=f'Hand\n{hand_vol*1e6:.0f}cm³')
    draw_bbox(ax, obj_min, obj_max, color='red', alpha=0.15,
             label=f'Object\n{obj_vol*1e6:.0f}cm³')

    # Draw intersection if exists
    if inter_min is not None:
        draw_bbox(ax, inter_min, inter_max, color='green', alpha=0.4,
                 label=f'Overlap\n{overlap_vol*1e6:.1f}cm³', linewidth=3)

    # Formatting
    ax.set_xlabel('X (m)', fontsize=10)
    ax.set_ylabel('Y (m)', fontsize=10)
    ax.set_zlabel('Z (m)', fontsize=10)
    ax.set_title(title, fontsize=12, weight='bold')
    ax.legend(loc='upper left', fontsize=8)

    # Set equal aspect ratio
    max_range = 0.15
    mid = (hand_pos + object_pos) / 2
    ax.set_xlim(mid[0] - max_range, mid[0] + max_range)
    ax.set_ylim(mid[1] - max_range, mid[1] + max_range)
    ax.set_zlim(mid[2] - max_range, mid[2] + max_range)

    # Add grid
    ax.grid(True, alpha=0.3)

    # Add stats text
    stats_text = f"Hand: {hand_vol*1e6:.0f} cm³\nObject: {obj_vol*1e6:.0f} cm³\nOverlap: {overlap_vol*1e6:.1f} cm³"
    if overlap_vol > 0:
        overlap_pct = (overlap_vol / min(hand_vol, obj_vol)) * 100
        stats_text += f"\nOverlap: {overlap_pct:.1f}%"
    ax.text2D(0.02, 0.98, stats_text, transform=ax.transAxes,
             fontsize=9, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))


def main():
    """Generate comprehensive visualization of bounding box overlap"""

    print("="*70)
    print("V7 BOUNDING BOX OVERLAP VISUALIZATION")
    print("="*70)
    print()

    # Object parameters (same as V7)
    object_radius = 0.075  # 5cm radius + 2.5cm safety margin = 7.5cm
    object_pos = np.array([0.25, 0.15, 0.35])

    # Create figure with 4 scenarios
    fig = plt.figure(figsize=(18, 12))

    # Scenario 1: No Overlap (far away)
    print("Scenario 1: NO OVERLAP (Hand far from object)")
    ax1 = fig.add_subplot(2, 2, 1, projection='3d')
    hand_pos1 = object_pos + np.array([-0.15, 0.0, 0.0])  # 15cm away
    visualize_scenario(hand_pos1, object_pos, object_radius,
                      "Scenario 1: No Overlap (Distance > 10cm)", ax1)
    print()

    # Scenario 2: Small Overlap (just touching)
    print("Scenario 2: SMALL OVERLAP (Just touching)")
    ax2 = fig.add_subplot(2, 2, 2, projection='3d')
    hand_pos2 = object_pos + np.array([-0.08, 0.01, 0.02])  # 8cm away, slight offset
    visualize_scenario(hand_pos2, object_pos, object_radius,
                      "Scenario 2: Small Overlap (~50-100 cm³)", ax2)
    print()

    # Scenario 3: Good Overlap (enveloping)
    print("Scenario 3: GOOD OVERLAP (Enveloping)")
    ax3 = fig.add_subplot(2, 2, 3, projection='3d')
    hand_pos3 = object_pos + np.array([-0.05, 0.0, 0.01])  # 5cm away, centered
    visualize_scenario(hand_pos3, object_pos, object_radius,
                      "Scenario 3: Good Overlap (~150-200 cm³)", ax3)
    print()

    # Scenario 4: Maximum Overlap (centered)
    print("Scenario 4: MAXIMUM OVERLAP (Centered)")
    ax4 = fig.add_subplot(2, 2, 4, projection='3d')
    hand_pos4 = object_pos + np.array([-0.03, 0.0, 0.0])  # 3cm away, perfectly centered
    visualize_scenario(hand_pos4, object_pos, object_radius,
                      "Scenario 4: Maximum Overlap (~250+ cm³)", ax4)
    print()

    plt.suptitle('V7 Bounding Box Overlap Computation Visualization\n' +
                'Blue = Hand (9 points), Red = Object (32 points), Green = Overlap',
                fontsize=14, weight='bold', y=0.98)

    plt.tight_layout(rect=[0, 0, 1, 0.96])

    # Save figure
    output_path = "bbox_overlap_visualization.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"✓ Visualization saved to: {output_path}")
    print()

    plt.close()

    print()
    print("="*70)
    print("KEY INSIGHTS:")
    print("="*70)
    print("1. Bounding Box = Axis-aligned rectangular prism containing all points")
    print("2. Hand volume × 0.5 correction (hand is concave, not solid box)")
    print("3. Object volume × 0.5 correction (sphere inscribed in box ≈ 52%)")
    print("4. Overlap volume × 0.3 correction (intersection overestimates more)")
    print("5. Fast computation: Just min/max operations, no ConvexHull needed!")
    print()
    print("SIZE COMPARISON:")
    print("  - Object: 10cm diameter sphere (5cm radius)")
    print("  - Object + safety margin: 15cm diameter (7.5cm radius)")
    print("  - Hand span: ~15-20cm when fingers spread")
    print("  - Hand SHOULD be bigger/similar to object (can envelop it)")
    print()
    print("Compare to your training:")
    print(f"  - Object volume: ~1622 cm³ (constant, matches your WandB!)")
    print(f"  - Hand volume: 200-1200 cm³ (variable based on finger position)")
    print(f"    * Compact: ~400-600 cm³ (fingers closed)")
    print(f"    * Spread:  ~800-1200 cm³ (fingers open wide)")
    print(f"  - Overlap: 0-250 cm³ (matches your training range!)")
    print()


if __name__ == "__main__":
    main()
