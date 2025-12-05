#!/usr/bin/env python3
"""
Convex Hull Overlap Reward Function for SC-1 Space Manipulator
Revolutionary approach: Maximize overlap volume between hand and object convex hulls
while maintaining safe clearance (no contact).

Key Innovation:
- Hand Convex Hull: 4 fingertips + palm center (5 points)
- Object Convex Hull: Sphere with safety margin (8-12 points around sphere)
- Reward: Maximize intersection volume while penalizing contact
- Visualization: Real-time PNG generation of dual hull system

This approach unifies multiple sub-tasks:
1. Distance closing (hulls must approach to overlap)
2. Envelopment (hand must surround object)
3. Safety (no contact allowed)
4. Spatial awareness (3D containment strategy)
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from scipy.spatial import ConvexHull
from scipy.spatial.distance import cdist
import trimesh
from typing import Dict, Tuple, Optional
import os
from pathlib import Path


class ConvexHullOverlapReward:
    """
    Advanced reward function based on convex hull overlap maximization

    Reward Components:
    1. Hull Overlap Volume: Primary reward for spatial containment
    2. Contact Penalty: Strong penalty for touching object
    3. Hull Quality Bonus: Reward for well-formed hand configuration
    4. Approach Bonus: Encouragement for getting hulls close
    """

    def __init__(self, config: Dict = None):
        """Initialize convex hull overlap reward system"""
        if config is None:
            config = {}

        # Object properties (sphere)
        self.OBJECT_RADIUS = config.get('object_radius', 0.05)  # 5cm sphere
        self.SAFETY_MARGIN = config.get('safety_margin', 0.025)  # 2.5cm clearance
        self.OBJECT_HULL_RADIUS = self.OBJECT_RADIUS + self.SAFETY_MARGIN  # 7.5cm total

        # Hull generation parameters
        self.OBJECT_HULL_POINTS = config.get('object_hull_points', 12)  # Icosphere-like
        self.MIN_HAND_HULL_VOLUME = config.get('min_hand_hull_volume', 0.0001)  # 0.1cm³
        self.MAX_HAND_HULL_VOLUME = config.get('max_hand_hull_volume', 0.01)    # 10cm³

        # Reward scaling
        self.OVERLAP_SCALE = config.get('overlap_scale', 10000.0)  # Scale overlap volume to meaningful reward
        self.CONTACT_PENALTY = config.get('contact_penalty', -5.0)  # Strong penalty per contact
        self.APPROACH_SCALE = config.get('approach_scale', 2.0)     # Reward for hull proximity
        self.QUALITY_SCALE = config.get('quality_scale', 1.0)      # Hand shape quality reward

        # Visualization settings
        self.VIS_DIR = config.get('vis_dir', '/tmp/convex_hull_vis')
        self.GENERATE_VIS = config.get('generate_vis', True)
        self.VIS_COUNTER = 0

        # Create visualization directory
        if self.GENERATE_VIS:
            Path(self.VIS_DIR).mkdir(parents=True, exist_ok=True)

        print("🔧 Initialized Convex Hull Overlap Reward:")
        print(f"   Object radius: {self.OBJECT_RADIUS}m")
        print(f"   Safety margin: {self.SAFETY_MARGIN}m")
        print(f"   Total hull radius: {self.OBJECT_HULL_RADIUS}m")
        print(f"   Overlap scale: {self.OVERLAP_SCALE}")
        print(f"   Contact penalty: {self.CONTACT_PENALTY}")
        if self.GENERATE_VIS:
            print(f"   Visualization dir: {self.VIS_DIR}")

    def generate_object_hull_points(self, object_pos: np.ndarray) -> np.ndarray:
        """
        Generate convex hull points around spherical object with safety margin

        Args:
            object_pos: 3D position of object center

        Returns:
            hull_points: Array of 3D points forming convex hull around object
        """
        # Generate icosphere-like point distribution
        points = []

        # Method 1: Fibonacci sphere for uniform distribution
        n = self.OBJECT_HULL_POINTS
        golden_angle = np.pi * (3. - np.sqrt(5.))  # Golden angle in radians

        for i in range(n):
            # y goes from 1 to -1
            y = 1 - (i / float(n - 1)) * 2

            # radius at y
            radius_at_y = np.sqrt(1 - y * y)

            # golden angle increment
            theta = golden_angle * i

            x = np.cos(theta) * radius_at_y
            z = np.sin(theta) * radius_at_y

            # Scale by hull radius and translate to object position
            point = object_pos + self.OBJECT_HULL_RADIUS * np.array([x, y, z])
            points.append(point)

        return np.array(points)

    def generate_hand_hull_points(self, finger_positions: np.ndarray, palm_position: np.ndarray) -> np.ndarray:
        """
        Generate hand convex hull from finger positions and palm center

        Args:
            finger_positions: 4x3 array of fingertip positions [index, middle, ring, thumb]
            palm_position: 3D palm center position

        Returns:
            hull_points: 5x3 array of points for hand convex hull
        """
        # Combine finger tips with palm center
        hull_points = np.vstack([finger_positions, palm_position.reshape(1, 3)])
        return hull_points

    def calculate_hull_intersection_volume(self, hull1_points: np.ndarray, hull2_points: np.ndarray) -> float:
        """
        Calculate intersection volume between two convex hulls using trimesh

        Args:
            hull1_points: Points defining first convex hull
            hull2_points: Points defining second convex hull

        Returns:
            intersection_volume: Volume of intersection (0 if no overlap)
        """
        try:
            # Create convex hulls
            hull1 = ConvexHull(hull1_points)
            hull2 = ConvexHull(hull2_points)

            # Convert to trimesh objects for boolean operations
            mesh1 = trimesh.Trimesh(vertices=hull1_points[hull1.vertices],
                                   faces=hull1.simplices)
            mesh2 = trimesh.Trimesh(vertices=hull2_points[hull2.vertices],
                                   faces=hull2.simplices)

            # Make sure meshes are watertight
            if not mesh1.is_watertight:
                mesh1 = mesh1.convex_hull
            if not mesh2.is_watertight:
                mesh2 = mesh2.convex_hull

            # Calculate intersection
            try:
                intersection = mesh1.intersection(mesh2)
                if intersection.is_empty:
                    return 0.0
                return float(intersection.volume)
            except:
                # Fallback: approximate using distance-based overlap
                return self._approximate_overlap_volume(hull1_points, hull2_points)

        except Exception as e:
            # Fallback for degenerate hulls
            return 0.0

    def _approximate_overlap_volume(self, hull1_points: np.ndarray, hull2_points: np.ndarray) -> float:
        """
        Approximate overlap volume when exact calculation fails

        Uses distance-based heuristic:
        - Sample points inside first hull
        - Check how many are inside second hull
        - Estimate overlap volume
        """
        try:
            hull1 = ConvexHull(hull1_points)
            hull2 = ConvexHull(hull2_points)

            # Sample points in first hull's bounding box
            min_pt = np.min(hull1_points, axis=0)
            max_pt = np.max(hull1_points, axis=0)

            n_samples = 1000
            sample_points = np.random.uniform(min_pt, max_pt, (n_samples, 3))

            # Count points inside both hulls
            # This is a simplified approximation
            hull1_center = np.mean(hull1_points, axis=0)
            hull2_center = np.mean(hull2_points, axis=0)

            # Distance-based approximation
            dist_between_centers = np.linalg.norm(hull1_center - hull2_center)
            hull1_radius = np.max(np.linalg.norm(hull1_points - hull1_center, axis=1))
            hull2_radius = np.max(np.linalg.norm(hull2_points - hull2_center, axis=1))

            if dist_between_centers > (hull1_radius + hull2_radius):
                return 0.0  # No overlap

            # Rough overlap estimate based on sphere intersection
            overlap_ratio = max(0, 1 - dist_between_centers / (hull1_radius + hull2_radius))
            estimated_volume = overlap_ratio * min(hull1.volume, hull2.volume) * 0.1  # Conservative

            return estimated_volume

        except:
            return 0.0

    def calculate_hull_proximity_reward(self, hull1_points: np.ndarray, hull2_points: np.ndarray) -> float:
        """
        Reward for hulls being close (encourages approach even without overlap)
        """
        # Calculate minimum distance between hull surfaces
        distances = cdist(hull1_points, hull2_points)
        min_distance = np.min(distances)

        # Exponential reward for proximity (max at safety margin distance)
        proximity_reward = np.exp(-5.0 * max(0, min_distance - self.SAFETY_MARGIN))
        return proximity_reward

    def calculate_hand_quality_reward(self, hand_hull_points: np.ndarray) -> float:
        """
        Reward for good hand configuration (reasonable hull volume and shape)
        """
        try:
            hull = ConvexHull(hand_hull_points)
            volume = hull.volume

            # Reward for volume in reasonable range
            if self.MIN_HAND_HULL_VOLUME <= volume <= self.MAX_HAND_HULL_VOLUME:
                # Normalize volume to [0, 1] range
                volume_ratio = (volume - self.MIN_HAND_HULL_VOLUME) / (self.MAX_HAND_HULL_VOLUME - self.MIN_HAND_HULL_VOLUME)
                # Peak reward at 60% of range (not too small, not too large)
                optimal_ratio = 0.6
                quality = 1.0 - abs(volume_ratio - optimal_ratio) / optimal_ratio
                return max(0, quality)
            else:
                return 0.0

        except:
            return 0.0  # Degenerate hull

    def visualize_hulls(self, hand_hull_points: np.ndarray, object_hull_points: np.ndarray,
                       finger_positions: np.ndarray, object_pos: np.ndarray,
                       overlap_volume: float, episode_step: int) -> Optional[str]:
        """
        Generate PNG visualization of both convex hulls

        Returns:
            filepath: Path to generated PNG file
        """
        if not self.GENERATE_VIS:
            return None

        try:
            fig = plt.figure(figsize=(12, 8))
            ax = fig.add_subplot(111, projection='3d')

            # Plot hand convex hull
            try:
                hand_hull = ConvexHull(hand_hull_points)
                for simplex in hand_hull.simplices:
                    triangle = hand_hull_points[simplex]
                    ax.plot_trisurf(triangle[:, 0], triangle[:, 1], triangle[:, 2],
                                   alpha=0.3, color='blue', label='Hand Hull')
            except:
                # Fallback: just plot points
                ax.scatter(hand_hull_points[:, 0], hand_hull_points[:, 1], hand_hull_points[:, 2],
                          c='blue', s=50, alpha=0.7, label='Hand Points')

            # Plot object convex hull
            try:
                object_hull = ConvexHull(object_hull_points)
                for simplex in object_hull.simplices:
                    triangle = object_hull_points[simplex]
                    ax.plot_trisurf(triangle[:, 0], triangle[:, 1], triangle[:, 2],
                                   alpha=0.3, color='red', label='Object Hull')
            except:
                # Fallback: just plot points
                ax.scatter(object_hull_points[:, 0], object_hull_points[:, 1], object_hull_points[:, 2],
                          c='red', s=50, alpha=0.7, label='Object Points')

            # Plot actual object sphere
            u = np.linspace(0, 2 * np.pi, 20)
            v = np.linspace(0, np.pi, 20)
            x_sphere = self.OBJECT_RADIUS * np.outer(np.cos(u), np.sin(v)) + object_pos[0]
            y_sphere = self.OBJECT_RADIUS * np.outer(np.sin(u), np.sin(v)) + object_pos[1]
            z_sphere = self.OBJECT_RADIUS * np.outer(np.ones(np.size(u)), np.cos(v)) + object_pos[2]
            ax.plot_surface(x_sphere, y_sphere, z_sphere, alpha=0.8, color='orange', label='Actual Object')

            # Plot finger positions
            finger_names = ['Index', 'Middle', 'Ring', 'Thumb']
            colors = ['green', 'purple', 'brown', 'pink']
            for i, (pos, name, color) in enumerate(zip(finger_positions, finger_names, colors)):
                ax.scatter(pos[0], pos[1], pos[2], c=color, s=100, marker='o', label=f'{name} Tip')

            # Plot palm position (last point in hand hull)
            palm_pos = hand_hull_points[-1]
            ax.scatter(palm_pos[0], palm_pos[1], palm_pos[2], c='black', s=150, marker='s', label='Palm Center')

            # Set labels and title
            ax.set_xlabel('X (m)')
            ax.set_ylabel('Y (m)')
            ax.set_zlabel('Z (m)')
            ax.set_title(f'Convex Hull Overlap Visualization\n'
                        f'Step: {episode_step}, Overlap Volume: {overlap_volume:.6f} m³')

            # Set equal aspect ratio
            all_points = np.vstack([hand_hull_points, object_hull_points])
            max_range = np.ptp(all_points, axis=0).max() / 2.0
            mid_x = (all_points[:, 0].max() + all_points[:, 0].min()) * 0.5
            mid_y = (all_points[:, 1].max() + all_points[:, 1].min()) * 0.5
            mid_z = (all_points[:, 2].max() + all_points[:, 2].min()) * 0.5
            ax.set_xlim(mid_x - max_range, mid_x + max_range)
            ax.set_ylim(mid_y - max_range, mid_y + max_range)
            ax.set_zlim(mid_z - max_range, mid_z + max_range)

            # Add legend (only unique labels)
            handles, labels = ax.get_legend_handles_labels()
            by_label = dict(zip(labels, handles))
            ax.legend(by_label.values(), by_label.keys(), loc='upper right', bbox_to_anchor=(1.15, 1))

            # Save visualization
            filename = f'hull_overlap_step_{episode_step:06d}.png'
            filepath = os.path.join(self.VIS_DIR, filename)
            plt.tight_layout()
            plt.savefig(filepath, dpi=150, bbox_inches='tight')
            plt.close()

            self.VIS_COUNTER += 1
            return filepath

        except Exception as e:
            print(f"Visualization error: {e}")
            if 'fig' in locals():
                plt.close(fig)
            return None

    def calculate_reward(self, obs_dict: Dict) -> Tuple[float, Dict]:
        """
        Calculate total reward based on convex hull overlap

        Args:
            obs_dict: Dictionary containing:
                - 'finger_positions': 4x3 array of fingertip positions
                - 'palm_position': 3D palm center position
                - 'object_pos': 3D object center position
                - 'binary_contact': 4D binary contact state per finger
                - 'episode_step': Current step number (for visualization)

        Returns:
            total_reward: Combined reward value
            reward_info: Detailed breakdown for logging
        """

        finger_positions = obs_dict['finger_positions']  # 4x3
        palm_position = obs_dict['palm_position']        # 3D
        object_pos = obs_dict['object_pos']              # 3D
        binary_contact = obs_dict['binary_contact']      # 4D
        episode_step = obs_dict.get('episode_step', 0)

        # Generate convex hull points
        hand_hull_points = self.generate_hand_hull_points(finger_positions, palm_position)
        object_hull_points = self.generate_object_hull_points(object_pos)

        # ================== COMPONENT 1: Hull Overlap Volume ==================
        overlap_volume = self.calculate_hull_intersection_volume(hand_hull_points, object_hull_points)
        overlap_reward = overlap_volume * self.OVERLAP_SCALE

        # ================== COMPONENT 2: Contact Penalty ==================
        # Strong penalty for any contact (safety requirement)
        num_contacts = np.sum(binary_contact)
        contact_penalty = num_contacts * self.CONTACT_PENALTY

        # ================== COMPONENT 3: Hull Proximity Reward ==================
        # Encourage approach even without overlap
        proximity_reward = self.calculate_hull_proximity_reward(hand_hull_points, object_hull_points)
        proximity_reward *= self.APPROACH_SCALE

        # ================== COMPONENT 4: Hand Quality Reward ==================
        # Reward for reasonable hand configuration
        quality_reward = self.calculate_hand_quality_reward(hand_hull_points)
        quality_reward *= self.QUALITY_SCALE

        # ================== TOTAL REWARD CALCULATION ==================
        total_reward = overlap_reward + contact_penalty + proximity_reward + quality_reward

        # ================== VISUALIZATION ==================
        vis_filepath = None
        if self.GENERATE_VIS and episode_step % 50 == 0:  # Generate every 50 steps
            vis_filepath = self.visualize_hulls(hand_hull_points, object_hull_points,
                                              finger_positions, object_pos,
                                              overlap_volume, episode_step)

        # ================== DETAILED INFO FOR LOGGING ==================
        # Calculate hull volumes with error handling
        try:
            hand_hull_volume = float(ConvexHull(hand_hull_points).volume) if len(hand_hull_points) >= 4 else 0.0
        except:
            hand_hull_volume = 0.0  # Degenerate hull

        try:
            object_hull_volume = float(ConvexHull(object_hull_points).volume)
        except:
            object_hull_volume = 0.0  # Degenerate hull

        reward_info = {
            'overlap_reward': float(overlap_reward),
            'contact_penalty': float(contact_penalty),
            'proximity_reward': float(proximity_reward),
            'quality_reward': float(quality_reward),
            'overlap_volume': float(overlap_volume),
            'num_contacts': int(num_contacts),
            'hand_hull_volume': hand_hull_volume,
            'object_hull_volume': object_hull_volume,
            'visualization_path': vis_filepath,
        }

        return total_reward, reward_info

    def reset(self):
        """Reset episode-specific tracking variables"""
        pass  # No episode-specific state to reset

    def get_expected_reward_range(self) -> Tuple[float, float]:
        """Return expected reward range for normalization"""
        # Minimum: multiple contacts
        min_reward = 4 * self.CONTACT_PENALTY  # All fingers touching

        # Maximum: perfect overlap + proximity + quality (no contact)
        max_overlap = self.MAX_HAND_HULL_VOLUME * self.OVERLAP_SCALE  # Conservative estimate
        max_reward = max_overlap + self.APPROACH_SCALE + self.QUALITY_SCALE

        return min_reward, max_reward

    def __str__(self) -> str:
        """String representation for debugging"""
        min_r, max_r = self.get_expected_reward_range()
        return (f"ConvexHullOverlapReward(components=4, range=[{min_r:.1f}, {max_r:.1f}], "
                f"safety_margin={self.SAFETY_MARGIN}m)")


def create_default_config() -> Dict:
    """Create default configuration for ConvexHullOverlapReward"""
    return {
        'object_radius': 0.05,           # 5cm sphere
        'safety_margin': 0.025,          # 2.5cm clearance
        'object_hull_points': 12,        # Icosphere points
        'min_hand_hull_volume': 0.0001,  # 0.1cm³
        'max_hand_hull_volume': 0.01,    # 10cm³
        'overlap_scale': 10000.0,        # Scale overlap to meaningful reward
        'contact_penalty': -5.0,         # Strong penalty per contact
        'approach_scale': 2.0,           # Proximity reward scaling
        'quality_scale': 1.0,            # Hand shape reward scaling
        'vis_dir': '/tmp/convex_hull_vis',
        'generate_vis': True,            # Enable PNG generation
    }


if __name__ == "__main__":
    # Quick test of convex hull overlap reward function
    print("=" * 60)
    print("🧪 CONVEX HULL OVERLAP REWARD FUNCTION TEST")
    print("=" * 60)

    # Create reward function
    reward_func = ConvexHullOverlapReward()
    print(f"\nReward function: {reward_func}")

    # Test scenarios
    test_cases = [
        {
            "name": "Far away hands",
            "finger_pos": np.array([[0.5, 0.2, 0.4], [0.5, 0.15, 0.4], [0.5, 0.1, 0.4], [0.45, 0.15, 0.4]]),
            "palm_pos": np.array([0.45, 0.15, 0.35]),
            "object_pos": np.array([0.25, 0.15, 0.35]),
            "contact": np.array([0, 0, 0, 0])
        },
        {
            "name": "Approaching object",
            "finger_pos": np.array([[0.3, 0.2, 0.4], [0.3, 0.15, 0.4], [0.3, 0.1, 0.4], [0.25, 0.15, 0.4]]),
            "palm_pos": np.array([0.25, 0.15, 0.35]),
            "object_pos": np.array([0.25, 0.15, 0.35]),
            "contact": np.array([0, 0, 0, 0])
        },
        {
            "name": "Engulfing object",
            "finger_pos": np.array([[0.27, 0.18, 0.37], [0.27, 0.12, 0.37], [0.23, 0.12, 0.37], [0.23, 0.18, 0.37]]),
            "palm_pos": np.array([0.25, 0.15, 0.32]),
            "object_pos": np.array([0.25, 0.15, 0.35]),
            "contact": np.array([0, 0, 0, 0])
        },
        {
            "name": "Contact violation",
            "finger_pos": np.array([[0.25, 0.18, 0.35], [0.25, 0.12, 0.35], [0.25, 0.12, 0.35], [0.25, 0.18, 0.35]]),
            "palm_pos": np.array([0.25, 0.15, 0.32]),
            "object_pos": np.array([0.25, 0.15, 0.35]),
            "contact": np.array([1, 1, 0, 0])  # Two fingers touching
        }
    ]

    for i, test in enumerate(test_cases):
        obs = {
            'finger_positions': test['finger_pos'],
            'palm_position': test['palm_pos'],
            'object_pos': test['object_pos'],
            'binary_contact': test['contact'],
            'episode_step': i * 100,
        }

        reward, info = reward_func.calculate_reward(obs)
        print(f"\n📋 {test['name']:20s}")
        print(f"    Total Reward: {reward:8.3f}")
        print(f"    Overlap:      {info['overlap_reward']:8.3f} (volume: {info['overlap_volume']:.6f})")
        print(f"    Contact:      {info['contact_penalty']:8.3f} (contacts: {info['num_contacts']})")
        print(f"    Proximity:    {info['proximity_reward']:8.3f}")
        print(f"    Quality:      {info['quality_reward']:8.3f}")
        if info['visualization_path']:
            print(f"    Visualization: {info['visualization_path']}")

    print(f"\n✅ Test complete. Expected range: {reward_func.get_expected_reward_range()}")
    print(f"💾 Visualizations saved to: {reward_func.VIS_DIR}")