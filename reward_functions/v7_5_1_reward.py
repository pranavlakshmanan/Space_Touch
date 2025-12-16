#!/usr/bin/env python3
"""
V7.5.1 Per-Axis Derivative Reward Calculator (Stabilized)

Key changes from V7.5:
- Reduced reward scaling (10x reduction) for stable learning
- Symmetric penalties (regression_multiplier = 1.0)
- Added reward clipping to prevent extreme values
- Per-axis rewards clipped to ±10.0
- Total reward clipped to ±20.0

Reward composition (same as V7.5, but scaled down):
- PRIMARY: Per-axis derivative rewards (50.0 scale, was 500.0)
- SECONDARY: Full 3D overlap derivative bonus (5000.0 scale, was 50000.0)
- TERTIARY: Distance reward (weak navigation guide)
- PENALTY: Contact penalty (distance-scaled)

This approach is spawn-position independent and generalizable.
"""

import numpy as np
import gc
from typing import Dict, Tuple


class V751RewardCalculator:
    """
    V7.5.1 Per-Axis Derivative reward calculator (Stabilized)

    Rewards progress on X, Y, Z axes independently, then bonus for full 3D overlap.
    Uses reduced scaling and reward clipping for stable learning.
    """

    def __init__(self, config: Dict = None):
        config = config or {}

        # Object geometry
        self.object_radius = config.get('object_radius', 0.05)
        self.safety_margin = config.get('safety_margin', 0.025)
        self.object_hull_points = config.get('object_hull_points', 32)

        # Hull validation
        self.MIN_VALID_VOLUME = 1e-9

        # Memory optimization
        self.hull_compute_freq = config.get('hull_compute_freq', 12)  # 20Hz
        self.step_counter = 0

        # Cached hull results
        self.cached_overlap = 0.0
        self.cached_hand_volume = 0.0
        self.cached_object_volume = 0.0
        self.cached_hull_valid = True

        # Pre-compute object hull (static)
        self.object_hull_template = self._generate_sphere_hull_points(
            self.object_radius + self.safety_margin,
            self.object_hull_points
        )

        # Smoothed overlap for observation space (EMA)
        self.overlap_ema = 0.0
        self.overlap_ema_alpha = config.get('overlap_ema_alpha', 0.1)

        # V7.5: Per-axis overlap tracking
        self.previous_axis_overlaps = np.zeros(3)  # X, Y, Z
        self.previous_full_overlap = 0.0

        # V7.5.1: STABILIZED per-axis reward parameters (10x reduction)
        self.axis_improvement_scale = config.get('axis_improvement_scale', 50.0)  # Reduced from 500.0
        self.axis_regression_multiplier = config.get('axis_regression_multiplier', 1.0)  # Symmetric (was 1.5)

        # V7.5.1: STABILIZED full overlap bonus (10x reduction)
        self.full_overlap_bonus_scale = config.get('full_overlap_bonus_scale', 5000.0)  # Reduced from 50000.0

        # Sustain bonus
        self.sustain_threshold = config.get('sustain_threshold', 0.00005)  # 50 cm³
        self.sustain_bonus = config.get('sustain_bonus', 2.0)

        # Distance reward (reduced from V7.4)
        self.distance_reward_scale = config.get('distance_reward_scale', 5.0)
        self.distance_reward_rate = config.get('distance_reward_rate', 2.0)

        # Distance-scaled contact penalty
        self.contact_penalty_min = config.get('contact_penalty_min', -1.0)
        self.contact_penalty_max = config.get('contact_penalty_max', -5.0)
        self.contact_distance_scale = config.get('contact_distance_scale', 0.25)

        # Dummy phase variable for compatibility
        self.current_phase = 0
        self.consecutive_success_steps = 0

    def _generate_sphere_hull_points(self, radius: float, n_points: int) -> np.ndarray:
        """Generate evenly distributed points on sphere using Fibonacci lattice"""
        points = []
        phi = np.pi * (3.0 - np.sqrt(5.0))

        for i in range(n_points):
            y = 1 - (i / float(n_points - 1)) * 2
            r = np.sqrt(1 - y * y)
            theta = phi * i

            x = np.cos(theta) * r
            z = np.sin(theta) * r
            points.append([x * radius, y * radius, z * radius])

        return np.array(points)

    def update_phase(self, phase: int):
        """V7.5: Phases disabled - no-op for compatibility"""
        pass

    def reset(self):
        """Reset episode state"""
        self.consecutive_success_steps = 0
        self.step_counter = 0  # CRITICAL FIX: Reset step counter each episode

        # Reset cached values
        self.cached_overlap = 0.0
        self.cached_hand_volume = 0.0
        self.cached_object_volume = 0.0
        self.cached_hull_valid = True

        # V7.5: Reset per-axis tracking
        self.previous_axis_overlaps = np.zeros(3)
        self.previous_full_overlap = 0.0

        # Reset smoothed overlap
        self.overlap_ema = 0.0

    def get_smoothed_overlap(self) -> float:
        """Get smoothed overlap for observation space"""
        return self.overlap_ema

    def _compute_axis_overlaps(self, hand_points: np.ndarray, object_pos: np.ndarray) -> np.ndarray:
        """
        Compute overlap amount on each axis independently.

        Returns:
            np.ndarray: Shape (3,) with overlap amount per axis
                        Negative = gap, Positive = overlap
        """
        # Hand bounding box
        hand_min = np.min(hand_points, axis=0)
        hand_max = np.max(hand_points, axis=0)

        # Object bounding box (sphere approximation)
        effective_radius = self.object_radius + self.safety_margin
        obj_min = object_pos - effective_radius
        obj_max = object_pos + effective_radius

        # Per-axis overlap: positive = overlap, negative = gap
        axis_overlaps = np.zeros(3)
        for i in range(3):
            overlap_start = max(hand_min[i], obj_min[i])
            overlap_end = min(hand_max[i], obj_max[i])
            axis_overlaps[i] = overlap_end - overlap_start

        return axis_overlaps

    def calculate_reward(self, obs: Dict) -> Tuple[float, Dict]:
        """
        V7.5 Per-Axis Derivative Reward Calculation

        Returns:
            tuple: (total_reward, info_dict)
        """
        try:
            self.step_counter += 1

            # Extract observations
            finger_positions = np.array(obs['finger_positions']).reshape(4, 3)
            finger_bases = np.array(obs['finger_bases']).reshape(4, 3)
            palm_position = np.array(obs['palm_position']).flatten()[:3]
            object_pos = np.array(obs['object_pos']).flatten()[:3]
            binary_contact = np.array(obs['binary_contact']).flatten()[:4]

            # Build 9-point hand hull
            hand_points = np.vstack([
                finger_positions,
                finger_bases,
                palm_position.reshape(1, 3)
            ])

            # Compute per-axis overlaps (ALWAYS, not just at hull_compute_freq)
            axis_overlaps = self._compute_axis_overlaps(hand_points, object_pos)

            # Compute full overlap at reduced frequency for bonus
            should_compute_hulls = (self.step_counter % self.hull_compute_freq == 0)

            if should_compute_hulls:
                # Translate object hull template
                object_points = self.object_hull_template + object_pos

                overlap_volume, hand_volume, object_volume, hull_valid = self._calculate_overlap_bbox_fast(
                    hand_points, object_points
                )

                self.cached_overlap = overlap_volume
                self.cached_hand_volume = hand_volume
                self.cached_object_volume = object_volume
                self.cached_hull_valid = hull_valid

                gc.collect()
            else:
                overlap_volume = self.cached_overlap
                hand_volume = self.cached_hand_volume
                object_volume = self.cached_object_volume
                hull_valid = self.cached_hull_valid

            if not hull_valid:
                return 0.0, self._error_info("Invalid hull geometry")

            # Update smoothed overlap (EMA)
            self.overlap_ema = (1 - self.overlap_ema_alpha) * self.overlap_ema + self.overlap_ema_alpha * overlap_volume

            # Calculate distance
            hand_center = np.mean(hand_points, axis=0)
            distance = np.linalg.norm(hand_center - object_pos)

            # Count contacts
            num_contacts = int(np.sum(binary_contact))

            # ============================================
            # V7.5 PER-AXIS DERIVATIVE REWARD
            # ============================================

            reward = 0.0
            reward_components = {}

            # Component 1: Per-axis derivative rewards (PRIMARY)
            delta_axis = axis_overlaps - self.previous_axis_overlaps

            axis_reward = 0.0
            for i, (delta, name) in enumerate(zip(delta_axis, ['x', 'y', 'z'])):
                if delta > 0:
                    # Improvement on this axis
                    axis_r = delta * self.axis_improvement_scale
                elif delta < 0:
                    # Regression on this axis (penalize harder)
                    axis_r = delta * self.axis_improvement_scale * self.axis_regression_multiplier
                else:
                    axis_r = 0.0

                axis_reward += axis_r
                reward_components[f'axis_{name}'] = axis_r

            # V7.5.1: Clip per-axis reward to prevent extreme values
            axis_reward = np.clip(axis_reward, -10.0, 10.0)

            reward += axis_reward
            reward_components['total_axis'] = axis_reward

            # Component 2: Full 3D overlap bonus (when all axes overlap)
            full_overlap_reward = 0.0
            if np.all(axis_overlaps > 0):  # All axes have positive overlap
                delta_full = overlap_volume - self.previous_full_overlap
                if delta_full > 0:
                    full_overlap_reward = delta_full * self.full_overlap_bonus_scale
                elif delta_full < 0:
                    full_overlap_reward = delta_full * self.full_overlap_bonus_scale * self.axis_regression_multiplier

            reward += full_overlap_reward
            reward_components['full_overlap_bonus'] = full_overlap_reward

            # Component 3: Sustain bonus (for maintaining high overlap)
            sustain_reward = 0.0
            if overlap_volume > self.sustain_threshold:
                sustain_reward = self.sustain_bonus
            reward += sustain_reward
            reward_components['sustain_bonus'] = sustain_reward

            # Component 4: Distance reward (weak guide)
            distance_reward = self.distance_reward_scale * np.exp(-self.distance_reward_rate * distance)
            reward += distance_reward
            reward_components['distance'] = distance_reward

            # Component 5: Contact penalty (distance-scaled)
            contact_penalty = 0.0
            if num_contacts > 0:
                distance_factor = np.clip(1.0 - (distance / self.contact_distance_scale), 0.0, 1.0)
                penalty_per_contact = self.contact_penalty_min + (self.contact_penalty_max - self.contact_penalty_min) * distance_factor
                contact_penalty = penalty_per_contact * num_contacts
            reward += contact_penalty
            reward_components['contact_penalty'] = contact_penalty

            # Update previous values for next step
            self.previous_axis_overlaps = axis_overlaps.copy()
            self.previous_full_overlap = overlap_volume

            # V7.5.1: Clip total reward to stabilize learning
            reward = np.clip(reward, -20.0, 20.0)

            # Check success (high overlap, no contact)
            is_success = (overlap_volume > 0.0001 and num_contacts == 0)
            if is_success:
                self.consecutive_success_steps += 1
            else:
                self.consecutive_success_steps = 0

            # Build info dict
            info = {
                'total_reward': reward,
                # Per-axis components
                'axis_x_reward': reward_components.get('axis_x', 0),
                'axis_y_reward': reward_components.get('axis_y', 0),
                'axis_z_reward': reward_components.get('axis_z', 0),
                'total_axis_reward': reward_components['total_axis'],
                'full_overlap_bonus': reward_components['full_overlap_bonus'],
                'sustain_bonus': reward_components['sustain_bonus'],
                'distance_reward': reward_components['distance'],
                'contact_penalty': reward_components['contact_penalty'],
                # Per-axis overlap values (for logging)
                'axis_overlap_x': axis_overlaps[0],
                'axis_overlap_y': axis_overlaps[1],
                'axis_overlap_z': axis_overlaps[2],
                # Existing fields
                'overlap_volume': overlap_volume,
                'smoothed_overlap_volume': self.overlap_ema,
                'hand_hull_volume': hand_volume,
                'object_hull_volume': object_volume,
                'hand_hull_valid': hull_valid,
                'distance_to_target': distance,
                'relative_distance_vector': object_pos - hand_center,
                'num_contacts': num_contacts,
                'is_success': is_success,
                'consecutive_success_steps': self.consecutive_success_steps,
                'current_phase': 0,
                'error': '',
                # Legacy compatibility
                'delta_overlap_reward': reward_components['full_overlap_bonus'],
                'proximity_reward': reward_components['distance'],
                'clearance_reward': 0.0,
                'quality_reward': 0.0,
                'sustained_bonus': reward_components['sustain_bonus'],
            }

            return reward, info

        except Exception as e:
            import traceback
            traceback.print_exc()
            return 0.0, self._error_info(str(e))

    def _calculate_overlap_bbox_fast(self, hand_points: np.ndarray,
                                       object_points: np.ndarray) -> Tuple[float, float, float, bool]:
        """Fast bounding-box overlap approximation"""
        try:
            # Validation
            if len(hand_points) < 4 or len(object_points) < 4:
                return 0.0, 0.0, 0.0, False

            if np.any(np.isnan(hand_points)) or np.any(np.isinf(hand_points)):
                return 0.0, 0.0, 0.0, False

            if np.any(np.isnan(object_points)) or np.any(np.isinf(object_points)):
                return 0.0, 0.0, 0.0, False

            # Compute bounding boxes
            hand_min = np.min(hand_points, axis=0)
            hand_max = np.max(hand_points, axis=0)
            obj_min = np.min(object_points, axis=0)
            obj_max = np.max(object_points, axis=0)

            # Approximate volumes
            hand_volume = np.prod(hand_max - hand_min) * 0.5
            obj_volume = np.prod(obj_max - obj_min) * 0.5

            # Check minimum volume
            if hand_volume < self.MIN_VALID_VOLUME or obj_volume < self.MIN_VALID_VOLUME:
                return 0.0, 0.0, 0.0, False

            # Compute intersection
            intersection_min = np.maximum(hand_min, obj_min)
            intersection_max = np.minimum(hand_max, obj_max)

            # Check if there's any intersection
            if np.any(intersection_min >= intersection_max):
                return 0.0, hand_volume, obj_volume, True

            # Compute overlap volume
            overlap_volume = np.prod(intersection_max - intersection_min) * 0.3

            return overlap_volume, hand_volume, obj_volume, True

        except Exception:
            return 0.0, 0.0, 0.0, False

    def _error_info(self, error_msg: str) -> Dict:
        """Return info dict for error cases"""
        return {
            'total_reward': 0.0,
            'axis_x_reward': 0.0,
            'axis_y_reward': 0.0,
            'axis_z_reward': 0.0,
            'total_axis_reward': 0.0,
            'full_overlap_bonus': 0.0,
            'sustain_bonus': 0.0,
            'distance_reward': 0.0,
            'contact_penalty': 0.0,
            'axis_overlap_x': 0.0,
            'axis_overlap_y': 0.0,
            'axis_overlap_z': 0.0,
            'overlap_volume': 0.0,
            'smoothed_overlap_volume': 0.0,
            'hand_hull_volume': 0.0,
            'object_hull_volume': 0.0,
            'hand_hull_valid': False,
            'object_hull_valid': False,
            'distance_to_target': 0.0,
            'relative_distance_vector': np.zeros(3),
            'num_contacts': 0,
            'is_success': False,
            'consecutive_success_steps': 0,
            'current_phase': 0,
            'error': error_msg,
            # Legacy fields
            'delta_overlap_reward': 0.0,
            'proximity_reward': 0.0,
            'clearance_reward': 0.0,
            'quality_reward': 0.0,
            'sustained_bonus': 0.0,
        }
