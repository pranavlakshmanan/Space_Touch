#!/usr/bin/env python3
"""
V7.4 Hybrid Derivative + Distance Reward Calculator

Key changes from V7.3:
- Adds WEAK distance reward (10.0 scale, 90:10 ratio with derivative)
- Adds smoothed overlap (EMA) for stable signal
- Removes bootstrap proximity (distance reward replaces it)
- Keeps derivative overlap as PRIMARY reward

Observation additions (vs V7.3):
- Smoothed overlap volume (for agent awareness)
- Relative distance vector (for navigation)

Reward composition:
- 90% Derivative overlap (±100 reward, dominant)
- 10% Distance reward (0-10 reward, navigation guide)
"""

import numpy as np
import gc
from typing import Dict, Tuple


class V74RewardCalculator:
    """
    V7.4 Hybrid reward: Derivative overlap (primary) + Distance (secondary)

    Ratio: 90:10 (derivative:distance)
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

        # V7.4: Smoothed overlap for observation space (EMA)
        self.overlap_ema = 0.0
        self.overlap_ema_alpha = config.get('overlap_ema_alpha', 0.1)  # EMA smoothing factor

        # V7.4: Previous overlap for derivative calculation
        self.previous_overlap = 0.0

        # V7.4: Derivative reward parameters (unchanged from V7.3)
        self.improvement_scale = config.get('improvement_scale', 50000.0)
        self.regression_multiplier = config.get('regression_multiplier', 1.5)
        self.sustain_threshold = config.get('sustain_threshold', 0.00005)  # 50 cm³
        self.sustain_bonus = config.get('sustain_bonus', 2.0)

        # V7.4: NEW - Distance reward parameters (90:10 ratio)
        self.distance_reward_scale = config.get('distance_reward_scale', 10.0)  # Small guide
        self.distance_reward_rate = config.get('distance_reward_rate', 2.0)  # exp(-2*d)

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
        """V7.4: Phases disabled - no-op for compatibility"""
        pass

    def reset(self):
        """Reset episode state"""
        self.consecutive_success_steps = 0

        # Reset cached values
        self.cached_overlap = 0.0
        self.cached_hand_volume = 0.0
        self.cached_object_volume = 0.0
        self.cached_hull_valid = True

        # V7.4: Reset smoothed overlap and previous
        self.overlap_ema = 0.0
        self.previous_overlap = 0.0

    def get_smoothed_overlap(self) -> float:
        """V7.4: Get smoothed overlap for observation space"""
        return self.overlap_ema

    def calculate_reward(self, obs: Dict) -> Tuple[float, Dict]:
        """
        V7.4 Hybrid reward calculation

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

            # Translate object hull template
            object_points = self.object_hull_template + object_pos

            # Compute hulls at reduced frequency (20Hz)
            should_compute_hulls = (self.step_counter % self.hull_compute_freq == 0)

            if should_compute_hulls:
                overlap_volume, hand_volume, object_volume, hull_valid = self._calculate_overlap_bbox_fast(
                    hand_points, object_points
                )

                # Update cache
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

            # V7.4: Update smoothed overlap (EMA)
            self.overlap_ema = (1 - self.overlap_ema_alpha) * self.overlap_ema + self.overlap_ema_alpha * overlap_volume

            # Calculate distance
            hand_center = np.mean(hand_points, axis=0)
            distance = np.linalg.norm(hand_center - object_pos)

            # Count contacts
            num_contacts = int(np.sum(binary_contact))

            # ============================================
            # V7.4 HYBRID REWARD CALCULATION
            # ============================================

            reward = 0.0
            reward_components = {}

            # Component 1: Derivative overlap (PRIMARY - 90%)
            delta_overlap = overlap_volume - self.previous_overlap

            if delta_overlap > 0:
                delta_reward = delta_overlap * self.improvement_scale
            elif delta_overlap < 0:
                delta_reward = delta_overlap * self.improvement_scale * self.regression_multiplier
            else:
                delta_reward = 0.0

            reward += delta_reward
            reward_components['delta_overlap'] = delta_reward

            # Component 2: Distance reward (SECONDARY - 10%, NEW!)
            distance_reward = self.distance_reward_scale * np.exp(-self.distance_reward_rate * distance)
            reward += distance_reward
            reward_components['distance'] = distance_reward

            # Component 3: Sustain bonus
            sustain_reward = 0.0
            if overlap_volume > self.sustain_threshold:
                sustain_reward = self.sustain_bonus
            reward += sustain_reward
            reward_components['sustain_bonus'] = sustain_reward

            # Component 4: Distance-scaled contact penalty
            contact_penalty = 0.0
            if num_contacts > 0:
                distance_factor = np.clip(1.0 - (distance / self.contact_distance_scale), 0.0, 1.0)
                penalty_per_contact = self.contact_penalty_min + (self.contact_penalty_max - self.contact_penalty_min) * distance_factor
                contact_penalty = penalty_per_contact * num_contacts

            reward += contact_penalty
            reward_components['contact_penalty'] = contact_penalty

            # Update previous overlap for next step
            self.previous_overlap = overlap_volume

            # Check success
            is_success = (overlap_volume > 0.0001 and num_contacts == 0)
            if is_success:
                self.consecutive_success_steps += 1
            else:
                self.consecutive_success_steps = 0

            # Build info dict
            info = {
                'total_reward': reward,
                'delta_overlap_reward': reward_components['delta_overlap'],
                'distance_reward': reward_components['distance'],
                'sustain_bonus': reward_components['sustain_bonus'],
                'contact_penalty': reward_components['contact_penalty'],
                'overlap_volume': overlap_volume,
                'smoothed_overlap_volume': self.overlap_ema,  # V7.4: For observation
                'delta_overlap_volume': delta_overlap,
                'hand_hull_volume': hand_volume,
                'object_hull_volume': object_volume,
                'hand_hull_valid': hull_valid,
                'object_hull_valid': hull_valid,
                'distance_to_target': distance,
                'relative_distance_vector': object_pos - hand_center,  # V7.4: For observation
                'num_contacts': num_contacts,
                'is_success': is_success,
                'consecutive_success_steps': self.consecutive_success_steps,
                'current_phase': 0,
                'error': '',
                # Legacy fields for compatibility
                'overlap_reward': reward_components['delta_overlap'],
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
            'delta_overlap_reward': 0.0,
            'distance_reward': 0.0,
            'sustain_bonus': 0.0,
            'contact_penalty': 0.0,
            'overlap_volume': 0.0,
            'smoothed_overlap_volume': 0.0,
            'delta_overlap_volume': 0.0,
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
            'overlap_reward': 0.0,
            'proximity_reward': 0.0,
            'clearance_reward': 0.0,
            'quality_reward': 0.0,
            'sustained_bonus': 0.0,
        }
