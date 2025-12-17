#!/usr/bin/env python3
"""
V7.6 Curriculum-Based Spatial Caging Reward Calculator

Key changes from V7.5.1:
- PRIMARY: Palm-to-target distance derivative (stable, not affected by fingers)
- SECONDARY: Bbox overlap volume derivative (activated when distance < threshold)
- PENALTY: Contact penalty (distance-scaled)
- Curriculum: 3 phases with decreasing spawn distance
- REMOVED: Memory optimization (hull_compute_freq) - compute every step

Reward composition:
- Phase 1-3: Distance derivative + Overlap derivative + Contact penalty
- Overlap reward only activates when distance < 8cm (close enough for overlap to matter)

Curriculum phases:
- Phase 1: Spawn 3.6cm from target (overlap possible at spawn)
- Phase 2: Spawn 5.8cm from target (overlap possible at spawn)
- Phase 3: Spawn 11.2cm from target (must approach for overlap)
"""

import numpy as np
from typing import Dict, Tuple


class V76RewardCalculator:
    """
    V7.6 Curriculum-based reward calculator with stable palm-distance metric.
    """

    def __init__(self, config: Dict = None):
        config = config or {}

        # Object geometry
        self.object_radius = config.get('object_radius', 0.05)
        self.safety_margin = config.get('safety_margin', 0.025)
        self.effective_radius = self.object_radius + self.safety_margin

        # Curriculum phases - spawn offsets from target
        self.phase_offsets = {
            1: np.array([-0.03, 0.0, 0.02]),   # 3.6cm, overlap at spawn
            2: np.array([-0.05, 0.0, 0.03]),   # 5.8cm, overlap at spawn
            3: np.array([-0.10, 0.0, 0.05]),   # 11.2cm, must approach
        }
        self.current_phase = 1

        # Distance reward parameters
        self.distance_improvement_scale = config.get('distance_improvement_scale', 500.0)
        self.distance_regression_scale = config.get('distance_regression_scale', 100.0)  # Symmetric

        # Overlap reward parameters (only active when close)
        self.overlap_activation_distance = config.get('overlap_activation_distance', 0.08)  # 8cm
        self.overlap_improvement_scale = config.get('overlap_improvement_scale', 10000.0)
        self.overlap_regression_scale = config.get('overlap_regression_scale', 10000.0)  # Symmetric

        # Sustain bonus (for maintaining overlap)
        self.sustain_threshold = config.get('sustain_threshold', 0.00001)  # 10 cm³
        self.sustain_bonus = config.get('sustain_bonus', 1.0)

        # Contact penalty
        self.contact_penalty = config.get('contact_penalty', -0.5)

        # State tracking
        self.previous_distance = None
        self.previous_overlap = 0.0
        self.consecutive_success_steps = 0

        # For observation space (smoothed overlap)
        self.overlap_ema = 0.0
        self.overlap_ema_alpha = config.get('overlap_ema_alpha', 0.1)

    def get_spawn_offset(self) -> np.ndarray:
        """Get spawn offset for current phase"""
        return self.phase_offsets.get(self.current_phase, self.phase_offsets[3])

    def update_phase(self, phase: int):
        """Update curriculum phase"""
        self.current_phase = np.clip(phase, 1, 3)

    def reset(self):
        """Reset episode state"""
        self.previous_distance = None
        self.previous_overlap = 0.0
        self.consecutive_success_steps = 0
        self.overlap_ema = 0.0

    def get_smoothed_overlap(self) -> float:
        """Get smoothed overlap for observation space"""
        return self.overlap_ema

    def _compute_overlap_volume(self, hand_points: np.ndarray, object_pos: np.ndarray) -> float:
        """
        Compute bounding box overlap volume between hand and object.

        Returns:
            float: Overlap volume in m³ (0 if no overlap)
        """
        # Hand bounding box
        hand_min = np.min(hand_points, axis=0)
        hand_max = np.max(hand_points, axis=0)

        # Object bounding box
        obj_min = object_pos - self.effective_radius
        obj_max = object_pos + self.effective_radius

        # Intersection
        intersection_min = np.maximum(hand_min, obj_min)
        intersection_max = np.minimum(hand_max, obj_max)

        # Check if valid intersection
        if np.any(intersection_min >= intersection_max):
            return 0.0

        # Volume (with 0.3 factor for bbox-to-realistic approximation)
        overlap_volume = np.prod(intersection_max - intersection_min) * 0.3
        return overlap_volume

    def calculate_reward(self, obs: Dict) -> Tuple[float, Dict]:
        """
        Calculate V7.6 reward.

        Args:
            obs: Dictionary containing:
                - finger_positions: (4, 3) array of fingertip positions
                - finger_bases: (4, 3) array of finger base positions
                - palm_position: (3,) array of palm/base position
                - object_pos: (3,) array of target position
                - binary_contact: (4,) array of contact flags

        Returns:
            tuple: (total_reward, info_dict)
        """
        try:
            # Extract observations
            finger_positions = np.array(obs['finger_positions']).reshape(4, 3)
            finger_bases = np.array(obs['finger_bases']).reshape(4, 3)
            palm_position = np.array(obs['palm_position']).flatten()[:3]
            object_pos = np.array(obs['object_pos']).flatten()[:3]
            binary_contact = np.array(obs['binary_contact']).flatten()[:4]

            # Build hand points for overlap calculation
            hand_points = np.vstack([
                finger_positions,
                finger_bases,
                palm_position.reshape(1, 3)
            ])

            # ============================================
            # METRIC 1: Palm-to-target distance (STABLE)
            # ============================================
            current_distance = np.linalg.norm(palm_position - object_pos)

            # ============================================
            # METRIC 2: Bbox overlap volume
            # ============================================
            current_overlap = self._compute_overlap_volume(hand_points, object_pos)

            # Update smoothed overlap for observation
            self.overlap_ema = (1 - self.overlap_ema_alpha) * self.overlap_ema + self.overlap_ema_alpha * current_overlap

            # ============================================
            # REWARD CALCULATION
            # ============================================
            reward = 0.0
            reward_components = {}

            # Component 1: Distance derivative (PRIMARY)
            distance_reward = 0.0
            if self.previous_distance is not None:
                delta_distance = self.previous_distance - current_distance  # Positive if got closer
                if delta_distance > 0:
                    distance_reward = delta_distance * self.distance_improvement_scale
                else:
                    distance_reward = delta_distance * self.distance_regression_scale  # Negative

            reward += distance_reward
            reward_components['distance_derivative'] = distance_reward

            # Component 2: Overlap derivative (SECONDARY - only when close)
            overlap_reward = 0.0
            if current_distance < self.overlap_activation_distance:
                delta_overlap = current_overlap - self.previous_overlap
                if delta_overlap > 0:
                    overlap_reward = delta_overlap * self.overlap_improvement_scale
                else:
                    overlap_reward = delta_overlap * self.overlap_regression_scale  # Negative

            reward += overlap_reward
            reward_components['overlap_derivative'] = overlap_reward

            # Component 3: Sustain bonus (for maintaining overlap)
            sustain_reward = 0.0
            if current_overlap > self.sustain_threshold:
                sustain_reward = self.sustain_bonus
            reward += sustain_reward
            reward_components['sustain_bonus'] = sustain_reward

            # Component 4: Contact penalty
            num_contacts = int(np.sum(binary_contact))
            contact_penalty_value = 0.0
            if num_contacts > 0:
                contact_penalty_value = self.contact_penalty * num_contacts
            reward += contact_penalty_value
            reward_components['contact_penalty'] = contact_penalty_value

            # Clip total reward for stability
            reward = np.clip(reward, -20.0, 20.0)

            # Update state for next step
            self.previous_distance = current_distance
            self.previous_overlap = current_overlap

            # Check success (overlap achieved, no contact)
            is_success = (current_overlap > self.sustain_threshold) and (num_contacts == 0)
            if is_success:
                self.consecutive_success_steps += 1
            else:
                self.consecutive_success_steps = 0

            # Per-axis overlap for logging
            hand_min = np.min(hand_points, axis=0)
            hand_max = np.max(hand_points, axis=0)
            obj_min = object_pos - self.effective_radius
            obj_max = object_pos + self.effective_radius
            axis_overlaps = np.array([
                min(hand_max[i], obj_max[i]) - max(hand_min[i], obj_min[i])
                for i in range(3)
            ])

            # Build info dict
            info = {
                'total_reward': reward,
                # Reward components
                'distance_derivative_reward': reward_components['distance_derivative'],
                'overlap_derivative_reward': reward_components['overlap_derivative'],
                'sustain_bonus': reward_components['sustain_bonus'],
                'contact_penalty': reward_components['contact_penalty'],
                # State metrics
                'palm_to_target_distance': current_distance,
                'overlap_volume': current_overlap,
                'smoothed_overlap_volume': self.overlap_ema,
                'distance_to_target': current_distance,  # Alias for compatibility
                # Per-axis overlaps (for debugging)
                'axis_overlap_x': axis_overlaps[0],
                'axis_overlap_y': axis_overlaps[1],
                'axis_overlap_z': axis_overlaps[2],
                # Contact info
                'num_contacts': num_contacts,
                'binary_contacts': binary_contact.tolist(),
                # Success tracking
                'is_success': is_success,
                'consecutive_success_steps': self.consecutive_success_steps,
                # Curriculum
                'current_phase': self.current_phase,
                # Hand geometry (for debugging)
                'hand_bbox_min': hand_min.tolist(),
                'hand_bbox_max': hand_max.tolist(),
                'palm_position': palm_position.tolist(),
                # Legacy compatibility
                'hand_hull_volume': np.prod(hand_max - hand_min) * 0.5,
                'object_hull_volume': (4/3) * np.pi * (self.effective_radius ** 3),
                'hand_hull_valid': True,
                'error': '',
            }

            return reward, info

        except Exception as e:
            import traceback
            traceback.print_exc()
            return 0.0, self._error_info(str(e))

    def _error_info(self, error_msg: str) -> Dict:
        """Return info dict for error cases"""
        return {
            'total_reward': 0.0,
            'distance_derivative_reward': 0.0,
            'overlap_derivative_reward': 0.0,
            'sustain_bonus': 0.0,
            'contact_penalty': 0.0,
            'palm_to_target_distance': 0.0,
            'overlap_volume': 0.0,
            'smoothed_overlap_volume': 0.0,
            'distance_to_target': 0.0,
            'axis_overlap_x': 0.0,
            'axis_overlap_y': 0.0,
            'axis_overlap_z': 0.0,
            'num_contacts': 0,
            'binary_contacts': [0, 0, 0, 0],
            'is_success': False,
            'consecutive_success_steps': 0,
            'current_phase': self.current_phase,
            'hand_bbox_min': [0, 0, 0],
            'hand_bbox_max': [0, 0, 0],
            'palm_position': [0, 0, 0],
            'hand_hull_volume': 0.0,
            'object_hull_volume': 0.0,
            'hand_hull_valid': False,
            'error': error_msg,
        }
