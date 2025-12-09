#!/usr/bin/env python3
"""
V7.3 Derivative Reward Calculator

Key innovation: Rewards CHANGE in overlap, not absolute overlap.
- Improvement → positive reward
- Regression → penalty (asymmetric, hurts 1.5x more)
- Plateau → zero reward (forces continued improvement)
- Sustain bonus for maintaining high overlap

Two stages (not phases with transitions):
- Bootstrap (first 30K steps): Small proximity guide when overlap=0
- Main (rest of training): Pure derivative reward

This eliminates:
- Local optima plateaus (can't rest at "good enough")
- Oscillation/reward hacking (close-far-close nets zero reward over time)
- Skill regression (any decrease is penalized)
- Phase transition catastrophic forgetting

Based on V7 reward calculator with all bug fixes retained.
"""

import numpy as np
import gc
from typing import Dict, Tuple


class V73RewardCalculator:
    """
    V7.3 Derivative-based reward system with bootstrap stage.

    No phases - single reward function throughout training.
    Bootstrap proximity guide helps agent discover target initially.
    """

    def __init__(self, config: Dict = None):
        config = config or {}

        # Object geometry
        self.object_radius = config.get('object_radius', 0.05)  # 5cm sphere
        self.safety_margin = config.get('safety_margin', 0.025)  # 2.5cm clearance
        self.object_hull_points = config.get('object_hull_points', 32)

        # Hull validation
        self.MIN_VALID_VOLUME = 1e-9  # 0.001 mm³

        # Memory optimization: Reduce hull computation frequency
        self.hull_compute_freq = config.get('hull_compute_freq', 12)  # 20Hz at 240Hz sim
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

        # V7.3: Previous overlap for derivative calculation
        self.previous_overlap = 0.0

        # V7.3: Total steps counter for bootstrap stage
        self.total_steps = 0

        # V7.3: Derivative reward parameters
        self.improvement_scale = config.get('improvement_scale', 50000.0)  # Scale small m³ deltas
        self.regression_multiplier = config.get('regression_multiplier', 1.5)  # Regression hurts more
        self.sustain_threshold = config.get('sustain_threshold', 0.00005)  # 50 cm³ in m³
        self.sustain_bonus = config.get('sustain_bonus', 2.0)  # Reward for maintaining high overlap
        self.bootstrap_proximity_scale = config.get('bootstrap_proximity_scale', 5.0)  # Small proximity guide
        self.bootstrap_steps = config.get('bootstrap_steps', 30000)  # When to disable proximity

        # V7.3: Distance-scaled contact penalty parameters
        self.contact_penalty_min = config.get('contact_penalty_min', -1.0)  # When far
        self.contact_penalty_max = config.get('contact_penalty_max', -5.0)  # When close
        self.contact_distance_scale = config.get('contact_distance_scale', 0.25)  # Distance normalization

        # Dummy phase variable for compatibility (always 0)
        self.current_phase = 0
        self.consecutive_success_steps = 0

    def _generate_sphere_hull_points(self, radius: float, n_points: int) -> np.ndarray:
        """Generate evenly distributed points on sphere using Fibonacci lattice"""
        points = []
        phi = np.pi * (3.0 - np.sqrt(5.0))  # Golden angle

        for i in range(n_points):
            y = 1 - (i / float(n_points - 1)) * 2
            r = np.sqrt(1 - y * y)
            theta = phi * i

            x = np.cos(theta) * r
            z = np.sin(theta) * r
            points.append([x * radius, y * radius, z * radius])

        return np.array(points)

    def update_phase(self, phase: int):
        """V7.3: Phases disabled - no-op for compatibility with environment"""
        pass

    def reset(self):
        """Reset episode state and set previous_overlap to current"""
        self.consecutive_success_steps = 0
        # CRITICAL: Do NOT reset step_counter (V6 bug fix)
        # step_counter is global across episodes for hull computation frequency

        # V7.3: Reset cached values
        self.cached_overlap = 0.0
        self.cached_hand_volume = 0.0
        self.cached_object_volume = 0.0
        self.cached_hull_valid = True

        # V7.3: IMPORTANT - Set previous_overlap to zero on episode start
        # This prevents artificial penalty from episode discontinuity
        # Will be updated to actual overlap after first step
        self.previous_overlap = 0.0

    def calculate_reward(self, obs: Dict) -> Tuple[float, Dict]:
        """
        V7.3 Derivative-based reward calculation

        Returns:
            tuple: (total_reward, info_dict)
        """
        # DEBUG: Print on first call
        if not hasattr(self, '_reward_calc_called'):
            import sys
            print("[DEBUG V7.3] calculate_reward() called for first time", flush=True)
            sys.stdout.flush()
            self._reward_calc_called = True

        try:
            # Increment step counters
            self.total_steps += 1
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

            # Memory optimization: Compute hulls at reduced frequency (20Hz)
            should_compute_hulls = (self.step_counter % self.hull_compute_freq == 0)

            # DEBUG: Print on first hull computation
            if should_compute_hulls and not hasattr(self, '_first_hull_compute'):
                import sys
                print(f"[DEBUG V7.3] First hull computation at step {self.step_counter}, freq={self.hull_compute_freq}", flush=True)
                sys.stdout.flush()
                self._first_hull_compute = True

            if should_compute_hulls:
                # Fast bounding-box approximation
                overlap_volume, hand_volume, object_volume, hull_valid = self._calculate_overlap_bbox_fast(
                    hand_points, object_points
                )

                # Update cache
                self.cached_overlap = overlap_volume
                self.cached_hand_volume = hand_volume
                self.cached_object_volume = object_volume
                self.cached_hull_valid = hull_valid

                # DEBUG: Print hull volumes every 480 steps (aligns with WandB logging)
                if self.step_counter % 480 == 0:
                    import sys
                    print(f"\n[DEBUG V7.3 Step {self.step_counter}] Hull computed:", flush=True)
                    print(f"  Hand volume: {hand_volume*1e6:.2f} cm³", flush=True)
                    print(f"  Object volume: {object_volume*1e6:.2f} cm³", flush=True)
                    print(f"  Overlap volume: {overlap_volume*1e6:.2f} cm³", flush=True)
                    print(f"  Previous overlap: {self.previous_overlap*1e6:.2f} cm³", flush=True)
                    print(f"  Delta: {(overlap_volume - self.previous_overlap)*1e6:.2f} cm³", flush=True)
                    sys.stdout.flush()

                gc.collect()
            else:
                # Use cached values
                overlap_volume = self.cached_overlap
                hand_volume = self.cached_hand_volume
                object_volume = self.cached_object_volume
                hull_valid = self.cached_hull_valid

            if not hull_valid:
                return 0.0, self._error_info("Invalid hull geometry")

            # Calculate distance
            hand_center = np.mean(hand_points, axis=0)
            distance = np.linalg.norm(hand_center - object_pos)

            # Count contacts
            num_contacts = int(np.sum(binary_contact))

            # ============================================
            # V7.3 DERIVATIVE REWARD CALCULATION
            # ============================================

            reward = 0.0
            reward_components = {}

            # Component 1: Delta overlap (THE CORE INNOVATION)
            delta_overlap = overlap_volume - self.previous_overlap

            if delta_overlap > 0:
                # Improvement: positive reward
                delta_reward = delta_overlap * self.improvement_scale
            elif delta_overlap < 0:
                # Regression: asymmetric penalty (hurts 1.5x more)
                delta_reward = delta_overlap * self.improvement_scale * self.regression_multiplier
            else:
                # No change: zero reward (can't plateau!)
                delta_reward = 0.0

            reward += delta_reward
            reward_components['delta_overlap'] = delta_reward

            # Component 2: Sustain bonus (small reward for maintaining high overlap)
            sustain_reward = 0.0
            if overlap_volume > self.sustain_threshold:
                sustain_reward = self.sustain_bonus
            reward += sustain_reward
            reward_components['sustain_bonus'] = sustain_reward

            # Component 3: Bootstrap proximity guide (only during early training AND when no overlap)
            proximity_reward = 0.0
            in_bootstrap = self.total_steps < self.bootstrap_steps

            if in_bootstrap and overlap_volume < 1e-8:  # Essentially zero overlap
                # Small inverse distance reward to guide agent toward target
                clamped_distance = max(distance, 0.03)  # Avoid division by zero
                proximity_reward = self.bootstrap_proximity_scale * (1.0 / clamped_distance - 1.0 / 0.5)
                proximity_reward = max(proximity_reward, 0.0)  # Don't penalize being far

            reward += proximity_reward
            reward_components['proximity'] = proximity_reward

            # Component 4: Distance-scaled contact penalty
            # Penalty ranges from -1 (far) to -5 (close)
            contact_penalty = 0.0
            if num_contacts > 0:
                # Scale penalty based on how close to target
                distance_factor = np.clip(1.0 - (distance / self.contact_distance_scale), 0.0, 1.0)
                # distance_factor = 0 when far (>0.25m), 1 when very close

                penalty_per_contact = self.contact_penalty_min + (self.contact_penalty_max - self.contact_penalty_min) * distance_factor
                # When far: penalty ≈ -1
                # When close: penalty ≈ -5

                contact_penalty = penalty_per_contact * num_contacts

            reward += contact_penalty
            reward_components['contact_penalty'] = contact_penalty

            # Update previous overlap for next step
            self.previous_overlap = overlap_volume

            # Check success (simple: high overlap, no contacts)
            is_success = (overlap_volume > 0.0001 and num_contacts == 0)  # 100 cm³, zero contact
            if is_success:
                self.consecutive_success_steps += 1
            else:
                self.consecutive_success_steps = 0

            # Build info dict (maintain compatibility with V7 logging)
            info = {
                'total_reward': reward,
                'delta_overlap_reward': reward_components['delta_overlap'],
                'sustain_bonus': reward_components['sustain_bonus'],
                'proximity_reward': reward_components['proximity'],
                'contact_penalty': reward_components['contact_penalty'],
                'overlap_volume': overlap_volume,
                'delta_overlap_volume': delta_overlap,
                'hand_hull_volume': hand_volume,
                'object_hull_volume': object_volume,
                'hand_hull_valid': hull_valid,
                'object_hull_valid': hull_valid,
                'distance_to_target': distance,
                'num_contacts': num_contacts,
                'is_success': is_success,
                'consecutive_success_steps': self.consecutive_success_steps,
                'current_phase': 0,  # Always 0 in V7.3
                'bootstrap_active': in_bootstrap,
                'error': '',
                # Legacy fields for WandB compatibility
                'overlap_reward': reward_components['delta_overlap'],  # Map to delta
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
        """Fast bounding-box overlap approximation (V6/V7 method - unchanged)"""
        # DEBUG: Add print for first call
        if not hasattr(self, '_debug_printed'):
            self._debug_printed = False

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

            # DEBUG: Print volume calculation on first successful call
            if not self._debug_printed:
                import sys
                print(f"[DEBUG V7.3 BBOX] First successful call:", flush=True)
                print(f"  Hand bbox: {hand_min} to {hand_max}", flush=True)
                print(f"  Hand dimensions: {hand_max - hand_min}", flush=True)
                print(f"  Hand volume: {hand_volume:.6f} m³ = {hand_volume*1e6:.2f} cm³", flush=True)
                print(f"  Object volume: {obj_volume:.6f} m³ = {obj_volume*1e6:.2f} cm³", flush=True)
                print(f"  MIN_VALID_VOLUME: {self.MIN_VALID_VOLUME:.2e} m³", flush=True)
                sys.stdout.flush()
                self._debug_printed = True

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
            'sustain_bonus': 0.0,
            'proximity_reward': 0.0,
            'contact_penalty': 0.0,
            'overlap_volume': 0.0,
            'delta_overlap_volume': 0.0,
            'hand_hull_volume': 0.0,
            'object_hull_volume': 0.0,
            'hand_hull_valid': False,
            'object_hull_valid': False,
            'distance_to_target': 0.0,
            'num_contacts': 0,
            'is_success': False,
            'consecutive_success_steps': 0,
            'current_phase': 0,
            'bootstrap_active': False,
            'error': error_msg,
            # Legacy fields
            'overlap_reward': 0.0,
            'clearance_reward': 0.0,
            'quality_reward': 0.0,
            'sustained_bonus': 0.0,
        }
