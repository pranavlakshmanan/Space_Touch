#!/usr/bin/env python3
"""
V7 Reward Calculator - Improved 3-Phase Curriculum
Based on V6 with cleaner phase progression and skill isolation.

Key improvements over V6:
- 3 clear phases (+ bootstrap Phase 0) vs V6's 4 overlapping phases
- Phase-specific skill focus: Navigation → Envelopment → Precision
- Removes proximity in Phase 2+ to force independent overlap learning
- Keeps all V6 bug fixes (no step_counter reset, memory optimization)
"""

import numpy as np
import gc
from typing import Dict, Tuple


class V7RewardCalculator:
    """
    4-Phase Curriculum (0-3):

    Phase 0: Bootstrap (0-30K steps)
        - Ultra-close starting position (5cm)
        - Learn basic movement controls
        - High proximity, low overlap introduction

    Phase 1: Approach (30K-90K steps)
        - PRIMARY: Proximity reward (get close)
        - SECONDARY: Overlap awareness (low weight)
        - Goal: Master navigation to target

    Phase 2: Envelopment (90K-160K steps)
        - PRIMARY: Overlap reward (maximize hull intersection)
        - NO proximity (force pure overlap learning)
        - Goal: Learn finger coordination for envelopment

    Phase 3: Precision (160K-200K steps)
        - PRIMARY: Overlap + Clearance balance
        - HARSH contact penalty
        - Goal: Maintain overlap without touching
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
        self.hull_compute_freq = config.get('hull_compute_freq', 24)  # 10Hz at 240Hz sim
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

        # Curriculum state
        self.current_phase = 0
        self.consecutive_success_steps = 0

        # V7 Phase Configuration - Clean skill progression
        self.phase_config = {
            0: {  # Bootstrap: Ultra-close learning
                'overlap_threshold': 0.000001,   # 1 cm³
                'proximity_weight': 50.0,        # Strong approach signal
                'overlap_weight': 20.0,          # Introduce concept
                'contact_penalty': -1.0,         # Gentle - allow exploration
                'quality_weight': 1.0,           # Minimal
                'success_overlap': 0.00001,      # 10 cm³
                'success_distance': 0.15,        # Within 15cm
            },
            1: {  # Phase 1: APPROACH - Focus on navigation
                'overlap_threshold': 0.000001,   # 1 cm³
                'proximity_weight': 50.0,        # PRIMARY objective
                'overlap_weight': 20.0,          # SECONDARY - awareness only
                'contact_penalty': -2.0,         # Moderate
                'quality_weight': 1.0,           # Minimal
                'success_overlap': 0.00001,      # 10 cm³ (just touching)
                'success_distance': 0.15,        # Mean < 15cm
            },
            2: {  # Phase 2: ENVELOPMENT - Pure overlap learning
                'overlap_threshold': 0.00001,    # 10 cm³ minimum
                'proximity_weight': 0.0,         # REMOVED - force pure overlap
                'overlap_weight': 500.0,         # SOLE primary objective
                'contact_penalty': -5.0,         # Moderate - some contact ok
                'quality_weight': 3.0,           # Reward finger spread
                'success_overlap': 0.0001,       # 100 cm³
                'success_distance': 0.20,        # Within 20cm (termination prevents >25cm)
            },
            3: {  # Phase 3: PRECISION - Balance overlap + no contact
                'overlap_threshold': 0.00001,    # 10 cm³ minimum
                'proximity_weight': 0.0,         # Still removed
                'overlap_weight': 300.0,         # High
                'clearance_weight': 200.0,       # NEW - reward optimal distance
                'contact_penalty': -20.0,        # HARSH - strong deterrent
                'quality_weight': 5.0,           # Full weight
                'target_clearance': 0.02,        # 2cm from surface
                'success_overlap': 0.00015,      # 150 cm³
                'success_consecutive': 50,       # For 50 steps
            },
        }

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
        """Update curriculum phase"""
        if phase != self.current_phase:
            self.current_phase = min(max(phase, 0), 3)
            self.consecutive_success_steps = 0

    def reset(self):
        """Reset episode state"""
        self.consecutive_success_steps = 0
        # CRITICAL: Do NOT reset step_counter (V6 bug fix)
        # step_counter is global across episodes for hull computation frequency
        self.cached_overlap = 0.0
        self.cached_hand_volume = 0.0
        self.cached_object_volume = 0.0
        self.cached_hull_valid = True

    def calculate_reward(self, obs: Dict) -> Tuple[float, Dict]:
        """Calculate reward based on current phase"""
        # DEBUG: Print on first call
        if not hasattr(self, '_reward_calc_called'):
            import sys
            print("[DEBUG] calculate_reward() called for first time", flush=True)
            sys.stdout.flush()
            self._reward_calc_called = True

        try:
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

            # Memory optimization: Compute hulls at reduced frequency
            self.step_counter += 1
            should_compute_hulls = (self.step_counter % self.hull_compute_freq == 0)

            # DEBUG: Print on first hull computation
            if should_compute_hulls and not hasattr(self, '_first_hull_compute'):
                import sys
                print(f"[DEBUG] First hull computation at step {self.step_counter}, freq={self.hull_compute_freq}", flush=True)
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
                    print(f"\n[DEBUG Step {self.step_counter}] Hull computed:", flush=True)
                    print(f"  Hand volume: {hand_volume*1e6:.2f} cm³", flush=True)
                    print(f"  Object volume: {object_volume*1e6:.2f} cm³", flush=True)
                    print(f"  Overlap volume: {overlap_volume*1e6:.2f} cm³", flush=True)
                    print(f"  Hull valid: {hull_valid}", flush=True)
                    print(f"  Hand points shape: {hand_points.shape}", flush=True)
                    print(f"  Object points shape: {object_points.shape}\n", flush=True)
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

            # Get phase config
            cfg = self.phase_config[self.current_phase]

            # Calculate phase-specific rewards
            reward_components = self._calculate_phase_rewards(
                distance, overlap_volume, num_contacts, cfg, hand_points, object_pos
            )

            # Check success
            is_success = self._check_success(overlap_volume, num_contacts, cfg, distance)
            if is_success:
                self.consecutive_success_steps += 1
            else:
                self.consecutive_success_steps = 0

            # Sustained success bonus (Phase 3)
            sustained_bonus = 0.0
            if self.current_phase == 3 and self.consecutive_success_steps >= cfg.get('success_consecutive', 50):
                sustained_bonus = 50.0

            # Total reward
            total_reward = sum(reward_components.values()) + sustained_bonus

            # Build info dict
            info = {
                'overlap_reward': reward_components.get('overlap', 0.0),
                'proximity_reward': reward_components.get('proximity', 0.0),
                'contact_penalty': reward_components.get('contact', 0.0),
                'clearance_reward': reward_components.get('clearance', 0.0),
                'quality_reward': reward_components.get('quality', 0.0),
                'sustained_bonus': sustained_bonus,
                'overlap_volume': overlap_volume,
                'hand_hull_volume': hand_volume,
                'object_hull_volume': object_volume,
                'hand_hull_valid': hull_valid,
                'object_hull_valid': hull_valid,
                'distance_to_target': distance,
                'num_contacts': num_contacts,
                'is_success': is_success,
                'consecutive_success_steps': self.consecutive_success_steps,
                'current_phase': self.current_phase,
                'error': '',
            }

            return total_reward, info

        except Exception as e:
            return 0.0, self._error_info(str(e))

    def _calculate_overlap_bbox_fast(self, hand_points: np.ndarray,
                                       object_points: np.ndarray) -> Tuple[float, float, float, bool]:
        """Fast bounding-box overlap approximation (V6 method)"""
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
                print(f"[DEBUG BBOX] First successful call:", flush=True)
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

    def _calculate_phase_rewards(self, distance: float, overlap: float,
                                  contacts: int, cfg: Dict,
                                  hand_points: np.ndarray, object_pos: np.ndarray) -> Dict:
        """Calculate reward components based on current phase"""
        rewards = {}

        # Proximity reward (Phases 0-1 only, removed in Phases 2-3)
        proximity_weight = cfg.get('proximity_weight', 0.0)
        if proximity_weight > 0:
            proximity = proximity_weight * np.exp(-2.0 * distance)
            rewards['proximity'] = proximity
        else:
            rewards['proximity'] = 0.0

        # Overlap reward (all phases, weight varies)
        overlap_cm3 = overlap * 1e6
        overlap_reward = cfg['overlap_weight'] * np.tanh(overlap_cm3 / 50.0)
        rewards['overlap'] = overlap_reward

        # Contact penalty (all phases)
        if contacts > 0:
            rewards['contact'] = cfg['contact_penalty'] * contacts
        else:
            rewards['contact'] = 0.0

        # Clearance reward (Phase 3 only)
        if self.current_phase == 3 and 'clearance_weight' in cfg:
            hand_to_obj_distances = np.linalg.norm(hand_points - object_pos, axis=1)
            min_clearance = np.min(hand_to_obj_distances) - self.object_radius
            target_clearance = cfg['target_clearance']
            clearance_error = abs(min_clearance - target_clearance)
            clearance_reward = cfg['clearance_weight'] * np.exp(-10.0 * clearance_error)
            rewards['clearance'] = clearance_reward
        else:
            rewards['clearance'] = 0.0

        # Quality bonus: Reward finger spread (all phases)
        finger_spread = np.std(hand_points[:4], axis=0).mean()
        quality_weight = cfg.get('quality_weight', 5.0)
        quality = quality_weight * np.clip(finger_spread / 0.05, 0, 1)
        rewards['quality'] = quality

        return rewards

    def _check_success(self, overlap: float, contacts: int, cfg: Dict, distance: float = None) -> bool:
        """Check if current state meets success criteria"""
        if contacts > 0:
            return False

        overlap_ok = overlap >= cfg.get('success_overlap', 0.000001)

        if 'success_distance' in cfg and distance is not None:
            distance_ok = distance <= cfg['success_distance']
            return overlap_ok and distance_ok

        return overlap_ok

    def _error_info(self, error_msg: str) -> Dict:
        """Return info dict for error cases"""
        return {
            'overlap_reward': 0.0,
            'proximity_reward': 0.0,
            'contact_penalty': 0.0,
            'clearance_reward': 0.0,
            'quality_reward': 0.0,
            'sustained_bonus': 0.0,
            'overlap_volume': 0.0,
            'hand_hull_volume': 0.0,
            'object_hull_volume': 0.0,
            'hand_hull_valid': False,
            'object_hull_valid': False,
            'distance_to_target': 0.0,
            'num_contacts': 0,
            'is_success': False,
            'consecutive_success_steps': 0,
            'current_phase': self.current_phase,
            'error': error_msg,
        }
