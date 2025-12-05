#!/usr/bin/env python3
"""
V6 Convex Hull Envelopment Reward - Clean Implementation
Uses Half-Space Intersection for reliable overlap calculation.
Minimal logging, maximum stability.
"""

import numpy as np
import gc
from typing import Dict, Tuple
from scipy.spatial import ConvexHull, HalfspaceIntersection
from scipy.optimize import linprog


class V6RewardCalculator:
    """
    3-Phase Curriculum Reward Calculator
    
    Phase 1 (0-300K): Proximity + Low Overlap Signal
        - Primary: Get close to target
        - Secondary: Begin forming valid hull around target
        
    Phase 2 (300K-650K): Maximize Overlap
        - Primary: Maximize hull intersection volume
        - Secondary: Maintain zero contact
        
    Phase 3 (650K-1M): Precision Clearance
        - Primary: Achieve target overlap while maintaining safe clearance
        - Secondary: Sustained success over consecutive steps
    """
    
    def __init__(self, config: Dict = None):
        config = config or {}

        # Object geometry
        self.object_radius = config.get('object_radius', 0.05)  # 5cm sphere
        self.safety_margin = config.get('safety_margin', 0.025)  # 2.5cm clearance
        self.object_hull_points = config.get('object_hull_points', 32)  # High resolution

        # Hull validation threshold (from V5)
        self.MIN_VALID_VOLUME = 1e-9  # 0.001 mm³ - minimum valid hull volume

        # MEMORY FIX: Reduce hull computation frequency from 240Hz to 10Hz
        self.hull_compute_freq = config.get('hull_compute_freq', 24)  # Compute every 24 steps (10Hz at 240Hz sim)
        self.step_counter = 0

        # Cached hull computation results
        self.cached_overlap = 0.0
        self.cached_hand_volume = 0.0
        self.cached_object_volume = 0.0
        self.cached_hull_valid = True
        
        # Pre-compute object hull (static - doesn't change)
        self.object_hull_template = self._generate_sphere_hull_points(
            self.object_radius + self.safety_margin,
            self.object_hull_points
        )
        
        # Curriculum state
        self.current_phase = 0  # Start at Phase 0 for ultra-close learning
        self.consecutive_success_steps = 0
        
        # Phase-specific thresholds (calibrated to realistic volumes)
        # Hand hull ~0.00003 m³ (30 cm³), Object hull ~0.00145 m³ (1450 cm³)
        self.phase_config = {
            0: {  # Phase 0: Ultra-close learning (NEW)
                'overlap_threshold': 0.000001,  # 1 cm³
                'distance_weight': 50.0,        # Very strong proximity reward
                'overlap_weight': 200.0,        # Strong overlap signal
                'contact_penalty': -1.0,        # Gentle - allow exploration
                'quality_weight': 1.0,          # Reduced - focus on approach
                'success_overlap': 0.000003,    # 3 cm³ to advance
                'success_distance': 0.12,       # Must stay within 12cm
            },
            1: {  # Proximity + Low Overlap
                'overlap_threshold': 0.000001,  # 1 cm³ - just detect any overlap
                'distance_weight': 35.0,        # INCREASED - stronger proximity reward
                'overlap_weight': 200.0,        # INCREASED - reward overlap more
                'contact_penalty': -2.0,        # Penalty per contact
                'quality_weight': 2.0,          # Reduced from implicit 5.0
                'success_overlap': 0.000005,    # 5 cm³ to advance
                'success_distance': 0.15,       # Must get within 15cm
            },
            2: {  # Maximize Overlap
                'overlap_threshold': 0.000005,  # 5 cm³ minimum
                'distance_weight': 5.0,         # Reduced - overlap matters more
                'overlap_weight': 500.0,        # Strong overlap reward
                'contact_penalty': -5.0,        # Stronger penalty
                'quality_weight': 3.0,          # Moderate
                'success_overlap': 0.000015,    # 15 cm³ to advance
                'success_distance': 0.12,       # Within 12cm
            },
            3: {  # Precision Clearance
                'overlap_threshold': 0.000010,  # 10 cm³ minimum
                'distance_weight': 2.0,         # Low but present
                'overlap_weight': 300.0,        # Moderate
                'clearance_weight': 200.0,      # New: reward safe distance
                'contact_penalty': -10.0,       # Harsh - no contact allowed
                'quality_weight': 5.0,          # Full quality reward
                'target_clearance': 0.02,       # 2cm from object surface
                'success_overlap': 0.000010,    # Maintain 10 cm³
                'success_consecutive': 50,      # For 50 steps
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
            self.current_phase = min(max(phase, 0), 3)  # Phase 0-3 now
            self.consecutive_success_steps = 0
    
    def reset(self):
        """Reset episode state"""
        self.consecutive_success_steps = 0
        # DO NOT reset step_counter - it's global across all episodes for hull computation frequency
        # self.step_counter = 0  # BUG: This prevented hull computation if episodes < 24 steps
        self.cached_overlap = 0.0
        self.cached_hand_volume = 0.0
        self.cached_object_volume = 0.0
        self.cached_hull_valid = True
    
    def calculate_reward(self, obs: Dict) -> Tuple[float, Dict]:
        """
        Calculate reward based on current phase

        Args:
            obs: Dictionary containing:
                - finger_positions: (4, 3) array of fingertip world positions
                - finger_bases: (4, 3) array of finger base positions
                - palm_position: (3,) array of palm center position
                - object_pos: (3,) array of target position
                - binary_contact: (4,) array of contact flags per finger

        Returns:
            total_reward: float
            info: dict with breakdown
        """
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
            
            # Build 9-point hand hull: 4 fingertips + 4 finger bases + 1 palm
            hand_points = np.vstack([
                finger_positions,  # 4 fingertips
                finger_bases,      # 4 finger bases
                palm_position.reshape(1, 3)  # Palm center
            ])
            
            # Translate object hull template to current object position
            object_points = self.object_hull_template + object_pos

            # MEMORY FIX: Only compute hulls every N steps (10Hz instead of 240Hz)
            self.step_counter += 1
            should_compute_hulls = (self.step_counter % self.hull_compute_freq == 0)

            # DEBUG: Print step counter on first few calls
            if not hasattr(self, '_step_debug_count'):
                self._step_debug_count = 0
            if self._step_debug_count < 5:
                import sys
                print(f"[DEBUG STEP] step_counter={self.step_counter}, hull_freq={self.hull_compute_freq}, "
                      f"modulo={self.step_counter % self.hull_compute_freq}, should_compute={should_compute_hulls}", flush=True)
                sys.stdout.flush()
                self._step_debug_count += 1

            # DEBUG: Print on first hull computation
            if should_compute_hulls and not hasattr(self, '_first_hull_compute'):
                import sys
                print(f"[DEBUG] First hull computation at step {self.step_counter}, freq={self.hull_compute_freq}")
                sys.stdout.flush()
                self._first_hull_compute = True

            if should_compute_hulls:
                # Compute hulls at reduced frequency using FAST bounding-box method
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
                    print(f"\n[DEBUG Step {self.step_counter}] Hull computed:")
                    print(f"  Hand volume: {hand_volume*1e6:.2f} cm³")
                    print(f"  Object volume: {object_volume*1e6:.2f} cm³")
                    print(f"  Overlap volume: {overlap_volume*1e6:.2f} cm³")
                    print(f"  Hull valid: {hull_valid}")
                    print(f"  Hand points shape: {hand_points.shape}")
                    print(f"  Object points shape: {object_points.shape}\n")

                # Aggressive garbage collection after hull computation
                gc.collect()
            else:
                # Use cached values
                overlap_volume = self.cached_overlap
                hand_volume = self.cached_hand_volume
                object_volume = self.cached_object_volume
                hull_valid = self.cached_hull_valid

            if not hull_valid:
                return 0.0, self._error_info("Invalid hull geometry")
            
            # Calculate distance (hand center to object)
            hand_center = np.mean(hand_points, axis=0)
            distance = np.linalg.norm(hand_center - object_pos)
            
            # Count contacts
            num_contacts = int(np.sum(binary_contact))
            
            # Get phase config
            cfg = self.phase_config[self.current_phase]
            
            # Calculate reward components based on phase
            reward_components = self._calculate_phase_rewards(
                distance, overlap_volume, num_contacts, cfg, hand_points, object_pos
            )
            
            # Check success and update consecutive counter
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
    
    def validate_hull(self, hull_points: np.ndarray, name: str = "hull") -> Tuple[bool, float, str]:
        """
        Validate hull by simply trying to create it.
        If ConvexHull succeeds with non-zero volume, it's valid.

        The previous implementation had overly strict collinearity/coplanarity checks
        that failed on valid hulls due to point ordering from np.unique.

        Returns:
            is_valid: bool
            volume: float (0.0 if invalid)
            error_msg: str (empty if valid)
        """
        hull = None
        try:
            # Check minimum points
            if len(hull_points) < 4:
                return False, 0.0, f"{name}: Need at least 4 points, got {len(hull_points)}"

            # Check for NaN/Inf
            if np.any(np.isnan(hull_points)) or np.any(np.isinf(hull_points)):
                return False, 0.0, f"{name}: Contains NaN or Inf values"

            # Just try to create the hull - this is the definitive test
            # ConvexHull internally handles degeneracy checks
            hull = ConvexHull(hull_points)
            volume = hull.volume

            # Check minimum volume threshold
            if volume < self.MIN_VALID_VOLUME:
                return False, 0.0, f"{name}: Volume too small ({volume:.2e} m³)"

            return True, volume, ""

        except Exception as e:
            return False, 0.0, f"{name}: ConvexHull failed - {str(e)}"

        finally:
            # Explicit cleanup - validation creates many temporary hulls
            del hull

    def _calculate_overlap_bbox_fast(self, hand_points: np.ndarray,
                                       object_points: np.ndarray) -> Tuple[float, float, float, bool]:
        """
        FAST bounding-box overlap approximation - NO ConvexHull objects created.
        Uses pure numpy for zero memory overhead.

        This is ~100x faster than HSI and good enough for RL reward signal.
        """
        # DEBUG: Add print for first call
        if not hasattr(self, '_debug_printed'):
            self._debug_printed = False

        try:
            # Basic validation without creating ConvexHull
            if len(hand_points) < 4 or len(object_points) < 4:
                if not self._debug_printed:
                    print(f"[DEBUG BBOX] FAILED: Not enough points - hand:{len(hand_points)}, obj:{len(object_points)}")
                    self._debug_printed = True
                return 0.0, 0.0, 0.0, False

            if np.any(np.isnan(hand_points)) or np.any(np.isinf(hand_points)):
                if not self._debug_printed:
                    print(f"[DEBUG BBOX] FAILED: Hand points contain NaN/Inf")
                    self._debug_printed = True
                return 0.0, 0.0, 0.0, False

            if np.any(np.isnan(object_points)) or np.any(np.isinf(object_points)):
                if not self._debug_printed:
                    print(f"[DEBUG BBOX] FAILED: Object points contain NaN/Inf")
                    self._debug_printed = True
                return 0.0, 0.0, 0.0, False

            # Compute bounding boxes (pure numpy - no scipy)
            hand_min = np.min(hand_points, axis=0)
            hand_max = np.max(hand_points, axis=0)
            obj_min = np.min(object_points, axis=0)
            obj_max = np.max(object_points, axis=0)

            # Approximate volumes from bounding boxes (no ConvexHull needed)
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
                print(f"[DEBUG BBOX] FAILED volume check: hand={hand_volume:.2e}, obj={obj_volume:.2e}, min={self.MIN_VALID_VOLUME:.2e}")
                return 0.0, 0.0, 0.0, False

            # Compute intersection of bounding boxes
            intersection_min = np.maximum(hand_min, obj_min)
            intersection_max = np.minimum(hand_max, obj_max)

            # Check if there's any intersection
            if np.any(intersection_min >= intersection_max):
                return 0.0, hand_volume, obj_volume, True

            # Compute overlap volume (30% of bbox intersection)
            overlap_volume = np.prod(intersection_max - intersection_min) * 0.3

            return overlap_volume, hand_volume, obj_volume, True

        except Exception as e:
            print(f"[DEBUG BBOX] EXCEPTION: {str(e)}")
            return 0.0, 0.0, 0.0, False

    def _calculate_overlap_hsi(self, hand_points: np.ndarray,
                                object_points: np.ndarray) -> Tuple[float, float, float, bool]:
        """
        Calculate overlap using Half-Space Intersection method.
        Much more reliable than Trimesh for small convex hulls.
        """
        hand_hull = None
        object_hull = None
        overlap_hull = None
        hsi = None

        try:
            # Validate hand hull with comprehensive checks
            hand_valid, hand_volume, hand_error = self.validate_hull(hand_points, "hand_hull")
            if not hand_valid:
                return 0.0, 0.0, 0.0, False

            # Validate object hull
            obj_valid, obj_volume, obj_error = self.validate_hull(object_points, "object_hull")
            if not obj_valid:
                return 0.0, hand_volume, 0.0, False

            # Build convex hulls (already validated, but need hull objects for equations)
            hand_hull = ConvexHull(hand_points)
            object_hull = ConvexHull(object_points)

            # Get half-space equations from both hulls
            # Each equation is [A, B, C, D] where Ax + By + Cz + D <= 0
            hand_equations = hand_hull.equations
            object_equations = object_hull.equations

            # Combine all half-spaces
            all_halfspaces = np.vstack([hand_equations, object_equations])

            # Find interior point using linear programming
            interior_point = self._find_interior_point(all_halfspaces)

            if interior_point is None:
                # No intersection exists
                return 0.0, hand_volume, obj_volume, True

            # Compute half-space intersection
            try:
                hsi = HalfspaceIntersection(all_halfspaces, interior_point)

                if len(hsi.intersections) < 4:
                    return 0.0, hand_volume, obj_volume, True

                # Compute volume of intersection
                overlap_hull = ConvexHull(hsi.intersections)
                overlap_volume = overlap_hull.volume

                return overlap_volume, hand_volume, obj_volume, True

            except Exception as e:
                # HSI failed but hulls are valid - return zero overlap
                return 0.0, hand_volume, obj_volume, True

        except Exception as e:
            # Unexpected error in overlap calculation
            return 0.0, 0.0, 0.0, False

        finally:
            # CRITICAL FIX: Explicitly delete large objects to help garbage collector
            # ConvexHull objects are created at 240Hz and can cause memory fragmentation
            del hand_hull, object_hull, overlap_hull, hsi
    
    def _find_interior_point(self, halfspaces: np.ndarray) -> np.ndarray:
        """
        Find a point strictly inside all half-spaces using linear programming.
        Uses Chebyshev center approach.
        """
        n_halfspaces = len(halfspaces)
        n_dims = halfspaces.shape[1] - 1  # 3D
        
        # Chebyshev center: maximize radius r such that point is r away from all boundaries
        # minimize -r
        # subject to: A_i @ x + b_i + ||A_i|| * r <= 0
        
        # Normalize half-space normals
        normals = halfspaces[:, :-1]
        offsets = halfspaces[:, -1]
        norms = np.linalg.norm(normals, axis=1, keepdims=True)
        norms = np.where(norms < 1e-10, 1.0, norms)
        
        # Build LP: variables are [x, y, z, r]
        c = np.zeros(n_dims + 1)
        c[-1] = -1  # Maximize r
        
        A_ub = np.zeros((n_halfspaces, n_dims + 1))
        A_ub[:, :n_dims] = normals
        A_ub[:, -1] = norms.flatten()
        b_ub = -offsets
        
        try:
            result = linprog(c, A_ub=A_ub, b_ub=b_ub, method='highs')
            
            if result.success and result.x[-1] > 1e-8:
                return result.x[:n_dims]
            return None
        except:
            return None
    
    def _calculate_phase_rewards(self, distance: float, overlap: float,
                                  contacts: int, cfg: Dict,
                                  hand_points: np.ndarray, object_pos: np.ndarray) -> Dict:
        """Calculate reward components based on current phase"""
        rewards = {}

        # Proximity reward (all phases, weighted differently)
        # TUNED: Slower decay rate (-2.0 instead of -5.0) for stronger gradient
        # At 0.2m: exp(-0.4) = 0.67 → reward = 50 * 0.67 = 33.5 (Phase 0)
        # At 0.5m: exp(-1.0) = 0.37 → reward = 50 * 0.37 = 18.5 (Phase 0)
        # At 1.0m: exp(-2.0) = 0.14 → reward = 50 * 0.14 = 7.0  (Phase 0)
        proximity = cfg['distance_weight'] * np.exp(-2.0 * distance)
        rewards['proximity'] = proximity

        # Overlap reward (all phases)
        # Scale overlap to reasonable range
        overlap_cm3 = overlap * 1e6  # Convert to cm³
        overlap_reward = cfg['overlap_weight'] * np.tanh(overlap_cm3 / 50.0)  # Saturates around 50 cm³
        rewards['overlap'] = overlap_reward

        # Contact penalty (all phases)
        if contacts > 0:
            rewards['contact'] = cfg['contact_penalty'] * contacts
        else:
            rewards['contact'] = 0.0

        # Phase 3: Clearance reward
        if self.current_phase == 3 and 'clearance_weight' in cfg:
            # Calculate minimum distance from any hand point to object surface
            hand_to_obj_distances = np.linalg.norm(hand_points - object_pos, axis=1)
            min_clearance = np.min(hand_to_obj_distances) - self.object_radius

            target_clearance = cfg['target_clearance']

            # Reward being close to target clearance (not too close, not too far)
            clearance_error = abs(min_clearance - target_clearance)
            clearance_reward = cfg['clearance_weight'] * np.exp(-10.0 * clearance_error)
            rewards['clearance'] = clearance_reward

        # Quality bonus: reward well-formed hull (spread out fingers)
        # TUNED: Now configurable per phase via quality_weight
        finger_spread = np.std(hand_points[:4], axis=0).mean()  # Fingertip spread
        quality_weight = cfg.get('quality_weight', 5.0)  # Default 5.0 for backward compatibility
        quality = quality_weight * np.clip(finger_spread / 0.05, 0, 1)  # Max at 5cm spread
        rewards['quality'] = quality

        return rewards
    
    def _check_success(self, overlap: float, contacts: int, cfg: Dict, distance: float = None) -> bool:
        """Check if current state meets success criteria"""
        if contacts > 0:
            return False

        # Check overlap threshold
        overlap_ok = overlap >= cfg.get('success_overlap', 0.000001)

        # Check distance threshold if specified
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
