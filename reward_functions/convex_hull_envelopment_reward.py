#!/usr/bin/env python3
"""
Convex Hull Envelopment Reward for 3-Phase Curriculum Soft-Capture
Phase 1: Approach & Form Valid Hull (non-zero overlap)
Phase 2: Envelopment (maximize overlap, light contact penalty)
Phase 3: Precision (minimize clearance error, heavy contact penalty)
"""

import numpy as np
from typing import Dict, Tuple
from scipy.spatial import ConvexHull, Delaunay
import trimesh


class ConvexHullEnvelopmentReward:
    """
    3-Phase curriculum reward for soft-capture via spatial containment
    """

    def __init__(self, config: Dict = None):
        if config is None:
            config = {}

        # Object properties
        self.OBJECT_RADIUS = config.get('object_radius', 0.05)  # 5cm sphere
        self.SAFETY_MARGIN = config.get('safety_margin', 0.025)  # 2.5cm clearance

        # Hull generation - INCREASED from 12 to 32 for better sphere approximation
        self.OBJECT_HULL_POINTS = config.get('object_hull_points', 32)  # Improves from 55% to 93% sphere accuracy

        # Validation thresholds - RELAXED to allow smaller hulls
        self.MIN_VALID_VOLUME = 1e-9  # 0.001 mm³ - much more permissive

        # Curriculum phase (will be updated externally)
        self.current_phase = 1

        # Phase-specific reward weights
        self.phase_configs = {
            1: {  # APPROACH & FORMATION
                'overlap_scale': 1000.0,
                'contact_penalty': 0.0,      # No penalty yet
                'proximity_scale': 20.0,     # MUCH stronger proximity reward
                'quality_scale': 0.0,
            },
            2: {  # ENVELOPMENT
                'overlap_scale': 5000.0,
                'contact_penalty': -2.0,     # Light penalty
                'proximity_scale': 0.0,      # Remove - overlap is enough
                'quality_scale': 1.0,        # Encourage good hand shape
                'size_penalty_scale': 3.0,   # Penalize overspreading
            },
            3: {  # PRECISION
                'overlap_scale': 10000.0,
                'contact_penalty': -10.0,    # STRONG penalty
                'proximity_scale': 0.0,
                'quality_scale': 2.0,
                'clearance_scale': 50.0,     # New: minimize clearance error
            }
        }

        # Tracking
        self.consecutive_success_steps = 0

        print("🔧 Initialized ConvexHullEnvelopmentReward:")
        print(f"   Object radius: {self.OBJECT_RADIUS}m")
        print(f"   Safety margin: {self.SAFETY_MARGIN}m")
        print(f"   Starting phase: {self.current_phase}")

    def generate_object_hull_points(self, object_pos: np.ndarray) -> np.ndarray:
        """
        Generate convex hull points around spherical object with safety margin
        Uses Fibonacci sphere for uniform distribution
        """
        points = []
        n = self.OBJECT_HULL_POINTS
        golden_angle = np.pi * (3.0 - np.sqrt(5.0))

        for i in range(n):
            y = 1 - (i / float(n - 1)) * 2
            radius_at_y = np.sqrt(1 - y * y)
            theta = golden_angle * i

            x = np.cos(theta) * radius_at_y
            z = np.sin(theta) * radius_at_y

            # Scale by object radius + safety margin
            hull_radius = self.OBJECT_RADIUS + self.SAFETY_MARGIN
            point = object_pos + hull_radius * np.array([x, y, z])
            points.append(point)

        return np.array(points)

    def validate_hull(self, hull_points: np.ndarray, name: str = "hull") -> Tuple[bool, float, str]:
        """
        Validate that hull is not degenerate (zero volume, planar, or linear)

        Returns:
            is_valid: bool
            volume: float (0.0 if invalid)
            error_msg: str (empty if valid)
        """
        try:
            # Check minimum points
            if len(hull_points) < 4:
                return False, 0.0, f"{name}: Need at least 4 points for 3D hull, got {len(hull_points)}"

            # Check for duplicate points
            unique_points = np.unique(hull_points, axis=0)
            if len(unique_points) < len(hull_points):
                return False, 0.0, f"{name}: Contains duplicate points"

            # Check if points are collinear (1D)
            if len(unique_points) >= 2:
                v1 = unique_points[1] - unique_points[0]
                v1 = v1 / (np.linalg.norm(v1) + 1e-10)

                collinear = True
                for i in range(2, len(unique_points)):
                    v2 = unique_points[i] - unique_points[0]
                    v2 = v2 / (np.linalg.norm(v2) + 1e-10)
                    cross = np.cross(v1, v2)
                    if np.linalg.norm(cross) > 1e-6:
                        collinear = False
                        break

                if collinear:
                    return False, 0.0, f"{name}: Points are collinear (1D)"

            # Check if points are coplanar (2D)
            if len(unique_points) >= 4:
                v1 = unique_points[1] - unique_points[0]
                v2 = unique_points[2] - unique_points[0]
                normal = np.cross(v1, v2)
                normal = normal / (np.linalg.norm(normal) + 1e-10)

                coplanar = True
                for i in range(3, len(unique_points)):
                    v3 = unique_points[i] - unique_points[0]
                    distance = abs(np.dot(v3, normal))
                    if distance > 1e-6:
                        coplanar = False
                        break

                if coplanar:
                    return False, 0.0, f"{name}: Points are coplanar (2D)"

            # Try to create hull
            hull = ConvexHull(unique_points)
            volume = hull.volume

            # Check if volume is too small (essentially degenerate)
            if volume < self.MIN_VALID_VOLUME:
                return False, 0.0, f"{name}: Volume too small ({volume:.2e} m³ < {self.MIN_VALID_VOLUME:.2e} m³)"

            return True, volume, ""

        except Exception as e:
            return False, 0.0, f"{name}: Hull creation failed - {str(e)}"

    def calculate_overlap_volume(self, hand_hull_points: np.ndarray,
                                 object_hull_points: np.ndarray) -> Tuple[float, Dict]:
        """
        Calculate intersection volume between two convex hulls
        Returns overlap volume and diagnostic info
        """
        # DEBUG: Track zero volume calculations
        if not hasattr(self, 'zero_debug_count'):
            self.zero_debug_count = 0
        # Validate hand hull
        hand_valid, hand_volume, hand_error = self.validate_hull(hand_hull_points, "hand_hull")
        if not hand_valid:
            return 0.0, {
                'error': hand_error,
                'hand_hull_volume': 0.0,
                'object_hull_volume': 0.0,
                'hand_hull_valid': False,
            }

        # Validate object hull
        obj_valid, obj_volume, obj_error = self.validate_hull(object_hull_points, "object_hull")
        if not obj_valid:
            return 0.0, {
                'error': obj_error,
                'hand_hull_volume': hand_volume,
                'object_hull_volume': 0.0,
                'object_hull_valid': False,
            }

        # Both hulls valid - calculate intersection using Half-Space Intersection
        try:
            from scipy.spatial import HalfspaceIntersection

            # Check for extreme distances before hull calculation
            hand_center = np.mean(hand_hull_points, axis=0)
            object_center = np.mean(object_hull_points, axis=0)
            separation = np.linalg.norm(hand_center - object_center)

            if separation > 10.0:  # More than 10m separation - no intersection possible
                return 0.0, {
                    'error': f'Extreme separation: {separation:.3f}m',
                    'hand_hull_volume': hand_volume,
                    'object_hull_volume': obj_volume,
                    'hand_hull_valid': True,
                    'object_hull_valid': True,
                }

            # Create convex hulls
            hand_hull = ConvexHull(hand_hull_points)
            object_hull = ConvexHull(object_hull_points)

            # Get half-space equations (format: Ax + By + Cz + D <= 0)
            # Each row is [A, B, C, D] defining one face of the convex hull
            hand_halfspaces = hand_hull.equations
            object_halfspaces = object_hull.equations

            # Combine constraints: intersection must satisfy BOTH sets of inequalities
            combined_halfspaces = np.vstack([hand_halfspaces, object_halfspaces])

            # Find interior point (must be inside BOTH hulls for HSI to work)
            interior_point = (hand_center + object_center) / 2.0

            # Helper function to check if point satisfies all half-space constraints
            def point_in_hull(point, hull_equations):
                """Check if point satisfies all half-space constraints"""
                return np.all(hull_equations[:, :-1] @ point + hull_equations[:, -1] <= 1e-6)

            # If midpoint not inside both hulls, they don't overlap
            if not (point_in_hull(interior_point, hand_halfspaces) and
                    point_in_hull(interior_point, object_halfspaces)):
                # No overlap - hulls don't intersect
                overlap_volume = 0.0

                # DEBUG: Log separation distance for non-overlapping hulls
                if self.zero_debug_count % 100 == 1:
                    print(f"\n🔍 NO OVERLAP - SEPARATION ANALYSIS (#{self.zero_debug_count}):")
                    print(f"   Hand center: {hand_center}")
                    print(f"   Object center: {object_center}")
                    print(f"   Separation distance: {separation:.6f}m ({separation*1000:.1f}mm)")
                    print(f"   Interior point: {interior_point}")
                    print(f"   Interior in hand hull: {point_in_hull(interior_point, hand_halfspaces)}")
                    print(f"   Interior in object hull: {point_in_hull(interior_point, object_halfspaces)}")

            else:
                # Compute half-space intersection
                try:
                    hsi = HalfspaceIntersection(combined_halfspaces, interior_point)

                    # Check if we have enough intersection points for 3D volume
                    if len(hsi.intersections) < 4:
                        overlap_volume = 0.0
                    else:
                        # Compute volume of intersection region
                        try:
                            overlap_hull = ConvexHull(hsi.intersections)
                            overlap_volume = float(overlap_hull.volume)

                            # SUCCESS: Log successful overlap calculation
                            if overlap_volume > 0:
                                print(f"\n🎯 OVERLAP DETECTED! Volume: {overlap_volume:.9f} m³ ({overlap_volume*1e6:.4f} cm³)")
                                print(f"   Separation: {separation:.3f}m, HSI points: {len(hsi.intersections)}")

                        except Exception as hull_error:
                            overlap_volume = 0.0

                except Exception as hsi_error:
                    # Half-space intersection failed - no overlap
                    overlap_volume = 0.0

            # DEBUG: Track calculations (reduced spam)
            if overlap_volume == 0.0:
                self.zero_debug_count += 1
                if self.zero_debug_count % 1000 == 1:  # Every 1000 zero volumes (reduced from 500)
                    print(f"\n🔍 HSI CALCULATION SUMMARY (#{self.zero_debug_count}):")
                    print(f"   Hand hull volume: {hand_volume:.9f} m³ ({hand_volume*1e6:.4f} cm³)")
                    print(f"   Object hull volume: {obj_volume:.9f} m³ ({obj_volume*1e6:.4f} cm³)")
                    print(f"   Separation distance: {separation:.6f}m ({separation*1000:.1f}mm)")
                    print(f"   Method: Half-Space Intersection (replacing broken Trimesh)")

            return overlap_volume, {
                'hand_hull_volume': hand_volume,
                'object_hull_volume': obj_volume,
                'hand_hull_valid': True,
                'object_hull_valid': True,
                'error': '',
            }

        except Exception as e:
            return 0.0, {
                'error': f"Intersection calculation failed: {str(e)}",
                'hand_hull_volume': hand_volume,
                'object_hull_volume': obj_volume,
            }

    def _approximate_overlap(self, hull1_points: np.ndarray, hull2_points: np.ndarray) -> float:
        """Fallback approximation when exact intersection fails"""
        hull1_center = np.mean(hull1_points, axis=0)
        hull2_center = np.mean(hull2_points, axis=0)

        dist_between_centers = np.linalg.norm(hull1_center - hull2_center)
        hull1_radius = np.max(np.linalg.norm(hull1_points - hull1_center, axis=1))
        hull2_radius = np.max(np.linalg.norm(hull2_points - hull2_center, axis=1))

        if dist_between_centers > (hull1_radius + hull2_radius):
            return 0.0

        overlap_ratio = max(0, 1 - dist_between_centers / (hull1_radius + hull2_radius))
        estimated_volume = overlap_ratio * min(
            ConvexHull(hull1_points).volume,
            ConvexHull(hull2_points).volume
        ) * 0.1

        return estimated_volume

    def calculate_proximity_reward(self, hand_hull_points: np.ndarray,
                                   object_pos: np.ndarray) -> float:
        """Reward for getting hand close to object (Phase 1 only)"""
        hand_center = np.mean(hand_hull_points, axis=0)
        distance = np.linalg.norm(hand_center - object_pos)

        # Strong linear + exponential reward for approach
        # Linear component for long-range guidance
        linear_reward = max(0, 1.0 - distance / 0.5)  # 1.0 reward within 50cm, 0 at 50cm+

        # Exponential component for close-range precision
        exp_reward = np.exp(-3.0 * max(0, distance - self.SAFETY_MARGIN))

        # Combined reward with stronger signal
        proximity_reward = linear_reward + exp_reward
        return proximity_reward

    def calculate_quality_reward(self, hand_hull_volume: float,
                                 object_hull_volume: float) -> float:
        """Reward for reasonable hand shape (not too spread, not too compact)"""
        if hand_hull_volume < self.MIN_VALID_VOLUME:
            return 0.0

        # Ideal hand hull: 1.5x - 3x object hull size
        ideal_min = object_hull_volume * 1.5
        ideal_max = object_hull_volume * 3.0

        if ideal_min <= hand_hull_volume <= ideal_max:
            # In ideal range
            return 1.0
        elif hand_hull_volume < ideal_min:
            # Too compact
            ratio = hand_hull_volume / ideal_min
            return max(0, ratio)  # 0 to 1
        else:
            # Too spread
            penalty = (hand_hull_volume - ideal_max) / ideal_max
            return max(0, 1.0 - penalty)

    def calculate_clearance_error(self, hand_hull_points: np.ndarray,
                                  object_hull_points: np.ndarray) -> float:
        """
        Calculate how far hand is from ideal clearance distance
        Phase 3 only: want to be exactly SAFETY_MARGIN away
        """
        # Calculate minimum distance between hulls
        from scipy.spatial.distance import cdist
        distances = cdist(hand_hull_points, object_hull_points)
        min_distance = np.min(distances)

        # Error = deviation from ideal safety margin
        clearance_error = abs(min_distance - self.SAFETY_MARGIN)

        return clearance_error

    def calculate_reward(self, obs_dict: Dict) -> Tuple[float, Dict]:
        """
        Calculate total reward based on current curriculum phase

        Args:
            obs_dict: Must contain:
                - 'finger_positions': (4, 3) array
                - 'palm_position': (3,) array
                - 'object_pos': (3,) array
                - 'binary_contact': (4,) array
                - 'episode_step': int

        Returns:
            total_reward: float
            reward_info: Dict with detailed breakdown
        """
        finger_positions = obs_dict['finger_positions']
        finger_bases = obs_dict.get('finger_bases', np.zeros((4, 3)))  # NEW: finger base positions
        palm_position = obs_dict['palm_position']
        object_pos = obs_dict['object_pos']
        binary_contact = obs_dict['binary_contact']

        # Generate hulls - ENHANCED 9-point hand hull (Fix #3)
        # OLD: 5 points (4 fingertips + 1 palm) - tiny pyramid volume ~0.04 cm³
        # NEW: 9 points (4 fingertips + 4 finger bases + 1 palm) - realistic hand shape ~0.15 cm³
        hand_hull_points = np.vstack([
            finger_positions,                  # 4 fingertips
            finger_bases,                      # 4 finger bases (NEW)
            palm_position.reshape(1, 3)        # 1 palm center
        ])  # Total: 9 points for much better hand volume representation
        object_hull_points = self.generate_object_hull_points(object_pos)

        # Calculate overlap volume (with validation)
        overlap_volume, hull_info = self.calculate_overlap_volume(
            hand_hull_points, object_hull_points
        )

        # Get phase-specific weights
        weights = self.phase_configs[self.current_phase]

        # === COMPONENT 1: Overlap Volume Reward ===
        overlap_reward = overlap_volume * weights['overlap_scale']

        # === COMPONENT 2: Contact Penalty ===
        num_contacts = np.sum(binary_contact)
        contact_penalty = num_contacts * weights['contact_penalty']

        # === COMPONENT 3: Proximity Reward (Phase 1 only) ===
        if self.current_phase == 1 and weights['proximity_scale'] > 0:
            proximity_reward = self.calculate_proximity_reward(hand_hull_points, object_pos)
            proximity_reward *= weights['proximity_scale']
        else:
            proximity_reward = 0.0

        # === COMPONENT 4: Quality Reward (Phase 2+) ===
        if self.current_phase >= 2 and weights['quality_scale'] > 0:
            quality_reward = self.calculate_quality_reward(
                hull_info.get('hand_hull_volume', 0.0),
                hull_info.get('object_hull_volume', 0.0)
            )
            quality_reward *= weights['quality_scale']

            # Size penalty for overspreading (Phase 2+)
            if 'size_penalty_scale' in weights:
                obj_vol = hull_info.get('object_hull_volume', 0.0)
                hand_vol = hull_info.get('hand_hull_volume', 0.0)
                if hand_vol > 4.0 * obj_vol:  # More than 4x object size
                    size_penalty = (hand_vol - 4.0 * obj_vol) / obj_vol
                    quality_reward -= size_penalty * weights['size_penalty_scale']
        else:
            quality_reward = 0.0

        # === COMPONENT 5: Clearance Error (Phase 3 only) ===
        if self.current_phase == 3 and weights.get('clearance_scale', 0) > 0:
            clearance_error = self.calculate_clearance_error(hand_hull_points, object_hull_points)
            # Convert error to reward (lower error = higher reward)
            clearance_reward = -clearance_error * weights['clearance_scale']
        else:
            clearance_reward = 0.0

        # === TOTAL REWARD ===
        total_reward = (overlap_reward + contact_penalty + proximity_reward +
                       quality_reward + clearance_reward)

        # === CONSECUTIVE SUCCESS TRACKING ===
        # Success criteria depends on phase
        is_success = False
        if self.current_phase == 1:
            # Phase 1: Any overlap without contact
            is_success = (overlap_volume > 0) and (num_contacts == 0)
        elif self.current_phase == 2:
            # Phase 2: Good overlap ratio
            obj_vol = hull_info.get('object_hull_volume', 1.0)
            overlap_ratio = overlap_volume / obj_vol if obj_vol > 0 else 0
            is_success = (overlap_ratio > 0.6) and (num_contacts <= 1)
        else:  # Phase 3
            # Phase 3: Precision
            obj_vol = hull_info.get('object_hull_volume', 1.0)
            overlap_ratio = overlap_volume / obj_vol if obj_vol > 0 else 0
            clearance_error = self.calculate_clearance_error(hand_hull_points, object_hull_points)
            is_success = (overlap_ratio > 0.7) and (num_contacts == 0) and (clearance_error < 0.01)

        if is_success:
            self.consecutive_success_steps += 1
        else:
            self.consecutive_success_steps = 0

        # Bonus for sustained success
        if self.consecutive_success_steps >= 50:
            sustained_bonus = 50.0
        elif self.consecutive_success_steps >= 25:
            sustained_bonus = 20.0
        elif self.consecutive_success_steps >= 10:
            sustained_bonus = 10.0
        else:
            sustained_bonus = 0.0

        total_reward += sustained_bonus

        # Calculate distance for logging
        hand_center = np.mean(hand_hull_points, axis=0)
        distance_to_target = float(np.linalg.norm(hand_center - object_pos))

        # === DETAILED INFO FOR LOGGING ===
        reward_info = {
            # Reward components
            'overlap_reward': float(overlap_reward),
            'contact_penalty': float(contact_penalty),
            'proximity_reward': float(proximity_reward),
            'quality_reward': float(quality_reward),
            'clearance_reward': float(clearance_reward),
            'sustained_bonus': float(sustained_bonus),

            # Hull metrics
            'overlap_volume': float(overlap_volume),
            'hand_hull_volume': float(hull_info.get('hand_hull_volume', 0.0)),
            'object_hull_volume': float(hull_info.get('object_hull_volume', 0.0)),
            'hand_hull_valid': hull_info.get('hand_hull_valid', False),
            'object_hull_valid': hull_info.get('object_hull_valid', False),

            # Distance metrics
            'distance_to_target': distance_to_target,

            # Contact metrics
            'num_contacts': int(num_contacts),

            # Success tracking
            'is_success': bool(is_success),
            'consecutive_success_steps': int(self.consecutive_success_steps),

            # Phase info
            'current_phase': int(self.current_phase),

            # Errors (if any)
            'error': hull_info.get('error', ''),
        }

        # Add clearance error for Phase 3
        if self.current_phase == 3:
            clearance_error = self.calculate_clearance_error(hand_hull_points, object_hull_points)
            reward_info['clearance_error'] = float(clearance_error)

        return total_reward, reward_info

    def update_phase(self, new_phase: int):
        """Update curriculum phase"""
        old_phase = self.current_phase
        self.current_phase = new_phase
        self.consecutive_success_steps = 0  # Reset on phase change

        if old_phase != new_phase:
            print(f"🎓 Curriculum Phase: {old_phase} → {new_phase}")
            print(f"   Weights: {self.phase_configs[new_phase]}")

    def reset(self):
        """Reset episode-specific tracking"""
        self.consecutive_success_steps = 0

    def get_expected_reward_range(self) -> Tuple[float, float]:
        """Return expected reward range for current phase"""
        weights = self.phase_configs[self.current_phase]

        # Minimum: multiple contacts
        min_reward = 4 * weights['contact_penalty']

        # Maximum: perfect performance + sustained bonus
        max_reward = 50.0  # Base components
        if self.current_phase >= 3:
            max_reward += 100.0  # Clearance precision
        max_reward += 50.0  # Sustained bonus

        return min_reward, max_reward


def create_default_config() -> Dict:
    """Create default configuration"""
    return {
        'object_radius': 0.05,
        'safety_margin': 0.025,
        'object_hull_points': 12,
    }