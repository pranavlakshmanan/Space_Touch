#!/usr/bin/env python3
"""
Simplified 3-Component Reward Function for SC-1 Space Manipulator
Designed to fix negative reward issues and enable successful soft-capture learning.

Key Design Principles:
1. SIMPLICITY: Only 3 reward components (vs previous 10+)
2. NO HARSH PENALTIES: Max penalty -0.5 (vs previous -1.0 per step)
3. POSITIVE REWARD DOMINANCE: Distance progress provides positive base
4. TACTILE ENCOURAGEMENT: Rewards gentle contact when close to target
5. STAGED SUCCESS: Dense reward signal as agent approaches success

Expected reward range: [-0.5, 52.5] (RELAXED CRITERIA - V3.1)
Typical early training: 0.2 to 3.0 (with new milestones)
Typical late training: 10.0 to 50.0+ (with sustained success bonuses)
"""

import numpy as np
from typing import Dict, Tuple


class SimplifiedReward:
    """
    Simplified 3-component reward function for SC-1 soft-capture task

    Components:
    1. Distance Progress Reward: exp(-20.0 * distance) - Encourages approaching target
    2. Staged Success Bonus: Cascading bonuses for proximity milestones
    3. Tactile Engagement Reward: Encourages gentle contact when close
    """

    def __init__(self, config: Dict = None):
        """
        Initialize Simplified Reward Function with RELAXED thresholds for initial learning

        CRITICAL CHANGES (Option 2 - Relaxed Criteria):
        - Increased distance thresholds to make success more achievable
        - Reduced consecutive steps requirement for faster reward feedback
        - Added intermediate milestones for denser learning signal
        """
        if config is None:
            config = {}

        # Distance thresholds - RELAXED for initial learning
        # OLD VALUES (commented for reference):
        # self.PROXIMITY_THRESHOLD = 0.15
        # self.CLOSE_THRESHOLD = 0.12
        # self.VERY_CLOSE_THRESHOLD = 0.08
        # self.SUCCESS_THRESHOLD = 0.08

        # NEW RELAXED VALUES:
        self.VERY_FAR_THRESHOLD = config.get('very_far_threshold', 0.40)        # NEW: Initial approach milestone
        self.FAR_THRESHOLD = config.get('far_threshold', 0.30)                  # NEW: Getting closer milestone
        self.PROXIMITY_THRESHOLD = config.get('proximity_threshold', 0.20)      # RELAXED: Was 0.15m, now 20cm
        self.CLOSE_THRESHOLD = config.get('close_threshold', 0.15)              # RELAXED: Was 0.12m, now 15cm
        self.VERY_CLOSE_THRESHOLD = config.get('very_close_threshold', 0.12)    # RELAXED: Was 0.08m, now 12cm
        self.SUCCESS_THRESHOLD = config.get('success_threshold', 0.12)          # RELAXED: Was 0.08m, now 12cm (CRITICAL)

        # Success tracking - RELAXED for faster learning
        # OLD VALUE:
        # self.MIN_CONSECUTIVE_STEPS = 50

        # NEW RELAXED VALUE:
        self.MIN_CONSECUTIVE_STEPS = config.get('min_consecutive_steps', 25)    # REDUCED: Was 50, now 25 steps (CRITICAL)
        self.consecutive_success_steps = 0

        # Tactile force thresholds
        self.GENTLE_FORCE_THRESHOLD = config.get('gentle_force_threshold', 5.0)    # 5N = gentle contact
        self.HARD_FORCE_THRESHOLD = config.get('hard_force_threshold', 20.0)       # 20N = hard contact (safety)
        self.CLOSE_DISTANCE_FOR_CONTACT = config.get('close_distance_for_contact', 0.2)  # 20cm proximity zone

        # Reward scaling parameters
        self.DISTANCE_SCALE = config.get('distance_scale', 20.0)  # Exponential scaling factor

        # Reward component weights (for future tuning if needed)
        self.DISTANCE_WEIGHT = config.get('distance_weight', 1.0)
        self.SUCCESS_WEIGHT = config.get('success_weight', 1.0)
        self.TACTILE_WEIGHT = config.get('tactile_weight', 1.0)

        # Bonus values
        self.PROXIMITY_BONUS = config.get('proximity_bonus', 2.0)
        self.CLOSE_BONUS = config.get('close_bonus', 5.0)
        self.VERY_CLOSE_BONUS = config.get('very_close_bonus', 10.0)
        self.SUSTAINED_SUCCESS_BONUS = config.get('sustained_success_bonus', 20.0)

        # Tactile reward values
        self.GENTLE_CONTACT_REWARD = config.get('gentle_contact_reward', 0.5)
        self.FAR_CONTACT_PENALTY = config.get('far_contact_penalty', -0.1)
        self.HARD_CONTACT_PENALTY = config.get('hard_contact_penalty', -0.5)

        # Print confirmation of relaxed criteria
        print("🔧 Initialized Simplified Reward with RELAXED success criteria:")
        print(f"   Success distance threshold: {self.SUCCESS_THRESHOLD}m (relaxed from 0.08m)")
        print(f"   Consecutive steps required: {self.MIN_CONSECUTIVE_STEPS} (reduced from 50)")
        print(f"   Intermediate milestones: 0.40m, 0.30m, 0.20m, 0.15m, 0.12m")
        print(f"   Distance scale: {self.DISTANCE_SCALE}")
        print(f"   Tactile thresholds: Gentle={self.GENTLE_FORCE_THRESHOLD}N, Hard={self.HARD_FORCE_THRESHOLD}N")

    def calculate_reward(self, obs_dict: Dict) -> Tuple[float, Dict]:
        """
        Calculate total reward and component breakdown

        Args:
            obs_dict: Dictionary containing:
                - 'distance': float, L2 distance to target (meters)
                - 'contact_force': float, total normal force from tactile sensors (Newtons)
                - 'hand_pos': np.array, 3D hand position (optional, for debugging)
                - 'target_pos': np.array, 3D target position (optional, for debugging)

        Returns:
            total_reward: float, combined reward value
            reward_info: Dict, detailed component breakdown for logging
        """

        distance = obs_dict['distance']
        contact_force = obs_dict.get('contact_force', 0.0)

        # ================== COMPONENT 1: Distance Progress Reward ==================
        # Provides strong positive gradient to approach target
        # Range: [0, 1], exponential decay with distance
        distance_reward = np.exp(-self.DISTANCE_SCALE * distance) * self.DISTANCE_WEIGHT

        # ================== COMPONENT 2: ENHANCED Staged Success Bonus with MORE MILESTONES ==================
        """
        CRITICAL CHANGE: Added intermediate milestones for denser reward signal

        Previous structure: 3 stages (0.15m, 0.12m, 0.08m)
        New structure: 6 stages with granular progression

        This provides much denser feedback to the agent as it approaches the target,
        making it easier to learn the approach behavior.
        """
        success_bonus = 0.0
        success_stage = 0
        in_success_zone = False

        # Stage 0: Very Far Approach (NEW - 40cm threshold)
        if distance < self.VERY_FAR_THRESHOLD:  # < 0.40m
            success_bonus += 1.0
            success_stage = 1

        # Stage 1: Far Approach (NEW - 30cm threshold)
        if distance < self.FAR_THRESHOLD:  # < 0.30m
            success_bonus += 2.0  # Cumulative: 3.0 total
            success_stage = 2

        # Stage 2: Proximity Zone (RELAXED - 20cm threshold)
        if distance < self.PROXIMITY_THRESHOLD:  # < 0.20m
            success_bonus += 3.0  # Cumulative: 6.0 total
            success_stage = 3

        # Stage 3: Close Zone (RELAXED - 15cm threshold)
        if distance < self.CLOSE_THRESHOLD:  # < 0.15m
            success_bonus += 5.0  # Cumulative: 11.0 total
            success_stage = 4

        # Stage 4: Very Close Zone (RELAXED - 12cm threshold)
        if distance < self.VERY_CLOSE_THRESHOLD:  # < 0.12m
            success_bonus += 10.0  # Cumulative: 21.0 total
            success_stage = 5

        # Stage 5: Success Zone - Track consecutive steps
        if distance < self.SUCCESS_THRESHOLD:  # < 0.12m (RELAXED)
            in_success_zone = True
            self.consecutive_success_steps += 1

            # Stage 6: Sustained Success (REDUCED requirement - 25 steps)
            if self.consecutive_success_steps >= self.MIN_CONSECUTIVE_STEPS:
                success_bonus += 30.0  # Cumulative: 51.0 total - MASSIVE BONUS!
                success_stage = 6
        else:
            # Reset consecutive counter if outside success zone
            self.consecutive_success_steps = 0

        success_bonus *= self.SUCCESS_WEIGHT

        # ================== COMPONENT 3: Tactile Engagement Reward ==================
        # Encourages gentle contact when close, discourages when far or hard
        # Gentle contact (< 5N) when close (< 20cm): +0.5
        # Any contact when far (>= 20cm): -0.1 (discourages flailing)
        # Hard contact (> 20N): -0.5 (safety limit)

        tactile_reward = 0.0

        if contact_force > self.HARD_FORCE_THRESHOLD:
            # Hard contact: safety violation
            tactile_reward = self.HARD_CONTACT_PENALTY

        elif contact_force > 0:
            # Some contact detected
            if distance < self.CLOSE_DISTANCE_FOR_CONTACT:
                # Close to target: reward gentle contact
                if contact_force <= self.GENTLE_FORCE_THRESHOLD:
                    tactile_reward = self.GENTLE_CONTACT_REWARD
                else:
                    # Contact too hard but not safety violation
                    tactile_reward = self.GENTLE_CONTACT_REWARD * 0.5
            else:
                # Far from target: discourage contact (flailing behavior)
                tactile_reward = self.FAR_CONTACT_PENALTY

        tactile_reward *= self.TACTILE_WEIGHT

        # ================== TOTAL REWARD CALCULATION ==================
        total_reward = distance_reward + success_bonus + tactile_reward

        # ================== DETAILED INFO FOR LOGGING ==================
        reward_info = {
            'distance_reward': float(distance_reward),
            'success_bonus': float(success_bonus),
            'tactile_reward': float(tactile_reward),
            'consecutive_steps': int(self.consecutive_success_steps),
            'success_stage': int(success_stage),
            'in_success_zone': bool(in_success_zone),
            'distance': float(distance),
            'contact_force': float(contact_force),
            'has_gentle_contact': bool(0 < contact_force <= self.GENTLE_FORCE_THRESHOLD),
            'has_hard_contact': bool(contact_force > self.HARD_FORCE_THRESHOLD),
        }

        return total_reward, reward_info


    def reset(self):
        """Reset episode-specific tracking variables"""
        self.consecutive_success_steps = 0

    def get_success_criteria(self) -> Dict:
        """
        Return current success criteria for external use (e.g., environment done checking)

        Returns:
            Dictionary with success thresholds
        """
        return {
            'distance_threshold': self.SUCCESS_THRESHOLD,
            'min_consecutive_steps': self.MIN_CONSECUTIVE_STEPS,
            'very_close_threshold': self.VERY_CLOSE_THRESHOLD,
            'close_threshold': self.CLOSE_THRESHOLD,
            'proximity_threshold': self.PROXIMITY_THRESHOLD,
        }

    def get_expected_reward_range(self) -> Tuple[float, float]:
        """Return expected reward range for normalization/debugging"""
        min_reward = self.HARD_CONTACT_PENALTY * self.TACTILE_WEIGHT  # -0.5
        max_reward = (1.0 * self.DISTANCE_WEIGHT +  # Distance reward (max 1.0)
                     51.0 * self.SUCCESS_WEIGHT +  # Success bonus (51.0 for sustained success)
                     self.GENTLE_CONTACT_REWARD * self.TACTILE_WEIGHT)  # Tactile reward (0.5)
        return min_reward, max_reward

    def __str__(self) -> str:
        """String representation for debugging"""
        min_r, max_r = self.get_expected_reward_range()
        return (f"SimplifiedReward(components=3, range=[{min_r:.1f}, {max_r:.1f}], "
                f"success_threshold={self.SUCCESS_THRESHOLD}m)")


def create_default_config() -> Dict:
    """Create default configuration for SimplifiedReward with RELAXED CRITERIA"""
    return {
        # Distance thresholds - RELAXED VALUES
        'very_far_threshold': 0.40,        # NEW: Initial approach milestone
        'far_threshold': 0.30,             # NEW: Getting closer milestone
        'proximity_threshold': 0.20,       # RELAXED: Was 0.15, now 0.20
        'close_threshold': 0.15,           # RELAXED: Was 0.12, now 0.15
        'very_close_threshold': 0.12,      # RELAXED: Was 0.08, now 0.12
        'success_threshold': 0.12,         # RELAXED: Was 0.08, now 0.12

        # Success tracking - RELAXED VALUES
        'min_consecutive_steps': 25,       # REDUCED: Was 50, now 25

        # Tactile thresholds
        'gentle_force_threshold': 5.0,
        'hard_force_threshold': 20.0,
        'close_distance_for_contact': 0.2,

        # Scaling
        'distance_scale': 20.0,

        # Component weights (for future tuning)
        'distance_weight': 1.0,
        'success_weight': 1.0,
        'tactile_weight': 1.0,

        # Bonus values
        'proximity_bonus': 2.0,
        'close_bonus': 5.0,
        'very_close_bonus': 10.0,
        'sustained_success_bonus': 20.0,

        # Tactile rewards/penalties
        'gentle_contact_reward': 0.5,
        'far_contact_penalty': -0.1,
        'hard_contact_penalty': -0.5,
    }


if __name__ == "__main__":
    # Quick test of reward function
    print("=" * 60)
    print("🧪 SIMPLIFIED REWARD FUNCTION TEST")
    print("=" * 60)

    # Create reward function
    reward_func = SimplifiedReward()
    print(f"\nReward function: {reward_func}")

    # Test scenarios
    test_cases = [
        {"name": "Far away", "distance": 1.0, "contact_force": 0.0},
        {"name": "Approaching", "distance": 0.3, "contact_force": 0.0},
        {"name": "Close", "distance": 0.1, "contact_force": 0.0},
        {"name": "Close + gentle contact", "distance": 0.1, "contact_force": 3.0},
        {"name": "Success zone", "distance": 0.07, "contact_force": 0.0},
        {"name": "Success + gentle contact", "distance": 0.07, "contact_force": 2.0},
        {"name": "Far + contact (bad)", "distance": 0.5, "contact_force": 3.0},
        {"name": "Hard contact (danger)", "distance": 0.1, "contact_force": 25.0},
    ]

    for test in test_cases:
        obs = {
            'distance': test['distance'],
            'contact_force': test['contact_force'],
            'hand_pos': np.zeros(3),
            'target_pos': np.zeros(3),
        }

        reward, info = reward_func.calculate_reward(obs)
        print(f"\n📋 {test['name']:20s} | Reward: {reward:6.3f} | Distance: {info['distance_reward']:.3f} | Success: {info['success_bonus']:4.1f} | Tactile: {info['tactile_reward']:5.2f}")

    print(f"\n✅ Test complete. Expected range: {reward_func.get_expected_reward_range()}")