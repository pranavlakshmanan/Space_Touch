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

Expected reward range: [-0.5, 21.5]
Typical early training: 0.2 to 0.8
Typical late training: 5.0 to 20.0+
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
        Initialize reward function with thresholds and parameters

        Args:
            config: Optional configuration dictionary for customization
        """
        if config is None:
            config = {}

        # Distance thresholds for staged success bonus
        self.PROXIMITY_THRESHOLD = config.get('proximity_threshold', 0.15)      # Stage 1: +2.0
        self.CLOSE_THRESHOLD = config.get('close_threshold', 0.12)              # Stage 2: +5.0
        self.VERY_CLOSE_THRESHOLD = config.get('very_close_threshold', 0.08)    # Stage 3: +10.0
        self.SUCCESS_THRESHOLD = config.get('success_threshold', 0.08)          # Stage 4: +20.0

        # Success tracking for sustained success bonus
        self.MIN_CONSECUTIVE_STEPS = config.get('min_consecutive_steps', 50)
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

        print("✅ Simplified Reward Function Initialized")
        print(f"   Distance scale: {self.DISTANCE_SCALE}")
        print(f"   Success thresholds: {self.SUCCESS_THRESHOLD}m (sustained: {self.MIN_CONSECUTIVE_STEPS} steps)")
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

        # ================== COMPONENT 2: Staged Success Bonus ==================
        # Cascading bonuses for reaching proximity milestones
        # Stage 1: Proximity (15cm) → +2.0
        # Stage 2: Close (12cm) → +5.0 total
        # Stage 3: Very Close (8cm) → +10.0 total
        # Stage 4: Sustained Success (8cm for 50+ steps) → +20.0 total

        success_bonus = 0.0

        if distance < self.PROXIMITY_THRESHOLD:
            success_bonus = self.PROXIMITY_BONUS

        if distance < self.CLOSE_THRESHOLD:
            success_bonus = self.CLOSE_BONUS

        if distance < self.VERY_CLOSE_THRESHOLD:
            success_bonus = self.VERY_CLOSE_BONUS

        # Track consecutive steps in success zone for sustained success bonus
        if distance < self.SUCCESS_THRESHOLD:
            self.consecutive_success_steps += 1
            if self.consecutive_success_steps >= self.MIN_CONSECUTIVE_STEPS:
                success_bonus = self.SUSTAINED_SUCCESS_BONUS
        else:
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
            'total_reward': total_reward,
            'distance_reward': distance_reward,
            'success_bonus': success_bonus,
            'tactile_reward': tactile_reward,
            'distance': distance,
            'contact_force': contact_force,
            'consecutive_steps': self.consecutive_success_steps,
            'success_stage': self._get_success_stage(distance, success_bonus),
            'in_success_zone': distance < self.SUCCESS_THRESHOLD,
            'has_gentle_contact': 0 < contact_force <= self.GENTLE_FORCE_THRESHOLD,
            'has_hard_contact': contact_force > self.HARD_FORCE_THRESHOLD,
        }

        return total_reward, reward_info

    def _get_success_stage(self, distance: float, success_bonus: float) -> int:
        """Helper to determine current success stage for logging"""
        if success_bonus >= self.SUSTAINED_SUCCESS_BONUS:
            return 4  # Sustained success
        elif distance < self.VERY_CLOSE_THRESHOLD:
            return 3  # Very close
        elif distance < self.CLOSE_THRESHOLD:
            return 2  # Close
        elif distance < self.PROXIMITY_THRESHOLD:
            return 1  # Proximity
        else:
            return 0  # Far

    def reset(self):
        """Reset episode-specific tracking variables"""
        self.consecutive_success_steps = 0

    def get_success_criteria(self) -> Dict:
        """Return success criteria for external evaluation"""
        return {
            'distance_threshold': self.SUCCESS_THRESHOLD,
            'min_consecutive_steps': self.MIN_CONSECUTIVE_STEPS,
            'requires_gentle_contact': False,  # Not required for success, just rewarded
        }

    def get_expected_reward_range(self) -> Tuple[float, float]:
        """Return expected reward range for normalization/debugging"""
        min_reward = self.HARD_CONTACT_PENALTY * self.TACTILE_WEIGHT  # -0.5
        max_reward = (1.0 * self.DISTANCE_WEIGHT +  # Distance reward (max 1.0)
                     self.SUSTAINED_SUCCESS_BONUS * self.SUCCESS_WEIGHT +  # Success bonus (20.0)
                     self.GENTLE_CONTACT_REWARD * self.TACTILE_WEIGHT)  # Tactile reward (0.5)
        return min_reward, max_reward

    def __str__(self) -> str:
        """String representation for debugging"""
        min_r, max_r = self.get_expected_reward_range()
        return (f"SimplifiedReward(components=3, range=[{min_r:.1f}, {max_r:.1f}], "
                f"success_threshold={self.SUCCESS_THRESHOLD}m)")


def create_default_config() -> Dict:
    """Create default configuration for SimplifiedReward"""
    return {
        # Distance thresholds
        'proximity_threshold': 0.15,
        'close_threshold': 0.12,
        'very_close_threshold': 0.08,
        'success_threshold': 0.08,

        # Success tracking
        'min_consecutive_steps': 50,

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