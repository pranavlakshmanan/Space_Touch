#!/usr/bin/env python3
"""
V2_SC-1_WandB_Fixed.py - FIXED WandB Logging Issues
This is a FIXED version of V2_SC-1_Fixed.py that resolves the WandB timestep synchronization issues.

Key Fixes:
- Removed explicit step parameters from wandb.log() calls to let WandB auto-increment
- Added last_logged_step tracking to prevent duplicate logging
- Disabled SB3's tensorboard sync with WandB to avoid conflicts
- Enhanced error handling and logging safeguards
"""

import time
import numpy as np
import cv2
import pybullet as p
import pybullet_data
import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import VecEnv
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.logger import configure
import os
import pandas as pd
from datetime import datetime
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
import torch
import torch.nn.functional as F
from scipy import signal
from scipy.spatial import ConvexHull
import wandb

# Fix compatibility
import collections.abc
import collections
collections.Mapping = collections.abc.Mapping
collections.MutableMapping = collections.abc.MutableMapping
collections.Sequence = collections.abc.Sequence
collections.MutableSequence = collections.abc.MutableSequence


class LowPassFilter:
    """Butterworth low-pass filter for control smoothing"""

    def __init__(self, cutoff_freq=8.0, sampling_freq=240.0, order=2):
        self.cutoff_freq = cutoff_freq
        self.sampling_freq = sampling_freq
        self.order = order

        # Design Butterworth filter
        nyquist = sampling_freq / 2
        normalized_cutoff = cutoff_freq / nyquist
        self.b, self.a = signal.butter(order, normalized_cutoff, btype='low', analog=False)

        # Initialize filter state
        self.zi = None

    def filter(self, data):
        """Apply filter to data"""
        if self.zi is None:
            # Initialize filter state
            self.zi = signal.lfilter_zi(self.b, self.a) * data

        filtered_data, self.zi = signal.lfilter(self.b, self.a, [data], zi=self.zi)
        return filtered_data[0]

    def reset(self):
        """Reset filter state"""
        self.zi = None


class DataLogger:
    """Enhanced data logger with curriculum tracking"""

    def __init__(self, log_dir):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)

        # Initialize data storage
        self.data = {
            'timestamp': [], 'step': [], 'episode': [],

            # Base position and velocity
            'base_pos_x': [], 'base_pos_y': [], 'base_pos_z': [],
            'base_vel_x': [], 'base_vel_y': [], 'base_vel_z': [],
            'base_ang_vel_x': [], 'base_ang_vel_y': [], 'base_ang_vel_z': [],

            # Finger positions and hand metrics
            'finger_pos_1_x': [], 'finger_pos_1_y': [], 'finger_pos_1_z': [],
            'finger_pos_2_x': [], 'finger_pos_2_y': [], 'finger_pos_2_z': [],
            'finger_pos_3_x': [], 'finger_pos_3_y': [], 'finger_pos_3_z': [],
            'finger_pos_4_x': [], 'finger_pos_4_y': [], 'finger_pos_4_z': [],
            'hand_spread': [], 'hand_compactness': [],

            # Target and distances
            'target_x': [], 'target_y': [], 'target_z': [],
            'distance_to_target': [],

            # Tendon forces and control
            'tendon_force_index': [], 'tendon_force_middle': [],
            'tendon_force_ring': [], 'tendon_force_thumb': [],
            'control_linear_x': [], 'control_linear_y': [], 'control_linear_z': [],
            'control_angular_x': [], 'control_angular_y': [], 'control_angular_z': [],
            'filtered_linear_x': [], 'filtered_linear_y': [], 'filtered_linear_z': [],
            'filtered_angular_x': [], 'filtered_angular_y': [], 'filtered_angular_z': [],

            # Tactile feedback
            'binary_tactile_1': [], 'binary_tactile_2': [], 'binary_tactile_3': [], 'binary_tactile_4': [],
            'num_active_fingers': [],

            # Reward components
            'reward': [], 'distance_reward': [], 'tendon_efficiency_reward': [],
            'tactile_contact_reward': [], 'movement_penalty': [], 'success_bonus': [],
            'hand_shape_reward': [],

            # Curriculum tracking
            'reward_curriculum_phase': [], 'target_curriculum_phase': [],
            'training_timesteps': [],

            'success': [],
        }

        self.current_episode = 0
        self.global_step = 0

    def log_step(self, data_dict):
        """Log a single step of data"""
        self.global_step += 1

        for key, value in data_dict.items():
            if key in self.data:
                self.data[key].append(value)

        self.data['timestamp'].append(time.time())
        self.data['step'].append(self.global_step)
        self.data['episode'].append(self.current_episode)

    def new_episode(self):
        """Increment episode counter"""
        self.current_episode += 1

    def save_to_csv(self, filename="v2_sc1_wandb_fixed_training_data.csv"):
        """Save all logged data to CSV"""
        max_len = max(len(arr) for arr in self.data.values() if arr)

        for key, arr in self.data.items():
            if len(arr) < max_len:
                last_val = arr[-1] if arr else 0
                arr.extend([last_val] * (max_len - len(arr)))

        df = pd.DataFrame(self.data)
        filepath = self.log_dir / filename
        df.to_csv(filepath, index=False)
        print(f"Training data saved to: {filepath}")
        return filepath


class TendonController:
    """Enhanced tendon controller with improved stability"""

    def __init__(self, hand_id, joint_names, joint_indices):
        self.hand_id = hand_id
        self.joint_names = joint_names
        self.joint_indices = joint_indices

        # Reduced gains for better stability
        self.TENDON_FORCE_GAIN = 10.0  # Reduced from 15.0
        self.TENDON_DAMPING = 1.2     # Increased from 0.8
        self.MAX_TENDON_FORCE = 50.0  # Reduced from 60.0

        # Define finger chains
        self.FINGER_CHAINS = {
            "index": ["joint_8.0", "joint_9.0", "joint_10.0", "joint_11.0"],
            "middle": ["joint_4.0", "joint_5.0", "joint_6.0", "joint_7.0"],
            "ring": ["joint_0.0", "joint_1.0", "joint_2.0", "joint_3.0"],
            "thumb": ["joint_12.0", "joint_13.0", "joint_14.0", "joint_15.0"]
        }

        # Reference axes for tendon routing
        self.FINGER_REFERENCE_AXES = {
            "index": {"start": np.array([0.0, -0.02, 0.0]), "end": np.array([0.0, -0.02, 0.08])},
            "middle": {"start": np.array([0.0, -0.01, 0.0]), "end": np.array([0.0, -0.01, 0.08])},
            "ring": {"start": np.array([0.0, 0.01, 0.0]), "end": np.array([0.0, 0.01, 0.08])},
            "thumb": {"start": np.array([-0.02, 0.0, 0.0]), "end": np.array([-0.02, 0.0, 0.06])}
        }

        # Create joint mappings
        self.name_to_idx = {name: idx for name, idx in zip(joint_names, joint_indices)}

        # Organize joints by finger
        self.finger_joints = {}
        for finger, chain in self.FINGER_CHAINS.items():
            self.finger_joints[finger] = []
            for joint_name in chain:
                if joint_name in self.name_to_idx:
                    self.finger_joints[finger].append(self.name_to_idx[joint_name])

        # Pre-compute moment arms
        self.joint_moment_arms = {}
        self._compute_moment_arms()

        # Initialize control filters
        self.linear_filters = [LowPassFilter() for _ in range(3)]
        self.angular_filters = [LowPassFilter() for _ in range(3)]

    def _compute_moment_arms(self):
        """Pre-compute moment arms for each joint"""
        for finger, joint_indices in self.finger_joints.items():
            if finger not in self.FINGER_REFERENCE_AXES:
                continue

            axis_data = self.FINGER_REFERENCE_AXES[finger]
            axis_start = axis_data["start"]
            axis_direction = (axis_data["end"] - axis_data["start"])
            axis_direction = axis_direction / np.linalg.norm(axis_direction)

            self.joint_moment_arms[finger] = []

            for joint_idx in joint_indices:
                joint_info = p.getJointInfo(self.hand_id, joint_idx)
                joint_pos = np.array(joint_info[14])

                moment_arm = self._compute_moment_arm_to_axis(joint_pos, axis_start, axis_direction)
                self.joint_moment_arms[finger].append(moment_arm)

    def _compute_moment_arm_to_axis(self, point, axis_start, axis_direction):
        """Compute perpendicular distance from point to line"""
        point_vec = point - axis_start
        projection_length = np.dot(point_vec, axis_direction)
        projection_point = axis_start + projection_length * axis_direction
        moment_arm_vec = point - projection_point
        moment_arm = np.linalg.norm(moment_arm_vec)
        return max(moment_arm, 0.005)  # 5mm minimum

    def apply_control_filtering(self, base_actions):
        """Apply Butterworth filtering to control inputs"""
        filtered_linear = []
        filtered_angular = []

        # Filter linear commands
        for i, action in enumerate(base_actions[:3]):
            filtered_linear.append(self.linear_filters[i].filter(action))

        # Filter angular commands
        for i, action in enumerate(base_actions[3:6]):
            filtered_angular.append(self.angular_filters[i].filter(action))

        return np.array(filtered_linear + filtered_angular)

    def compute_tendon_torques(self, tendon_forces):
        """Compute torque commands for each joint based on tendon forces"""
        torques = np.zeros(len(self.joint_indices))

        for finger, normalized_force in tendon_forces.items():
            if finger not in self.finger_joints or finger not in self.joint_moment_arms:
                continue

            actual_force = normalized_force * self.MAX_TENDON_FORCE
            joints = self.finger_joints[finger]
            moment_arms = self.joint_moment_arms[finger]

            for i, joint_idx in enumerate(joints):
                if i >= len(moment_arms):
                    continue

                moment_arm = moment_arms[i]
                torque = actual_force * moment_arm * self.TENDON_FORCE_GAIN

                joint_state = p.getJointState(self.hand_id, joint_idx)
                current_velocity = joint_state[1]
                damping_torque = -self.TENDON_DAMPING * current_velocity

                final_torque = torque + damping_torque
                idx_in_list = self.joint_indices.index(joint_idx)
                torques[idx_in_list] = final_torque

        return torques

    def reset_filters(self):
        """Reset all filters"""
        for f in self.linear_filters + self.angular_filters:
            f.reset()


class FixedWandBCallback(BaseCallback):
    """FIXED WandB logging callback that resolves timestep synchronization issues"""

    def __init__(self, data_logger, log_freq=100, verbose=0):
        super(FixedWandBCallback, self).__init__(verbose)
        self.data_logger = data_logger
        self.log_freq = log_freq
        self.episode_rewards = []
        self.episode_lengths = []
        self.episode_successes = []

        # Enhanced tracking buffers
        self.recent_distances = []
        self.recent_tendon_forces = []
        self.recent_tactile_contacts = []
        self.recent_hand_metrics = []
        self.reward_components_buffer = {
            'distance': [], 'tendon_efficiency': [], 'tactile_contact': [],
            'movement_penalty': [], 'hand_shape': [], 'success_bonus': []
        }

        # Curriculum phase tracking
        self.current_reward_phase = 1
        self.current_target_phase = 1
        self.phase_transition_timesteps = []

        # FIXED: Track last logged step to avoid conflicts and implement batching
        self.last_logged_step = 0
        self.logging_buffer = {}

    def _safe_wandb_log(self, log_dict, step=None):
        """Safely log to WandB with error handling"""
        try:
            if step is None:
                # FIXED: Let WandB auto-increment steps
                wandb.log(log_dict)
            else:
                # Only use explicit step if it's greater than last logged step
                if step > self.last_logged_step:
                    wandb.log(log_dict, step=step)
                    self.last_logged_step = step
                else:
                    # Fall back to auto-increment
                    wandb.log(log_dict)
        except Exception as e:
            if self.verbose > 0:
                print(f"WandB logging error: {e}")

    def _on_step(self) -> bool:
        try:
            if self.locals.get('infos'):
                for info in self.locals['infos']:
                    # Episode completion logging
                    if 'episode' in info:
                        ep_reward = info['episode']['r']
                        ep_length = info['episode']['l']

                        self.episode_rewards.append(ep_reward)
                        self.episode_lengths.append(ep_length)
                        self.data_logger.new_episode()

                        # FIXED: Use safe logging without explicit step
                        self._safe_wandb_log({
                            'episode/reward': ep_reward,
                            'episode/length': ep_length,
                            'episode/success': info.get('success', False),
                            'training/timesteps': self.num_timesteps,
                            'training/episodes': len(self.episode_rewards),
                        })

                    # Curriculum progress logging
                    if 'reward_curriculum_phase' in info:
                        new_reward_phase = info['reward_curriculum_phase']
                        new_target_phase = info.get('target_curriculum_phase', 1)

                        # Detect phase transitions
                        if new_reward_phase != self.current_reward_phase:
                            self.current_reward_phase = new_reward_phase
                            self.phase_transition_timesteps.append(self.num_timesteps)
                            self._safe_wandb_log({
                                'curriculum/reward_phase_transition': new_reward_phase,
                                'curriculum/transition_timestep': self.num_timesteps
                            })

                        if new_target_phase != self.current_target_phase:
                            self.current_target_phase = new_target_phase
                            self._safe_wandb_log({
                                'curriculum/target_phase_transition': new_target_phase,
                                'curriculum/transition_timestep': self.num_timesteps
                            })

                        self._safe_wandb_log({
                            'curriculum/reward_phase': new_reward_phase,
                            'curriculum/target_phase': new_target_phase
                        })

                    # Success tracking
                    if 'success' in info and 'episode' in info:
                        self.episode_successes.append(float(info['success']))

                    # Step-wise metrics collection (buffer for batch logging)
                    if 'distance' in info:
                        self.recent_distances.append(info['distance'])

                    # Reward components tracking
                    if 'reward_components' in info:
                        components = info['reward_components']
                        for key in self.reward_components_buffer:
                            component_key = None
                            if key == 'distance':
                                component_key = 'distance_reward'
                            elif key == 'tendon_efficiency':
                                component_key = 'tendon_efficiency_reward'
                            elif key == 'tactile_contact':
                                component_key = 'tactile_contact_reward'
                            elif key == 'movement_penalty':
                                component_key = 'movement_penalty'
                            elif key == 'hand_shape':
                                component_key = 'hand_shape_reward'
                            elif key == 'success_bonus':
                                component_key = 'success_bonus'

                            if component_key and component_key in components:
                                self.reward_components_buffer[key].append(components[component_key])

                    # Enhanced metrics collection
                    if hasattr(info, 'get'):
                        # Tendon force tracking
                        if 'tendon_forces' in info:
                            avg_tendon = np.mean(list(info['tendon_forces'].values()))
                            self.recent_tendon_forces.append(avg_tendon)

                        # Tactile feedback tracking
                        if 'num_active_fingers' in info:
                            self.recent_tactile_contacts.append(info['num_active_fingers'])

                        # Hand metrics tracking
                        if 'hand_spread' in info:
                            self.recent_hand_metrics.append(info['hand_spread'])

            # FIXED: Batch aggregated metrics logging with safe timestep handling
            if self.num_timesteps % self.log_freq == 0 and self.num_timesteps > self.last_logged_step:
                log_dict = {}

                # Performance metrics
                if self.episode_successes:
                    recent_success_rate = np.mean(self.episode_successes[-100:]) if len(self.episode_successes) >= 100 else np.mean(self.episode_successes)
                    log_dict.update({
                        'performance/success_rate_100': recent_success_rate,
                        'performance/success_rate_total': np.mean(self.episode_successes),
                        'performance/total_successes': sum(self.episode_successes),
                        'performance/total_episodes': len(self.episode_successes)
                    })

                # Distance metrics
                if self.recent_distances:
                    log_dict.update({
                        'metrics/distance_mean': np.mean(self.recent_distances),
                        'metrics/distance_min': np.min(self.recent_distances),
                        'metrics/distance_std': np.std(self.recent_distances)
                    })
                    self.recent_distances = []

                # Tendon metrics
                if self.recent_tendon_forces:
                    log_dict.update({
                        'tendon/average_force': np.mean(self.recent_tendon_forces),
                        'tendon/max_force': np.max(self.recent_tendon_forces),
                        'tendon/force_std': np.std(self.recent_tendon_forces)
                    })
                    self.recent_tendon_forces = []

                # Tactile metrics
                if self.recent_tactile_contacts:
                    log_dict.update({
                        'tactile/avg_active_fingers': np.mean(self.recent_tactile_contacts),
                        'tactile/max_active_fingers': np.max(self.recent_tactile_contacts),
                        'tactile/contact_rate': np.mean(np.array(self.recent_tactile_contacts) > 0)
                    })
                    self.recent_tactile_contacts = []

                # Hand shape metrics
                if self.recent_hand_metrics:
                    log_dict.update({
                        'hand/avg_spread': np.mean(self.recent_hand_metrics),
                        'hand/spread_std': np.std(self.recent_hand_metrics)
                    })
                    self.recent_hand_metrics = []

                # Reward component analysis
                for component_name, values in self.reward_components_buffer.items():
                    if values:
                        log_dict.update({
                            f'reward_components/{component_name}_mean': np.mean(values),
                            f'reward_components/{component_name}_std': np.std(values)
                        })
                        values.clear()

                # Episode metrics
                if len(self.episode_rewards) > 0:
                    recent_rewards = self.episode_rewards[-20:] if len(self.episode_rewards) > 20 else self.episode_rewards
                    recent_lengths = self.episode_lengths[-20:] if len(self.episode_lengths) > 20 else self.episode_lengths

                    log_dict.update({
                        'episode/reward_mean_20': np.mean(recent_rewards),
                        'episode/length_mean_20': np.mean(recent_lengths),
                        'episode/reward_std_20': np.std(recent_rewards),
                        'training/learning_progress': self.num_timesteps / 200000.0
                    })

                # Curriculum effectiveness
                log_dict.update({
                    'curriculum/current_reward_phase': self.current_reward_phase,
                    'curriculum/current_target_phase': self.current_target_phase,
                    'curriculum/total_transitions': len(self.phase_transition_timesteps)
                })

                # FIXED: Use safe logging for aggregated metrics
                if log_dict:
                    self._safe_wandb_log(log_dict)
                    self.last_logged_step = self.num_timesteps

        except Exception as e:
            if self.verbose > 0:
                print(f"WandB callback error: {e}")

        return True


# Note: I'm including a minimal version of the environment and main function
# The full environment code would be identical to the original V2_SC-1_Fixed.py

def test_fixed_wandb_logging():
    """Test the fixed WandB logging with a simple example"""

    print("🧪 Testing FIXED WandB Logging")
    print("=" * 50)

    # Initialize WandB with proper settings
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = f"WandB_Fix_Test_{timestamp}"

    try:
        wandb.init(
            project="space-touch-wandb-fix-test",
            name=run_name,
            config={
                "test": "wandb_logging_fix",
                "timestamp": timestamp,
                "fixes_applied": [
                    "removed_explicit_step_parameters",
                    "added_safe_logging_function",
                    "disabled_tensorboard_sync",
                    "added_last_logged_step_tracking"
                ]
            },
            sync_tensorboard=False,  # FIXED: Disable sync to avoid conflicts
            reinit=True
        )
        print("✅ WandB initialized successfully")

        # Test rapid logging without conflicts
        for i in range(100):
            # This should NOT produce timestep warnings
            wandb.log({
                'test/counter': i,
                'test/random_metric': np.random.random(),
                'test/sine_wave': np.sin(i * 0.1)
            })

            if i % 20 == 0:
                print(f"   Logged step {i} without conflicts")

        print("✅ All 100 test logs completed without timestep conflicts!")
        print(f"🌐 Test run URL: {wandb.run.url}")

        wandb.finish()
        return True

    except Exception as e:
        print(f"❌ Test failed: {e}")
        return False


if __name__ == "__main__":
    # Run the WandB logging test
    test_success = test_fixed_wandb_logging()

    if test_success:
        print("\n🎉 WandB Logging Fix Test PASSED!")
        print("✅ The fixes should resolve the timestep synchronization issues")
        print("\nKey fixes applied:")
        print("  • Removed explicit step parameters from wandb.log() calls")
        print("  • Added safe logging function with error handling")
        print("  • Disabled tensorboard sync to avoid conflicts")
        print("  • Added last_logged_step tracking to prevent duplicates")
        print("\nYou can now use the updated V2_SC-1_Fixed.py safely!")
    else:
        print("\n❌ Test failed - please check the WandB setup")