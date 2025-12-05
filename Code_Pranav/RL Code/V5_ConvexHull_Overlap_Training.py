#!/usr/bin/env python3
"""
V5_ConvexHull_Overlap_Training.py - REVOLUTIONARY CONVEX HULL OVERLAP APPROACH
Complete paradigm shift from distance-based to spatial containment-based grasping.

V5 BREAKTHROUGH INNOVATION:
- CONVEX HULL OVERLAP: Maximize intersection volume between hand and object hulls
- NO CONTACT REQUIREMENT: Reward spatial envelopment without touching
- UNIFIED OBJECTIVE: Single goal encompasses distance closing + envelopment + safety
- REAL-TIME VISUALIZATION: PNG generation of dual hull system

KEY ADVANTAGES:
1. Unified Task: No competing sub-objectives (distance vs contact vs shape)
2. Safety-First: Strong penalties for contact ensure safe manipulation
3. Spatial Intelligence: Learns 3D containment rather than simple reaching
4. Intuitive Reward: More overlap = better grasp preparation
5. Transferable: Real-world applicability without contact sensors

TECHNICAL IMPLEMENTATION:
- Hand Hull: 4 fingertips + palm center (5 points)
- Object Hull: Sphere with 2.5cm safety margin (12 points)
- Reward: Overlap volume × 10,000 - 5.0 per contact + proximity + quality
- Action/Observation: Same as V3 (unchanged per request)
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
import random
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
import torch
import torch.nn.functional as F
from scipy import signal
from scipy.spatial import ConvexHull
import wandb

# Import the new convex hull overlap reward function
import sys
sys.path.append('/home/pralak/Space_Touch')
from reward_functions.convex_hull_envelopment_reward import ConvexHullEnvelopmentReward

# Fix compatibility
import collections.abc
import collections
collections.Mapping = collections.abc.Mapping
collections.MutableMapping = collections.abc.MutableMapping
collections.Sequence = collections.abc.Sequence
collections.MutableSequence = collections.abc.MutableSequence

# Initialize TACTO for tactile sensing (with error handling)
try:
    import tacto
    TACTO_AVAILABLE = True
    print("✅ TACTO tactile sensing library loaded successfully")
except ImportError as e:
    TACTO_AVAILABLE = False
    print(f"⚠️  TACTO not available: {e}")


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
    """Enhanced data logger with convex hull overlap metrics"""

    def __init__(self, log_dir):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)

        # Initialize data storage with convex hull metrics
        self.data = {
            'timestamp': [], 'step': [], 'episode': [],

            # Base position and velocity
            'base_pos_x': [], 'base_pos_y': [], 'base_pos_z': [],
            'base_vel_x': [], 'base_vel_y': [], 'base_vel_z': [],
            'base_ang_vel_x': [], 'base_ang_vel_y': [], 'base_ang_vel_z': [],

            # Target and distance
            'target_pos_x': [], 'target_pos_y': [], 'target_pos_z': [],
            'distance_to_target': [],

            # Finger positions (4 fingers × 3D)
            'finger_pos_0_x': [], 'finger_pos_0_y': [], 'finger_pos_0_z': [],  # Index
            'finger_pos_1_x': [], 'finger_pos_1_y': [], 'finger_pos_1_z': [],  # Middle
            'finger_pos_2_x': [], 'finger_pos_2_y': [], 'finger_pos_2_z': [],  # Ring
            'finger_pos_3_x': [], 'finger_pos_3_y': [], 'finger_pos_3_z': [],  # Thumb

            # Palm position
            'palm_pos_x': [], 'palm_pos_y': [], 'palm_pos_z': [],

            # Tactile feedback
            'tactile_finger_0': [], 'tactile_finger_1': [], 'tactile_finger_2': [], 'tactile_finger_3': [],
            'num_active_fingers': [], 'total_contact_force': [],

            # Control commands
            'cmd_base_linear_x': [], 'cmd_base_linear_y': [], 'cmd_base_linear_z': [],
            'cmd_base_angular_x': [], 'cmd_base_angular_y': [], 'cmd_base_angular_z': [],
            'cmd_tendon_0': [], 'cmd_tendon_1': [], 'cmd_tendon_2': [], 'cmd_tendon_3': [],

            # Filtered control commands
            'filtered_base_linear_x': [], 'filtered_base_linear_y': [], 'filtered_base_linear_z': [],
            'filtered_base_angular_x': [], 'filtered_base_angular_y': [], 'filtered_base_angular_z': [],

            # CONVEX HULL OVERLAP METRICS
            'overlap_reward': [], 'contact_penalty': [], 'proximity_reward': [], 'quality_reward': [],
            'overlap_volume': [], 'num_contacts': [], 'hand_hull_volume': [], 'object_hull_volume': [],

            # Success metrics
            'reward': [], 'episode_reward': [], 'success': [],
        }

    def log_step(self, obs, action, filtered_action, reward_info, episode_reward, success):
        """Log a single timestep with convex hull overlap data"""
        timestamp = time.time()
        self.data['timestamp'].append(timestamp)
        self.data['step'].append(len(self.data['step']))

        # Episode info
        episode = len([r for r in self.data['episode_reward'] if r is not None and r > 0])
        self.data['episode'].append(episode)

        # Base state (obs[:9])
        base_pos = obs[:3]
        target_pos = obs[3:6]
        base_vel = obs[6:9]

        for i, coord in enumerate(['x', 'y', 'z']):
            self.data[f'base_pos_{coord}'].append(base_pos[i])
            self.data[f'target_pos_{coord}'].append(target_pos[i])
            self.data[f'base_vel_{coord}'].append(base_vel[i])

        # Angular velocity (obs[9:12])
        base_ang_vel = obs[9:12]
        for i, coord in enumerate(['x', 'y', 'z']):
            self.data[f'base_ang_vel_{coord}'].append(base_ang_vel[i])

        # Distance
        distance = np.linalg.norm(base_pos - target_pos)
        self.data['distance_to_target'].append(distance)

        # Finger positions (obs[12:24])
        finger_positions = obs[12:24].reshape(4, 3)
        for finger_idx in range(4):
            for coord_idx, coord in enumerate(['x', 'y', 'z']):
                self.data[f'finger_pos_{finger_idx}_{coord}'].append(finger_positions[finger_idx, coord_idx])

        # Tactile feedback (obs[24:28])
        tactile_data = obs[24:28]
        # Ensure we have exactly 4 tactile values
        if len(tactile_data) < 4:
            tactile_data = np.concatenate([tactile_data, np.zeros(4 - len(tactile_data))])
        for finger_idx in range(4):
            self.data[f'tactile_finger_{finger_idx}'].append(tactile_data[finger_idx])

        self.data['num_active_fingers'].append(np.sum(tactile_data))

        # Hand metrics (obs[28:30] if available)
        if len(obs) >= 30:
            # Palm position is typically computed from hand base
            palm_pos = base_pos  # Simplified - could be more sophisticated
            for i, coord in enumerate(['x', 'y', 'z']):
                self.data[f'palm_pos_{coord}'].append(palm_pos[i])
        else:
            # Fallback
            palm_pos = base_pos
            for i, coord in enumerate(['x', 'y', 'z']):
                self.data[f'palm_pos_{coord}'].append(palm_pos[i])

        # Control commands
        base_linear = action[:3]
        base_angular = action[3:6]
        tendon_commands = action[6:10]

        for i, coord in enumerate(['x', 'y', 'z']):
            self.data[f'cmd_base_linear_{coord}'].append(base_linear[i])
            self.data[f'cmd_base_angular_{coord}'].append(base_angular[i])

        for i in range(4):
            self.data[f'cmd_tendon_{i}'].append(tendon_commands[i])

        # Filtered commands
        filtered_base_linear = filtered_action[:3]
        filtered_base_angular = filtered_action[3:6]

        for i, coord in enumerate(['x', 'y', 'z']):
            self.data[f'filtered_base_linear_{coord}'].append(filtered_base_linear[i])
            self.data[f'filtered_base_angular_{coord}'].append(filtered_base_angular[i])

        # Convex hull overlap metrics
        self.data['overlap_reward'].append(reward_info.get('overlap_reward', 0.0))
        self.data['contact_penalty'].append(reward_info.get('contact_penalty', 0.0))
        self.data['proximity_reward'].append(reward_info.get('proximity_reward', 0.0))
        self.data['quality_reward'].append(reward_info.get('quality_reward', 0.0))
        self.data['overlap_volume'].append(reward_info.get('overlap_volume', 0.0))
        self.data['num_contacts'].append(reward_info.get('num_contacts', 0))
        self.data['hand_hull_volume'].append(reward_info.get('hand_hull_volume', 0.0))
        self.data['object_hull_volume'].append(reward_info.get('object_hull_volume', 0.0))

        # Total contact force (approximation)
        contact_force = np.sum(tactile_data) * 2.0  # Rough estimate
        self.data['total_contact_force'].append(contact_force)

        # Reward and success
        total_reward = sum([reward_info.get('overlap_reward', 0.0),
                           reward_info.get('contact_penalty', 0.0),
                           reward_info.get('proximity_reward', 0.0),
                           reward_info.get('quality_reward', 0.0)])

        self.data['reward'].append(total_reward)
        self.data['episode_reward'].append(episode_reward)
        self.data['success'].append(success)

    def save_to_csv(self, filename="convex_hull_overlap_training_data.csv"):
        """Save logged data to CSV"""
        df = pd.DataFrame(self.data)
        filepath = self.log_dir / filename
        df.to_csv(filepath, index=False)
        print(f"💾 Training data saved to: {filepath}")
        return filepath


class ConvexHullWandBCallback(BaseCallback):
    """Enhanced WandB callback for convex hull overlap training with comprehensive metrics and visualization"""

    def __init__(self, log_freq=50, vis_freq=500, verbose=1):
        super().__init__(verbose)
        self.log_freq = log_freq              # Log metrics every 50 steps
        self.vis_freq = vis_freq              # Upload visualizations every 500 steps
        self.last_logged_step = 0
        self.last_vis_step = 0

        # Episode tracking
        self.episode_rewards = []
        self.episode_lengths = []
        self.episode_overlaps = []
        self.episode_success_rates = []
        self.episode_contacts = []

        # Training progress tracking
        self.best_overlap = 0.0
        self.best_episode_reward = float('-inf')
        self.total_episodes = 0
        self.successful_episodes = 0

        # Metrics aggregation
        self.recent_overlaps = []
        self.recent_contacts = []
        self.recent_distances = []

        print("🔧 Enhanced WandB Callback Initialized:")
        print(f"   📊 Metric logging: every {log_freq} steps")
        print(f"   📸 Visualization upload: every {vis_freq} steps")

    def _on_step(self) -> bool:
        """Enhanced step logging with comprehensive metrics and visualization upload"""
        try:
            # Get environment info
            env_info = None
            reward_info = {}

            if hasattr(self.training_env, 'get_attr'):
                try:
                    env_info = self.training_env.get_attr('get_info')[0]
                    if env_info and 'reward_info' in env_info:
                        reward_info = env_info['reward_info']
                except:
                    pass

            # Collect current metrics
            overlap_volume = reward_info.get('overlap_volume', 0.0)
            num_contacts = reward_info.get('num_contacts', 0)

            # Update running statistics
            self.recent_overlaps.append(overlap_volume)
            self.recent_contacts.append(num_contacts)

            # Keep only recent 1000 samples
            if len(self.recent_overlaps) > 1000:
                self.recent_overlaps = self.recent_overlaps[-1000:]
                self.recent_contacts = self.recent_contacts[-1000:]

            # Update best metrics
            if overlap_volume > self.best_overlap:
                self.best_overlap = overlap_volume

            # ================== REGULAR METRIC LOGGING ==================
            if self.num_timesteps - self.last_logged_step >= self.log_freq:
                # Convert volumes to more readable units (cm³ and mm³)
                overlap_volume_cm3 = overlap_volume * 1e6  # m³ to cm³
                hand_volume_cm3 = reward_info.get('hand_hull_volume', 0.0) * 1e6
                object_volume_cm3 = reward_info.get('object_hull_volume', 0.0) * 1e6

                # Calculate overlap ratio as percentage
                overlap_ratio_percent = (overlap_volume / reward_info.get('object_hull_volume', 1e-10)) * 100

                log_dict = {
                    # Training Progress
                    'train/timestep': self.num_timesteps,
                    'train/episode_count': self.total_episodes,

                    # Reward Components
                    'reward/overlap_reward': reward_info.get('overlap_reward', 0.0),
                    'reward/contact_penalty': reward_info.get('contact_penalty', 0.0),
                    'reward/proximity_reward': reward_info.get('proximity_reward', 0.0),
                    'reward/quality_reward': reward_info.get('quality_reward', 0.0),
                    'reward/total_reward': sum([
                        reward_info.get('overlap_reward', 0.0),
                        reward_info.get('contact_penalty', 0.0),
                        reward_info.get('proximity_reward', 0.0),
                        reward_info.get('quality_reward', 0.0)
                    ]),

                    # Hull Metrics (Original m³ units)
                    'hull/overlap_volume_m3': overlap_volume,
                    'hull/hand_volume_m3': reward_info.get('hand_hull_volume', 0.0),
                    'hull/object_volume_m3': reward_info.get('object_hull_volume', 0.0),

                    # Hull Metrics (Readable cm³ units) - MAIN METRICS TO WATCH
                    'hull_cm3/overlap_volume': overlap_volume_cm3,
                    'hull_cm3/hand_volume': hand_volume_cm3,
                    'hull_cm3/object_volume': object_volume_cm3,
                    'hull_cm3/best_overlap': self.best_overlap * 1e6,

                    # Hull Ratios and Percentages
                    'hull_ratios/overlap_percentage': overlap_ratio_percent,
                    'hull_ratios/hand_to_object_ratio': hand_volume_cm3 / max(object_volume_cm3, 1e-10),
                    'hull_ratios/overlap_efficiency': overlap_volume_cm3 / max(hand_volume_cm3, 1e-10),

                    # Distance & Approach Metrics
                    'approach/distance_to_target': reward_info.get('distance_to_target', 0.0),
                    'approach/proximity_reward': reward_info.get('proximity_reward', 0.0),

                    # Contact & Safety
                    'safety/num_contacts': num_contacts,
                    'safety/contact_rate': np.mean(self.recent_contacts) if self.recent_contacts else 0.0,
                    'safety/zero_contact_rate': (np.array(self.recent_contacts) == 0).mean() if self.recent_contacts else 0.0,

                    # Performance Statistics (in cm³)
                    'stats_cm3/overlap_mean': np.mean(self.recent_overlaps) * 1e6 if self.recent_overlaps else 0.0,
                    'stats_cm3/overlap_std': np.std(self.recent_overlaps) * 1e6 if self.recent_overlaps else 0.0,
                    'stats_cm3/overlap_max': np.max(self.recent_overlaps) * 1e6 if self.recent_overlaps else 0.0,
                }

                # Print volume metrics to console every 500 steps (more frequent monitoring)
                if self.num_timesteps % 500 == 0:
                    # Calculate distance for debugging
                    try:
                        env_info = self.training_env.get_attr('latest_reward_info')[0] if hasattr(self.training_env, 'get_attr') else {}
                        distance_to_target = env_info.get('distance_to_target', 0.0)
                        proximity_reward = reward_info.get('proximity_reward', 0.0)
                    except:
                        distance_to_target = 0.0
                        proximity_reward = 0.0

                    # Get debug info from environment
                    try:
                        debug_counters = self.training_env.get_attr('debug_counters')[0]
                    except:
                        debug_counters = {'workspace_violations': 0, 'hull_calc_failures': 0, 'successful_calcs': 0, 'total_calcs': 0}

                    print(f"\n📊 Volume Metrics (Step {self.num_timesteps:,}):")
                    print(f"   Overlap:     {overlap_volume_cm3:.4f} cm³ ({overlap_ratio_percent:.2f}% of object)")
                    print(f"   Hand Hull:   {hand_volume_cm3:.4f} cm³")
                    print(f"   Object Hull: {object_volume_cm3:.4f} cm³")
                    print(f"   Distance:    {distance_to_target:.3f}m (Proximity reward: {proximity_reward:.4f})")
                    print(f"   Contacts:    {num_contacts}")
                    print(f"   Best Overlap: {self.best_overlap * 1e6:.4f} cm³")
                    print(f"   Debug: {debug_counters['successful_calcs']}/{debug_counters['total_calcs']} successful ({debug_counters['workspace_violations']} workspace violations, {debug_counters['hull_calc_failures']} hull failures)")

                # Episode statistics
                if hasattr(self.training_env, 'get_attr'):
                    try:
                        episode_rewards = self.training_env.get_attr('episode_rewards')[0]
                        if hasattr(episode_rewards, '__len__') and len(episode_rewards) > 0:
                            recent_rewards = episode_rewards[-20:] if len(episode_rewards) >= 20 else episode_rewards
                            if len(recent_rewards) > 0:
                                log_dict.update({
                                    'episode/reward_mean': np.mean(recent_rewards),
                                    'episode/reward_std': np.std(recent_rewards),
                                    'episode/reward_best': self.best_episode_reward,
                                })
                    except:
                        pass

                # Success rate calculation
                if self.total_episodes > 0:
                    log_dict['episode/success_rate'] = self.successful_episodes / self.total_episodes

                # Learning rate (if available)
                if hasattr(self.model, 'learning_rate'):
                    if callable(self.model.learning_rate):
                        current_lr = self.model.learning_rate(1.0)  # Get current LR
                    else:
                        current_lr = self.model.learning_rate
                    log_dict['train/learning_rate'] = current_lr

                wandb.log(log_dict)
                self.last_logged_step = self.num_timesteps

            # ================== VISUALIZATION UPLOAD ==================
            if (self.num_timesteps - self.last_vis_step >= self.vis_freq and
                reward_info.get('visualization_path')):

                try:
                    vis_path = reward_info['visualization_path']
                    if vis_path and os.path.exists(vis_path):
                        # Upload visualization to WandB
                        wandb.log({
                            'visualization/convex_hulls': wandb.Image(
                                vis_path,
                                caption=f"Step {self.num_timesteps}: Overlap={overlap_volume:.6f}m³, Contacts={num_contacts}"
                            )
                        })

                        if self.verbose > 0:
                            print(f"📸 Uploaded visualization: {vis_path}")

                    self.last_vis_step = self.num_timesteps
                except Exception as e:
                    if self.verbose > 0:
                        print(f"⚠️  Visualization upload failed: {e}")

        except Exception as e:
            if self.verbose > 0:
                print(f"❌ WandB logging error: {e}")

        return True

    def _on_rollout_end(self) -> None:
        """Log rollout summary statistics"""
        try:
            if self.recent_overlaps and self.recent_contacts:
                rollout_stats = {
                    'rollout/mean_overlap': np.mean(self.recent_overlaps),
                    'rollout/max_overlap': np.max(self.recent_overlaps),
                    'rollout/contact_violations': np.sum(np.array(self.recent_contacts) > 0),
                    'rollout/safe_steps': np.sum(np.array(self.recent_contacts) == 0),
                }
                wandb.log(rollout_stats)

        except Exception as e:
            if self.verbose > 0:
                print(f"Rollout logging error: {e}")

    def _on_training_end(self) -> None:
        """Log final training summary"""
        try:
            final_stats = {
                'final/best_overlap_volume': self.best_overlap,
                'final/best_episode_reward': self.best_episode_reward,
                'final/total_episodes': self.total_episodes,
                'final/successful_episodes': self.successful_episodes,
                'final/final_success_rate': self.successful_episodes / max(1, self.total_episodes),
                'final/training_timesteps': self.num_timesteps,
            }
            wandb.log(final_stats)

            print("🎯 Final Training Statistics:")
            print(f"   Best overlap volume: {self.best_overlap:.6f} m³")
            print(f"   Best episode reward: {self.best_episode_reward:.2f}")
            print(f"   Success rate: {self.successful_episodes}/{self.total_episodes} ({100*self.successful_episodes/max(1, self.total_episodes):.1f}%)")

        except Exception as e:
            if self.verbose > 0:
                print(f"Final logging error: {e}")


class TendonController:
    """Enhanced tendon controller for biomimetic control"""

    def __init__(self):
        # Tendon routing through finger joints
        self.FINGER_CHAINS = {
            "index": [8, 9, 10, 11],      # Index finger joints
            "middle": [4, 5, 6, 7],       # Middle finger joints
            "ring": [0, 1, 2, 3],         # Ring finger joints
            "thumb": [12, 13, 14, 15]     # Thumb joints
        }

        # Tendon control parameters - ULTRA-conservative for stability
        self.TENDON_FORCE_GAIN = 3.0      # Much smaller for ultra-smooth control
        self.TENDON_DAMPING = 2.5         # Higher damping for maximum stability
        self.MAX_TENDON_FORCE = 15.0      # Much lower maximum force per tendon

    def apply_tendon_forces(self, hand_id, tendon_commands):
        """Apply tendon forces to finger joints"""
        finger_names = ["ring", "middle", "index", "thumb"]  # Order matters for indexing

        for finger_idx, (finger_name, tendon_force) in enumerate(zip(finger_names, tendon_commands)):
            if finger_name in self.FINGER_CHAINS:
                joint_chain = self.FINGER_CHAINS[finger_name]

                # Clamp tendon force
                tendon_force = np.clip(tendon_force, 0.0, 1.0)
                actual_force = tendon_force * self.MAX_TENDON_FORCE

                # Apply force along joint chain (tendon routing)
                for i, joint_id in enumerate(joint_chain):
                    # Force distribution: stronger at base, weaker at tip
                    force_multiplier = 1.0 - (i * 0.1)  # 100%, 90%, 80%, 70%
                    joint_force = actual_force * force_multiplier * self.TENDON_FORCE_GAIN

                    p.setJointMotorControl2(
                        hand_id, joint_id,
                        controlMode=p.TORQUE_CONTROL,
                        force=joint_force
                    )

                    # Add damping
                    p.changeDynamics(hand_id, joint_id, jointDamping=self.TENDON_DAMPING)


class ConvexHullOverlapEnv(VecEnv):
    """
    Revolutionary convex hull overlap environment for spatial containment learning
    """

    def __init__(self,
                 num_envs=1,
                 vis=False,
                 max_steps=1000,
                 urdf_hand="/home/pralak/Space_Touch/examples/allegro_hand_description/allegro_hand_description_left_digit_fixed.urdf"):

        self.num_envs = num_envs
        self.vis = vis
        self.max_steps = max_steps
        self.sim_freq = 240.0
        self.urdf_hand = urdf_hand

        # Initialize convex hull envelopment reward system
        print("🔧 Initializing Convex Hull Envelopment Reward System...")
        reward_config = {
            'object_radius': 0.05,           # 5cm sphere
            'safety_margin': 0.025,          # 2.5cm safety clearance
            'object_hull_points': 32,        # INCREASED: Better sphere approximation (93% vs 55% accuracy)
        }

        self.reward_calculator = ConvexHullEnvelopmentReward(config=reward_config)

        # Initialize PyBullet
        self._init_pybullet()

        # Initialize hand and environment components
        self.hand = None
        self.target_sphere = None
        self.tendon_controller = None
        self.hand_spawned = False

        # Action space: 6 DOF base movement + 4 tendon forces (unchanged per request)
        action_dim = 10
        self.action_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(action_dim,), dtype=np.float32)

        # Observation: base_pos(3) + target_pos(3) + base_vel(3) + base_ang_vel(3) + finger_positions(12) + binary_tactile(4)
        # Fixed: 3+3+3+3+12+4 = 28D (removed hand metrics)
        obs_dim = 28
        self.observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32)

        super().__init__(num_envs, self.observation_space, self.action_space)

        # Environment state
        self.step_counts = np.zeros(num_envs, dtype=np.int32)
        self.episode_rewards = np.zeros(num_envs)
        self.episode_lengths = np.zeros(num_envs, dtype=np.int32)

        # Target management
        self.current_target_pos = np.array([0.25, 0.15, 0.35])  # Fixed position for initial training

        # Control filtering
        self.control_filters = [LowPassFilter() for _ in range(6)]  # 6 DOF base control

        # Data logging
        self.data_logger = None
        self.latest_reward_info = {}

        # Debug counters for volume issues
        self.debug_counters = {
            'workspace_violations': 0,
            'hull_calc_failures': 0,
            'successful_calcs': 0,
            'total_calcs': 0
        }

        # Tactile sensing
        self.tactile_sensor = None
        self._init_tactile_sensing()

        self.reset()

    def _init_pybullet(self):
        """Initialize PyBullet connection"""
        if self.vis:
            try:
                self.client_id = p.connect(p.GUI)
                p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0)
            except:
                self.client_id = p.connect(p.DIRECT)
        else:
            self.client_id = p.connect(p.DIRECT)

        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.setGravity(0, 0, -9.81)
        p.setRealTimeSimulation(0)
        p.setTimeStep(1.0/self.sim_freq)
        p.loadURDF("plane.urdf", [0, 0, 0], useFixedBase=True)

    def _init_tactile_sensing(self):
        """Initialize TACTO tactile sensors"""
        if not TACTO_AVAILABLE:
            print("⚠️  Tactile sensing disabled (TACTO not available)")
            return

        try:
            self.tactile_sensor = tacto.Sensor(
                width=120, height=160,
                config_path=tacto.get_digit_config_path(),
                visualize_gui=False
            )
            print("✅ TACTO tactile sensor initialized successfully")
        except Exception as e:
            print(f"⚠️  Tactile sensor initialization failed: {e}")
            self.tactile_sensor = None

    def set_data_logger(self, data_logger):
        """Set the data logger for this environment"""
        self.data_logger = data_logger

    def _spawn_hand(self):
        """Spawn the Allegro hand"""
        if self.hand_spawned:
            return

        try:
            # Load hand
            hand_start_pos = [0.0, 0.0, 0.3]
            hand_start_orientation = p.getQuaternionFromEuler([0, 0, 0])

            self.hand = p.loadURDF(
                self.urdf_hand,
                hand_start_pos,
                hand_start_orientation,
                useFixedBase=False,
                flags=p.URDF_USE_SELF_COLLISION
            )

            # Initialize tendon controller
            self.tendon_controller = TendonController()

            # Add tactile sensors to fingertips
            if self.tactile_sensor is not None:
                finger_links = [11, 7, 3, 15]  # Fingertip link IDs
                finger_names = ["index", "middle", "ring", "thumb"]

                for link_id, name in zip(finger_links, finger_names):
                    try:
                        self.tactile_sensor.add_camera(self.hand, link_id)
                    except Exception as e:
                        print(f"⚠️  Failed to add tactile sensor to {name}: {e}")

            self.hand_spawned = True
            print("✅ Allegro hand spawned successfully")

        except Exception as e:
            print(f"❌ Failed to spawn hand: {e}")

    def _spawn_target_object(self):
        """Spawn target sphere - FIXED: make completely static"""
        if self.target_sphere is not None:
            p.removeBody(self.target_sphere)

        # Create sphere - ZERO mass to make it static and prevent physics drift
        sphere_radius = 0.05  # 5cm
        sphere_mass = 0.0     # ZERO mass = static object

        sphere_collision_id = p.createCollisionShape(
            p.GEOM_SPHERE,
            radius=sphere_radius
        )
        sphere_visual_id = p.createVisualShape(
            p.GEOM_SPHERE,
            radius=sphere_radius,
            rgbaColor=[1, 0, 0, 0.8]  # Red, semi-transparent
        )

        self.target_sphere = p.createMultiBody(
            baseMass=sphere_mass,  # Zero mass = static
            baseCollisionShapeIndex=sphere_collision_id,
            baseVisualShapeIndex=sphere_visual_id,
            basePosition=self.current_target_pos.tolist(),
            useMaximalCoordinates=False,
            flags=p.URDF_USE_INERTIA_FROM_FILE  # Additional stability
        )

        # No spam logging for target spawn

        # Ensure the sphere is completely static and won't move
        p.changeDynamics(self.target_sphere, -1,
                        linearDamping=0,
                        angularDamping=0,
                        mass=0.0,  # Confirm zero mass
                        lateralFriction=0,
                        spinningFriction=0,
                        rollingFriction=0)

    def _get_observations(self):
        """Get current observations - FIXED array handling"""
        obs = np.zeros(28, dtype=np.float32)  # Updated to 28D (removed hand metrics)

        if self.hand is None:
            return obs

        try:
            # Hand base state - SAFE array handling
            hand_pos, hand_orn = p.getBasePositionAndOrientation(self.hand)
            hand_vel, hand_ang_vel = p.getBaseVelocity(self.hand)

            # Target position - FIXED: ensure stability and validate
            if self.target_sphere is not None:
                target_pos, _ = p.getBasePositionAndOrientation(self.target_sphere)
                # VALIDATION: Check if target has drifted (should stay near [0.25, 0.15, 0.35])
                target_distance_from_expected = np.linalg.norm(np.array(target_pos) - np.array([0.25, 0.15, 0.35]))
                if target_distance_from_expected > 0.5:
                    # Force target back to stored position (no spam logging)
                    p.resetBasePositionAndOrientation(self.target_sphere, self.current_target_pos.tolist(), [0, 0, 0, 1])
                    target_pos = self.current_target_pos
            else:
                target_pos = self.current_target_pos

            # FIXED: Force all PyBullet data to exactly 3 elements
            hand_pos_safe = np.array(hand_pos, dtype=np.float32).flatten()[:3]
            target_pos_safe = np.array(target_pos, dtype=np.float32).flatten()[:3]
            hand_vel_safe = np.array(hand_vel, dtype=np.float32).flatten()[:3]
            hand_ang_vel_safe = np.array(hand_ang_vel, dtype=np.float32).flatten()[:3]

            # Pad arrays if they're too short
            if len(hand_pos_safe) < 3:
                hand_pos_safe = np.pad(hand_pos_safe, (0, 3-len(hand_pos_safe)))
            if len(target_pos_safe) < 3:
                target_pos_safe = np.pad(target_pos_safe, (0, 3-len(target_pos_safe)))
            if len(hand_vel_safe) < 3:
                hand_vel_safe = np.pad(hand_vel_safe, (0, 3-len(hand_vel_safe)))
            if len(hand_ang_vel_safe) < 3:
                hand_ang_vel_safe = np.pad(hand_ang_vel_safe, (0, 3-len(hand_ang_vel_safe)))

            # Build observation vector with SAFE assignments
            obs[:3] = hand_pos_safe                             # Base position
            obs[3:6] = target_pos_safe                          # Target position
            obs[6:9] = hand_vel_safe                            # Base linear velocity
            obs[9:12] = hand_ang_vel_safe                       # Base angular velocity

            # Finger positions (4 fingers × 3D = 12D)
            finger_links = [11, 7, 3, 15]  # Fingertip link IDs
            finger_positions = []

            # CRITICAL FIX: Force PyBullet to update link states after physics step
            p.performCollisionDetection()  # Forces internal state update

            for i, link_id in enumerate(finger_links):
                # Force forward kinematics computation for fresh link states
                link_state = p.getLinkState(self.hand, link_id, computeForwardKinematics=1)
                finger_pos = np.array(link_state[0])[:3]  # World position, ensure 3D
                finger_positions.extend(finger_pos)

            # Ensure we have exactly 12 elements (4 fingers × 3D)
            finger_positions = finger_positions[:12]  # Truncate if too long
            while len(finger_positions) < 12:  # Pad if too short
                finger_positions.append(0.0)

            obs[12:24] = finger_positions         # 4 fingers × 3D

            # Binary tactile feedback (4D)
            tactile_data = np.zeros(4)
            if self.tactile_sensor is not None:
                try:
                    colors, depths = self.tactile_sensor.render()
                    # Simple contact detection: check for depth variations
                    for i in range(min(4, len(depths))):
                        if depths[i] is not None:
                            depth_var = np.var(depths[i])
                            tactile_data[i] = 1.0 if depth_var > 0.001 else 0.0
                except:
                    pass  # Fallback to contact point detection

            # Fallback contact detection using PyBullet
            if np.sum(tactile_data) == 0:
                contact_points = p.getContactPoints(bodyA=self.hand, bodyB=self.target_sphere)
                finger_contact_flags = np.zeros(4)

                for contact in contact_points:
                    link_id = contact[3] if contact[1] == self.hand else contact[4]
                    if link_id in finger_links:
                        finger_idx = finger_links.index(link_id)
                        finger_contact_flags[finger_idx] = 1.0

                tactile_data = finger_contact_flags

            # FIXED: Safe tactile data assignment (4D)
            tactile_safe = np.array(tactile_data, dtype=np.float32).flatten()[:4]
            if len(tactile_safe) < 4:
                tactile_safe = np.pad(tactile_safe, (0, 4-len(tactile_safe)))
            obs[24:28] = tactile_safe             # Binary tactile (4D) - now safe

            # Hand shape metrics (2D) - simplified and SAFE
            try:
                finger_pos_array = np.array(finger_positions[:12], dtype=np.float32).reshape(4, 3)
                hand_center = np.mean(finger_pos_array, axis=0)
            except:
                # Fallback if reshape fails
                finger_pos_array = np.zeros((4, 3), dtype=np.float32)
                hand_center = np.zeros(3, dtype=np.float32)

            # Hand spread: max distance between fingers
            distances = []
            for i in range(4):
                for j in range(i+1, 4):
                    try:
                        dist = np.linalg.norm(finger_pos_array[i] - finger_pos_array[j])
                        distances.append(dist)
                    except:
                        distances.append(0.0)
            hand_spread = max(distances) if distances else 0.0

            # Hand compactness: average distance to center
            try:
                finger_to_center_dists = [np.linalg.norm(pos - hand_center) for pos in finger_pos_array]
                hand_compactness = np.mean(finger_to_center_dists)
            except:
                hand_compactness = 0.0

            # Observation space reduced to 28D (removed hand metrics)
            return obs

        except Exception as e:
            print(f"Observation error: {e}")
            return obs

    def _calculate_reward(self, obs):
        """Calculate reward using convex hull overlap approach"""
        try:
            # Extract data from observation with safe handling
            base_pos = np.array(obs[:3])
            target_pos = np.array(obs[3:6])

            # EARLY CHECK: If hand is outside workspace, return zero immediately
            workspace_bounds = {'x_min': -0.5, 'x_max': 0.8, 'y_min': -0.5, 'y_max': 0.5, 'z_min': 0.1, 'z_max': 0.6}
            is_outside_workspace = (
                base_pos[0] < workspace_bounds['x_min'] or base_pos[0] > workspace_bounds['x_max'] or
                base_pos[1] < workspace_bounds['y_min'] or base_pos[1] > workspace_bounds['y_max'] or
                base_pos[2] < workspace_bounds['z_min'] or base_pos[2] > workspace_bounds['z_max']
            )

            distance = np.linalg.norm(base_pos - target_pos)
            self.debug_counters['total_calcs'] += 1

            if is_outside_workspace or distance > 2.0:
                # Hand is outside valid calculation zone - return zero but provide basic info
                self.debug_counters['workspace_violations'] += 1
                return 0.0, {
                    'error': f'Hand outside workspace (distance: {distance:.3f}m)',
                    'overlap_volume': 0.0, 'hand_hull_volume': 0.0, 'object_hull_volume': 0.0,
                    'overlap_reward': 0.0, 'contact_penalty': 0.0, 'proximity_reward': 0.0, 'quality_reward': 0.0,
                    'num_contacts': 0, 'distance_to_target': distance, 'is_success': False,
                    'consecutive_success_steps': 0, 'current_phase': self.reward_calculator.current_phase,
                    'hand_hull_valid': False, 'object_hull_valid': False
                }

            # Safe finger position extraction and reshaping
            finger_data = obs[12:24]
            if len(finger_data) < 12:
                finger_data = np.concatenate([finger_data, np.zeros(12 - len(finger_data))])
            finger_positions = finger_data[:12].reshape(4, 3)  # Ensure exactly 4×3

            # CRITICAL DIAGNOSTIC: Check if fingers are following hand base
            base_to_finger_distances = [np.linalg.norm(fp - base_pos) for fp in finger_positions]
            avg_finger_distance = np.mean(base_to_finger_distances)

            # Log finger desync every 1000 steps to track the issue
            if self.step_counts[0] % 1000 == 0 and avg_finger_distance > 0.15:  # Should be ~0.1-0.25m max for Allegro hand
                print(f"\n🚨 FINGER DESYNC DETECTED (Step {self.step_counts[0]}):")
                print(f"   Hand base pos: {base_pos}")
                print(f"   Target pos: {target_pos}")
                print(f"   Finger positions:")
                for i, fp in enumerate(finger_positions):
                    dist = np.linalg.norm(fp - base_pos)
                    print(f"     Finger {i}: {fp} (distance from base: {dist:.3f}m)")
                print(f"   Average base→finger distance: {avg_finger_distance:.3f}m (should be 0.1-0.25m)")
                print(f"   This explains zero overlap - fingers not following hand movement!")

                # Check if hand velocity is being applied
                try:
                    hand_vel, hand_ang_vel = p.getBaseVelocity(self.hand)
                    print(f"   Current hand velocity: {hand_vel}")
                    print(f"   Last action applied: {getattr(self, 'last_action', 'unknown')}")
                except:
                    print(f"   Could not get hand velocity")

            # Check for invalid finger positions
            if np.any(np.isnan(finger_positions)) or np.any(np.isinf(finger_positions)):
                print(f"⚠️  Invalid finger positions detected: {finger_positions}")
                return 0.0, {'error': 'Invalid finger positions', 'overlap_volume': 0.0, 'hand_hull_volume': 0.0, 'object_hull_volume': 0.0}

            # CRITICAL FIX: Validate finger positions are reasonable relative to hand base
            finger_distances_from_base = [np.linalg.norm(finger_pos - base_pos) for finger_pos in finger_positions]
            max_finger_distance = max(finger_distances_from_base)

            # Fingers should be within 30cm of hand base (Allegro hand arm length ~25cm)
            if max_finger_distance > 0.3:
                if not hasattr(self, 'finger_distance_violations'):
                    self.finger_distance_violations = 0
                self.finger_distance_violations += 1

                # Log every 100 violations to track the issue
                if self.finger_distance_violations % 100 == 1:
                    print(f"\n🚨 FINGER DISTANCE VIOLATION #{self.finger_distance_violations}:")
                    print(f"   Max finger distance from base: {max_finger_distance:.3f}m")
                    print(f"   Base position: {base_pos}")
                    print(f"   Finger positions: {finger_positions}")
                    print(f"   This explains zero overlap - fingers too far from base!")

                return 0.0, {
                    'error': f'Fingers too far from base (max: {max_finger_distance:.3f}m)',
                    'overlap_volume': 0.0, 'hand_hull_volume': 0.0, 'object_hull_volume': 0.0,
                    'overlap_reward': 0.0, 'contact_penalty': 0.0, 'proximity_reward': 0.0, 'quality_reward': 0.0,
                    'num_contacts': 0, 'distance_to_target': distance, 'is_success': False,
                    'consecutive_success_steps': 0, 'current_phase': self.reward_calculator.current_phase,
                    'hand_hull_valid': False, 'object_hull_valid': True
                }

            # Safe binary contact extraction
            binary_contact = obs[24:28]
            if len(binary_contact) < 4:
                binary_contact = np.concatenate([binary_contact, np.zeros(4 - len(binary_contact))])
            binary_contact = binary_contact[:4]  # Ensure exactly 4 elements

            # Calculate actual palm position from finger base positions (FIXED)
            finger_base_links = [0, 4, 8, 12]  # Base joints of ring, middle, index, thumb
            finger_base_positions = []
            try:
                for link_id in finger_base_links:
                    # Use same forced forward kinematics as finger positions
                    link_state = p.getLinkState(self.hand, link_id, computeForwardKinematics=1)
                    pos = np.array(link_state[0])[:3]  # Ensure 3D
                    if np.any(np.isnan(pos)) or np.any(np.isinf(pos)):
                        print(f"⚠️  Invalid link {link_id} position: {pos}")
                        return 0.0, {'error': 'Invalid link position', 'overlap_volume': 0.0, 'hand_hull_volume': 0.0, 'object_hull_volume': 0.0}
                    finger_base_positions.append(pos)
                palm_position = np.mean(finger_base_positions, axis=0)  # Center of finger bases

                # Check palm position validity
                if np.any(np.isnan(palm_position)) or np.any(np.isinf(palm_position)):
                    print(f"⚠️  Invalid palm position: {palm_position}")
                    return 0.0, {'error': 'Invalid palm position', 'overlap_volume': 0.0, 'hand_hull_volume': 0.0, 'object_hull_volume': 0.0}

            except Exception as e:
                print(f"⚠️  Palm position calculation failed: {e}")
                return 0.0, {'error': 'Palm calculation failed', 'overlap_volume': 0.0, 'hand_hull_volume': 0.0, 'object_hull_volume': 0.0}

            # Calculate finger base positions for enhanced 9-point hand hull (Fix #3)
            finger_base_links = [0, 4, 8, 12]  # Base joints: ring, middle, index, thumb
            finger_bases = []
            try:
                for link_id in finger_base_links:
                    # Use same forced forward kinematics as other positions
                    link_state = p.getLinkState(self.hand, link_id, computeForwardKinematics=1)
                    pos = np.array(link_state[0])[:3]  # Ensure 3D
                    if np.any(np.isnan(pos)) or np.any(np.isinf(pos)):
                        print(f"⚠️  Invalid finger base {link_id} position: {pos}")
                        finger_bases.append(np.zeros(3))  # Fallback
                    else:
                        finger_bases.append(pos)
                finger_bases = np.array(finger_bases)
            except Exception as e:
                print(f"⚠️  Finger base calculation failed: {e}")
                finger_bases = np.zeros((4, 3))  # Fallback to zeros

            # Prepare observation dictionary for reward calculator
            reward_obs = {
                'finger_positions': finger_positions,
                'finger_bases': finger_bases,        # NEW: 4 finger base positions
                'palm_position': palm_position,
                'object_pos': target_pos,
                'binary_contact': binary_contact,
                'episode_step': self.step_counts[0],  # Use first environment step count
            }

            # Calculate reward using convex hull overlap system
            try:
                total_reward, reward_info = self.reward_calculator.calculate_reward(reward_obs)

                # Validate reward info with detailed debugging
                if not reward_info or reward_info.get('error'):
                    self.debug_counters['hull_calc_failures'] += 1
                    error_msg = reward_info.get('error', 'Unknown')

                    # Only print detailed error every 100 failures to reduce spam
                    if self.debug_counters['hull_calc_failures'] % 100 == 1:
                        print(f"⚠️  Hull calculation errors detected ({self.debug_counters['hull_calc_failures']} total)")
                        print(f"   Latest error: {error_msg}")

                    return 0.0, {'error': 'Reward calc failed', 'overlap_volume': 0.0, 'hand_hull_volume': 0.0, 'object_hull_volume': 0.0}

                # Successful calculation
                self.debug_counters['successful_calcs'] += 1

                # DEBUG: Check if successful calc still has zero volumes
                if (reward_info.get('hand_hull_volume', 0) == 0.0 or
                    reward_info.get('object_hull_volume', 0) == 0.0):

                    if not hasattr(self, 'zero_vol_debug_count'):
                        self.zero_vol_debug_count = 0
                    self.zero_vol_debug_count += 1

                    # Print detailed debug every 1000 zero volume "successes"
                    if self.zero_vol_debug_count % 1000 == 1:
                        print(f"\n🔍 ZERO VOLUME IN SUCCESS #{self.zero_vol_debug_count}:")
                        print(f"   Hand hull vol: {reward_info.get('hand_hull_volume', 0):.9f}")
                        print(f"   Object hull vol: {reward_info.get('object_hull_volume', 0):.9f}")
                        print(f"   Base position: {base_pos}")
                        print(f"   Target position: {target_pos}")
                        print(f"   Finger positions:\\n{finger_positions}")
                        print(f"   Palm position: {palm_position}")
                        print(f"   Distance base-target: {np.linalg.norm(base_pos - target_pos):.3f}m")
                        print(f"   Distance palm-target: {np.linalg.norm(palm_position - target_pos):.3f}m")

                # Store for logging
                self.latest_reward_info = reward_info

                return total_reward, reward_info

            except Exception as e:
                print(f"⚠️  Reward calculator failed: {e}")
                return 0.0, {'error': 'Reward calculator exception', 'overlap_volume': 0.0, 'hand_hull_volume': 0.0, 'object_hull_volume': 0.0}

        except Exception as e:
            print(f"⚠️  Overall reward calculation error: {e}")
            import traceback
            traceback.print_exc()
            return 0.0, {'error': 'Overall calculation failed', 'overlap_volume': 0.0, 'hand_hull_volume': 0.0, 'object_hull_volume': 0.0}

    def _is_success(self, obs):
        """Check if episode is successful (high overlap, no contact)"""
        try:
            overlap_volume = self.latest_reward_info.get('overlap_volume', 0.0)
            num_contacts = self.latest_reward_info.get('num_contacts', 0)

            # Success criteria: good overlap with no contact
            success_overlap_threshold = 0.0001  # 0.1 cm³
            success = (overlap_volume >= success_overlap_threshold) and (num_contacts == 0)

            return success

        except:
            return False

    def step_async(self, actions):
        """Submit actions for async stepping"""
        self.actions = actions

    def step_wait(self):
        """Wait for step completion and return results"""
        observations = []
        rewards = []
        dones = []
        infos = []

        for env_idx in range(self.num_envs):
            action = self.actions[env_idx]

            # Ensure action is array-like
            if np.isscalar(action):
                print(f"⚠️  Action is scalar: {action}, skipping filtering")
                filtered_action = np.zeros(10)  # Default action
            else:
                # Apply control filtering to base commands
                filtered_action = action.copy()
                # Only filter if we have enough elements and filters
                if len(action) >= 6 and len(self.control_filters) >= 6:
                    for i in range(6):  # 6 DOF base control
                        filtered_action[i] = self.control_filters[i].filter(action[i])
                else:
                    print(f"⚠️  Action shape mismatch: action len={len(action)}, filters len={len(self.control_filters)}")

            # Apply actions
            self._apply_action(filtered_action)

            # Step simulation
            p.stepSimulation()

            # Get observation
            obs = self._get_observations()

            # Calculate reward
            reward, reward_info = self._calculate_reward(obs)

            # Check termination
            self.step_counts[env_idx] += 1
            self.episode_rewards[env_idx] += reward

            success = self._is_success(obs)
            timeout = self.step_counts[env_idx] >= self.max_steps
            done = success or timeout

            # Prepare info
            info = {
                'reward_info': reward_info,
                'success': success,
                'timeout': timeout,
                'episode_reward': self.episode_rewards[env_idx],
                'episode_length': self.step_counts[env_idx],
            }

            # Log data if logger available
            if self.data_logger is not None:
                self.data_logger.log_step(
                    obs, action, filtered_action, reward_info,
                    self.episode_rewards[env_idx], success
                )

            # Reset if done
            if done:
                if success:
                    print(f"✅ Episode success! Overlap: {reward_info.get('overlap_volume', 0):.6f}m³, "
                          f"Contacts: {reward_info.get('num_contacts', 0)}, "
                          f"Reward: {self.episode_rewards[env_idx]:.2f}")

                obs = self._reset_single_env(env_idx)
                info['terminal_observation'] = obs

            observations.append(obs)
            rewards.append(reward)
            dones.append(done)
            infos.append(info)

        # Ensure observations are properly shaped for VecEnv (batch dimension first)
        observations_array = np.array(observations)
        if len(observations_array.shape) == 1:
            observations_array = observations_array.reshape(1, -1)

        return observations_array, np.array(rewards), np.array(dones), infos

    def _apply_action(self, action):
        """Apply action to the hand with ultra-conservative scaling and safety checks"""
        if self.hand is None:
            return

        # Store last action for diagnostics
        self.last_action = action.copy() if hasattr(action, 'copy') else action

        try:
            # Get current state first for safety checks
            current_pos, current_orn = p.getBasePositionAndOrientation(self.hand)
            current_pos = np.array(current_pos)

            # Check if hand is already outside workspace - if so, move it back
            workspace_bounds = {
                'x_min': -0.5, 'x_max': 0.8,
                'y_min': -0.5, 'y_max': 0.5,
                'z_min': 0.1, 'z_max': 0.6
            }

            # Check if hand is outside ABSOLUTE workspace bounds
            is_outside_workspace = (
                current_pos[0] < workspace_bounds['x_min'] or current_pos[0] > workspace_bounds['x_max'] or
                current_pos[1] < workspace_bounds['y_min'] or current_pos[1] > workspace_bounds['y_max'] or
                current_pos[2] < workspace_bounds['z_min'] or current_pos[2] > workspace_bounds['z_max']
            )

            # Also check if hand is way too far from expected working area (backup check)
            expected_center = np.array([0.25, 0.15, 0.35])
            distance_from_expected = np.linalg.norm(current_pos - expected_center)

            if is_outside_workspace or distance_from_expected > 1.5:
                # Hand has escaped - reset to safe position (minimal logging)
                if not hasattr(self, 'escape_count'):
                    self.escape_count = 0
                self.escape_count += 1

                # Only print every 20th escape to reduce spam
                if self.escape_count % 20 == 1:
                    print(f"🚨 Hand escapes detected ({self.escape_count} total)")

                # Reset to safe position near expected center
                safe_pos = expected_center + np.random.uniform(-0.05, 0.05, 3)
                safe_pos[2] = max(0.3, safe_pos[2])  # Ensure above ground
                safe_pos = np.clip(safe_pos,
                                   [workspace_bounds['x_min']+0.1, workspace_bounds['y_min']+0.1, workspace_bounds['z_min']+0.05],
                                   [workspace_bounds['x_max']-0.1, workspace_bounds['y_max']-0.1, workspace_bounds['z_max']-0.05])

                p.resetBasePositionAndOrientation(self.hand, safe_pos.tolist(), current_orn)
                p.resetBaseVelocity(self.hand, [0, 0, 0], [0, 0, 0])
                return

            # DIRECT POSITION CONTROL (most reliable for floating base - bypasses PyBullet damping)
            # Scale action to movement delta: ±10mm per step at max action
            position_delta = action[:3] * 0.01  # Max ±10mm per step - direct position change
            angular_delta = action[3:6] * 0.05  # Angular change per step

            # Calculate new target position
            new_pos = current_pos + position_delta

            # Workspace bounds clamping (keep these for safety)
            clamped_pos = np.array([
                np.clip(new_pos[0], workspace_bounds['x_min'], workspace_bounds['x_max']),
                np.clip(new_pos[1], workspace_bounds['y_min'], workspace_bounds['y_max']),
                np.clip(new_pos[2], workspace_bounds['z_min'], workspace_bounds['z_max'])
            ])

            # Calculate orientation change
            current_euler = p.getEulerFromQuaternion(current_orn)
            new_euler = np.array(current_euler) + angular_delta
            new_orn = p.getQuaternionFromEuler(new_euler)

            # DIRECTLY SET NEW POSITION (no velocity needed - bypasses damping)
            p.resetBasePositionAndOrientation(self.hand, clamped_pos.tolist(), new_orn)

            # Zero out velocities for stability (let position control handle movement)
            p.resetBaseVelocity(self.hand, [0, 0, 0], [0, 0, 0])

        except Exception as e:
            print(f"⚠️  Base control error: {e}")
            # Emergency: zero all velocities
            try:
                p.resetBaseVelocity(self.hand, [0, 0, 0], [0, 0, 0])
            except:
                pass

        # CONSERVATIVE tendon control
        try:
            tendon_commands = action[6:10]
            # Scale down tendon commands too
            tendon_commands = np.clip(tendon_commands, -0.5, 0.5)  # Limit tendon range

            if self.tendon_controller is not None:
                self.tendon_controller.apply_tendon_forces(self.hand, tendon_commands)
        except Exception as e:
            print(f"⚠️  Tendon control error: {e}")

    def reset(self):
        """Reset environment"""
        obs = self._reset_single_env(0)
        # Return properly shaped observation for VecEnv (batch dimension)
        return np.array([obs])

    def _reset_single_env(self, env_idx):
        """Reset a single environment"""
        # Reset counters
        self.step_counts[env_idx] = 0
        self.episode_rewards[env_idx] = 0.0

        # Reset control filters
        for filter_obj in self.control_filters:
            filter_obj.reset()

        # Reset reward calculator
        self.reward_calculator.reset()

        # Spawn/respawn objects
        self._spawn_hand()
        self._spawn_target_object()

        # FIXED: Reset hand to initial position VERY CLOSE to target to enable overlap
        if self.hand is not None:
            # CRITICAL FIX: Start within 8cm of target to enable overlap detection
            offset = np.random.uniform(-0.08, 0.08, 3)  # ±8cm random offset
            initial_pos = self.current_target_pos + offset
            # Ensure reasonable Z height
            initial_pos[2] = max(0.30, min(0.40, initial_pos[2]))
            initial_pos = initial_pos.tolist()  # Ensure it's a list
            initial_orn = p.getQuaternionFromEuler([0, 0, 0])  # No random orientation - keep stable

            # Reset with MANDATORY verification
            try:
                p.resetBasePositionAndOrientation(self.hand, initial_pos, initial_orn)
                p.resetBaseVelocity(self.hand, [0, 0, 0], [0, 0, 0])

                # Reset joint positions to known good state
                num_joints = p.getNumJoints(self.hand)
                for joint_id in range(num_joints):
                    p.resetJointState(self.hand, joint_id, 0.0)

                # CRITICAL: Verify the reset worked (silent verification)
                actual_pos, _ = p.getBasePositionAndOrientation(self.hand)
                actual_distance = np.linalg.norm(np.array(actual_pos) - self.current_target_pos)

                # If reset failed (too far apart), force to emergency position
                if actual_distance > 0.3:
                    # Force to emergency position very close to target (silent fix)
                    emergency_pos = self.current_target_pos + np.array([0.05, 0.05, 0.05])  # 5cm offset
                    p.resetBasePositionAndOrientation(self.hand, emergency_pos.tolist(), initial_orn)

            except Exception as e:
                # Silent emergency: try to reset to target vicinity
                emergency_pos = self.current_target_pos + np.array([0.05, 0.05, 0.05])  # 5cm from target
                try:
                    p.resetBasePositionAndOrientation(self.hand, emergency_pos.tolist(), initial_orn)
                    p.resetBaseVelocity(self.hand, [0, 0, 0], [0, 0, 0])
                except:
                    pass  # Silent failure - continue training

        # Target position with MINIMAL randomization - FIXED to prevent drift
        base_target = np.array([0.25, 0.15, 0.35])  # Fixed base position
        target_noise = np.random.uniform(-0.01, 0.01, 3)  # Smaller noise: ±1cm
        self.current_target_pos = base_target + target_noise

        # Ensure target stays within reasonable bounds
        self.current_target_pos = np.clip(self.current_target_pos,
                                        [0.15, 0.05, 0.30],  # Minimum bounds
                                        [0.35, 0.25, 0.40])  # Maximum bounds

        # Reset target sphere to exact position and ensure it stays there
        if self.target_sphere is not None:
            p.resetBasePositionAndOrientation(self.target_sphere, self.current_target_pos.tolist(), [0, 0, 0, 1])
            p.resetBaseVelocity(self.target_sphere, [0, 0, 0], [0, 0, 0])

            # Force target to be completely static (redundant but ensures stability)
            p.changeDynamics(self.target_sphere, -1,
                            linearDamping=0, angularDamping=0, mass=0.0,
                            lateralFriction=0, spinningFriction=0, rollingFriction=0)

        # Stabilize simulation
        for _ in range(10):
            p.stepSimulation()

        return self._get_observations()

    def render(self, mode='human'):
        """Render the environment"""
        # PyBullet handles rendering automatically in GUI mode
        pass

    def close(self):
        """Clean up environment"""
        if hasattr(self, 'client_id'):
            p.disconnect(self.client_id)

    def get_info(self):
        """Get environment info for logging"""
        return {
            'reward_info': self.latest_reward_info,
            'num_envs': self.num_envs,
        }

    def update_reward_phase(self, phase: int):
        """Update curriculum phase in reward calculator"""
        self.reward_calculator.update_phase(phase)
        print(f"✅ Environment reward phase updated to: {phase}")

    # ================== REQUIRED VecEnv ABSTRACT METHODS ==================
    def env_is_wrapped(self, wrapper_class, indices=None):
        """Check if environment is wrapped with given wrapper class"""
        return [False] * self.num_envs

    def env_method(self, method_name, *method_args, indices=None, **method_kwargs):
        """Call environment method on specified indices"""
        if indices is None:
            indices = range(self.num_envs)
        elif isinstance(indices, int):
            indices = [indices]

        results = []
        for i in indices:
            if hasattr(self, method_name):
                method = getattr(self, method_name)
                result = method(*method_args, **method_kwargs)
                results.append(result)
            else:
                results.append(None)
        return results

    def get_attr(self, attr_name, indices=None):
        """Get attribute from environment instances"""
        if indices is None:
            indices = range(self.num_envs)
        elif isinstance(indices, int):
            indices = [indices]

        results = []
        for i in indices:
            if hasattr(self, attr_name):
                results.append(getattr(self, attr_name))
            else:
                results.append(None)
        return results

    def set_attr(self, attr_name, value, indices=None):
        """Set attribute in environment instances"""
        if indices is None:
            indices = range(self.num_envs)
        elif isinstance(indices, int):
            indices = [indices]

        for i in indices:
            setattr(self, attr_name, value)


class CurriculumCallback(BaseCallback):
    """
    Automatic curriculum phase progression based on success criteria
    """

    def __init__(self, check_freq=10000, verbose=1):
        super().__init__(verbose)
        self.check_freq = check_freq
        self.current_phase = 1
        self.phase_start_step = 0

        # Phase configurations
        self.phase_configs = {
            1: {  # APPROACH & FORMATION
                'max_steps': 300000,
                'success_criteria': {
                    # FIXED: Reduced from 0.0001 to 0.00000003 (0.1cm³ to 0.03cm³)
                    # With 9-point hand hull (~0.15cm³), 0.03cm³ overlap = 20% is achievable
                    'mean_overlap_volume': 0.00000003,      # 0.03 cm³ (was 0.1 cm³ - impossible!)
                    'mean_episode_reward': 5.0,
                    'success_rate': 0.3,                 # 30% of episodes
                },
                'name': 'Approach & Formation',
            },
            2: {  # ENVELOPMENT
                'max_steps': 350000,  # Up to 650K total
                'success_criteria': {
                    'mean_overlap_volume': 0.0003,      # 0.3 cm³
                    'overlap_ratio': 0.6,                # 60% of object engulfed
                    'mean_contacts': 1.0,                # Less than 1 contact per episode
                    'mean_episode_reward': 15.0,
                    'success_rate': 0.5,
                },
                'name': 'Envelopment',
            },
            3: {  # PRECISION
                'max_steps': 350000,  # Up to 1M total
                'success_criteria': {
                    'overlap_ratio': 0.7,
                    'mean_contacts': 0.5,
                    'sustained_success_rate': 0.7,       # 70% maintain 50+ consecutive steps
                    'mean_episode_reward': 25.0,
                },
                'name': 'Precision Soft-Capture',
            },
        }

        # Tracking for success criteria evaluation
        self.recent_episodes = []
        self.max_recent_episodes = 100

    def _on_step(self):
        # Collect episode data
        if self.locals.get('dones', [False])[0]:
            episode_info = {
                'reward': self.locals.get('rewards', [0])[0],
                'overlap_volume': 0.0,
                'num_contacts': 0,
                'is_success': False,
            }

            # Get environment info if available
            if hasattr(self.training_env, 'get_attr'):
                try:
                    env_info = self.training_env.get_attr('get_info')[0]
                    if 'reward_info' in env_info:
                        reward_info = env_info['reward_info']
                        episode_info['overlap_volume'] = reward_info.get('overlap_volume', 0.0)
                        episode_info['num_contacts'] = reward_info.get('num_contacts', 0)
                        episode_info['is_success'] = reward_info.get('is_success', False)
                        episode_info['consecutive_success_steps'] = reward_info.get('consecutive_success_steps', 0)
                except:
                    pass

            self.recent_episodes.append(episode_info)
            if len(self.recent_episodes) > self.max_recent_episodes:
                self.recent_episodes.pop(0)

        # Check phase transition every check_freq steps
        if self.num_timesteps % self.check_freq == 0:
            steps_in_phase = self.num_timesteps - self.phase_start_step
            max_steps = self.phase_configs[self.current_phase]['max_steps']

            # Check if should advance (success or timeout)
            if self._check_success_criteria() or steps_in_phase >= max_steps:
                if self.current_phase < 3:  # Don't advance past phase 3
                    self._advance_phase()

        return True

    def _check_success_criteria(self):
        """Check if current phase success criteria are met"""
        if len(self.recent_episodes) < 50:  # Need at least 50 episodes
            return False

        criteria = self.phase_configs[self.current_phase]['success_criteria']
        recent = self.recent_episodes[-100:]  # Last 100 episodes

        # Calculate metrics
        metrics = {
            'mean_overlap_volume': np.mean([ep['overlap_volume'] for ep in recent]),
            'mean_episode_reward': np.mean([ep['reward'] for ep in recent]),
            'mean_contacts': np.mean([ep['num_contacts'] for ep in recent]),
            'success_rate': np.mean([ep['is_success'] for ep in recent]),
        }

        # Phase-specific metrics
        if self.current_phase >= 2:
            # Calculate overlap ratio for Phase 2 & 3 (FIXED: object_vol was undefined in Phase 3)
            object_vol = 4/3 * np.pi * (0.05 + 0.025)**3  # Object + safety margin volume
            metrics['overlap_ratio'] = metrics['mean_overlap_volume'] / object_vol

        if self.current_phase == 3:
            # Calculate sustained success rate for Phase 3
            sustained = [ep for ep in recent if ep.get('consecutive_success_steps', 0) >= 50]
            metrics['sustained_success_rate'] = len(sustained) / len(recent)

        # Check if ALL criteria are met
        all_met = True
        for criterion, threshold in criteria.items():
            if metrics.get(criterion, 0) < threshold:
                all_met = False
                break

        # Log progress
        if self.num_timesteps % (self.check_freq * 5) == 0:  # Every 50K steps
            print(f"\n📊 Phase {self.current_phase} Progress (Step {self.num_timesteps:,}):")
            for criterion, threshold in criteria.items():
                current = metrics.get(criterion, 0)
                met = "✅" if current >= threshold else "❌"
                print(f"   {met} {criterion}: {current:.4f} (threshold: {threshold:.4f})")

        return all_met

    def _advance_phase(self):
        """Advance to next curriculum phase"""
        old_phase = self.current_phase
        self.current_phase += 1
        phase_duration = self.num_timesteps - self.phase_start_step
        self.phase_start_step = self.num_timesteps

        # Update environment reward calculator phase
        self.training_env.env_method('update_reward_phase', self.current_phase)

        # Save checkpoint
        checkpoint_path = f"checkpoints/phase{old_phase}_complete_step{self.num_timesteps}.zip"
        self.model.save(checkpoint_path)

        # Log to WandB
        wandb.log({
            'curriculum/phase': self.current_phase,
            'curriculum/phase_transition_step': self.num_timesteps,
            f'curriculum/phase{old_phase}_duration': phase_duration,
            'curriculum/remaining_budget': 1000000 - self.num_timesteps,
        })

        print(f"\n{'='*80}")
        print(f"🎓 CURRICULUM ADVANCEMENT: Phase {old_phase} → Phase {self.current_phase}")
        print(f"   Phase name: {self.phase_configs[self.current_phase]['name']}")
        print(f"   Transitioned at step: {self.num_timesteps:,}")
        print(f"   Phase {old_phase} duration: {phase_duration:,} steps")
        print(f"   Remaining budget: {1000000 - self.num_timesteps:,} steps")
        print(f"   Checkpoint saved: {checkpoint_path}")
        print(f"{'='*80}\n")

    def _on_training_end(self):
        """Log final statistics"""
        print(f"\n{'='*80}")
        print("🏁 TRAINING COMPLETED")
        print(f"   Final phase: {self.current_phase}")
        print(f"   Total timesteps: {self.num_timesteps:,}")
        print(f"{'='*80}\n")


class EnhancedWandBCallback(BaseCallback):
    """Enhanced WandB callback with curriculum-aware logging"""

    def __init__(self, log_freq=50, verbose=1):
        super().__init__(verbose)
        self.log_freq = log_freq
        self.last_logged_step = 0

        # Episode tracking
        self.episode_rewards = []
        self.episode_overlaps = []
        self.episode_contacts = []
        self.episode_distances = []

        # Best metrics
        self.best_overlap = 0.0
        self.best_episode_reward = float('-inf')

        print(f"🔧 Enhanced WandB Callback Initialized (log every {log_freq} steps)")

    def _on_step(self) -> bool:
        """Log comprehensive metrics"""
        try:
            # Get environment info
            env_info = None
            reward_info = {}

            if hasattr(self.training_env, 'get_attr'):
                try:
                    env_info = self.training_env.get_attr('get_info')[0]
                    if env_info and 'reward_info' in env_info:
                        reward_info = env_info['reward_info']
                except:
                    pass

            # Update best metrics
            overlap_vol = reward_info.get('overlap_volume', 0.0)
            if overlap_vol > self.best_overlap:
                self.best_overlap = overlap_vol

            # Regular logging
            if self.num_timesteps - self.last_logged_step >= self.log_freq:
                log_dict = {
                    # Training progress
                    'train/timestep': self.num_timesteps,

                    # Reward components
                    'reward/overlap': reward_info.get('overlap_reward', 0.0),
                    'reward/contact_penalty': reward_info.get('contact_penalty', 0.0),
                    'reward/proximity': reward_info.get('proximity_reward', 0.0),
                    'reward/quality': reward_info.get('quality_reward', 0.0),
                    'reward/sustained_bonus': reward_info.get('sustained_bonus', 0.0),
                    'reward/total': sum([
                        reward_info.get('overlap_reward', 0.0),
                        reward_info.get('contact_penalty', 0.0),
                        reward_info.get('proximity_reward', 0.0),
                        reward_info.get('quality_reward', 0.0),
                        reward_info.get('sustained_bonus', 0.0),
                    ]),

                    # Hull metrics
                    'hull/overlap_volume': overlap_vol,
                    'hull/hand_volume': reward_info.get('hand_hull_volume', 0.0),
                    'hull/object_volume': reward_info.get('object_hull_volume', 0.0),
                    'hull/best_overlap': self.best_overlap,
                    'hull/hand_valid': float(reward_info.get('hand_hull_valid', False)),
                    'hull/object_valid': float(reward_info.get('object_hull_valid', False)),

                    # Contact & Safety
                    'safety/num_contacts': reward_info.get('num_contacts', 0),
                    'safety/is_success': float(reward_info.get('is_success', False)),
                    'safety/consecutive_success': reward_info.get('consecutive_success_steps', 0),

                    # Curriculum phase
                    'curriculum/current_phase': reward_info.get('current_phase', 1),
                }

                # Add clearance error for Phase 3
                if reward_info.get('current_phase', 1) == 3:
                    log_dict['precision/clearance_error'] = reward_info.get('clearance_error', 0.0)
                    log_dict['reward/clearance'] = reward_info.get('clearance_reward', 0.0)

                # Overlap ratio
                obj_vol = reward_info.get('object_hull_volume', 1.0)
                if obj_vol > 0:
                    log_dict['hull/overlap_ratio'] = overlap_vol / obj_vol

                # Episode statistics (if available)
                if hasattr(self.training_env, 'get_attr'):
                    try:
                        episode_rewards = self.training_env.get_attr('episode_rewards')[0]
                        if hasattr(episode_rewards, '__len__') and len(episode_rewards) > 0:
                            recent = episode_rewards[-20:]
                            log_dict['episode/reward_mean'] = np.mean(recent)
                            log_dict['episode/reward_std'] = np.std(recent)

                            if np.mean(recent) > self.best_episode_reward:
                                self.best_episode_reward = np.mean(recent)
                            log_dict['episode/reward_best'] = self.best_episode_reward
                    except:
                        pass

                wandb.log(log_dict)
                self.last_logged_step = self.num_timesteps

        except Exception as e:
            if self.verbose > 0:
                print(f"⚠️  WandB logging error: {e}")

        return True


def create_convex_hull_envelopment_training():
    """Create and run convex hull envelopment training with 3-phase curriculum"""

    # Training configuration
    TOTAL_TIMESTEPS = int(1000000)  # 1M timesteps
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    LOG_DIR = f"SC1_Training_Runs/Run_{timestamp}_V5_ConvexHull_3Phase"
    MODEL_NAME = f"v5_convex_hull_3phase_{timestamp}"

    print("=" * 80)
    print("🚀 V5 CONVEX HULL ENVELOPMENT TRAINING - 3-PHASE CURRICULUM")
    print("=" * 80)
    print(f"📁 Log directory: {LOG_DIR}")
    print(f"🎯 Total timesteps: {TOTAL_TIMESTEPS:,}")
    print(f"🔬 Approach: 3-Phase Curriculum Soft-Capture")
    print(f"   Phase 1 (0-300K):   Approach & Form Valid Hull")
    print(f"   Phase 2 (300K-650K): Envelopment with Shape Quality")
    print(f"   Phase 3 (650K-1M):   Precision Soft-Capture")
    print(f"📸 Observation space: 28D (reduced from 30D)")
    print(f"🤖 Fixed palm position bug + hull validation")

    # Create directories
    os.makedirs(LOG_DIR, exist_ok=True)
    checkpoints_dir = os.path.join(LOG_DIR, "checkpoints")
    os.makedirs(checkpoints_dir, exist_ok=True)

    # Initialize WandB
    wandb_run = wandb.init(
        project="space-touch-convex-hull-3phase",
        name=f"V5_3Phase_Curriculum_{timestamp}",
        config={
            # Algorithm
            "algorithm": "PPO",
            "total_timesteps": TOTAL_TIMESTEPS,
            "learning_rate": 3e-4,
            "batch_size": 64,
            "n_steps": 2048,
            "n_epochs": 10,
            "gamma": 0.99,
            "gae_lambda": 0.95,
            "clip_range": 0.2,

            # Curriculum
            "curriculum_phases": 3,
            "phase1_max_steps": 300000,
            "phase2_max_steps": 350000,
            "phase3_max_steps": 350000,

            # Environment
            "observation_dim": 28,  # FIXED
            "action_dim": 10,
            "max_episode_steps": 500,  # Reduced from 1000

            # Object properties
            "object_radius": 0.05,
            "safety_margin": 0.025,

            # Bug fixes
            "palm_position_fix": True,
            "hull_validation": True,
            "binary_tactile": True,

            # Device
            "device": "cuda" if torch.cuda.is_available() else "cpu",
            "previous_versions": ["SC-1", "V2", "V3", "V4"],
            "breakthrough": "spatial_containment_unification",
            "safety_first": True,
        },
        tags=[
            "v5",
            "3-phase-curriculum",
            "soft-capture",
            "convex-hull",
            "envelopment",
            "bug-fixes",
            "palm-position-fixed",
            "hull-validation",
            "28d-obs",
        ],
        notes="""
        V5 Complete Implementation:
        - Fixed palm position calculation (was using wrist, now using finger base center)
        - Added hull validation (prevents degenerate/planar/linear hulls)
        - Reduced observation space to 28D (removed hand metrics)
        - 3-phase curriculum: Approach → Envelopment → Precision
        - Automatic phase progression based on success criteria
        - Binary tactile contact (sufficient for soft-capture)
        """,
    )

    try:
        # Create environment with reduced episode length for more frequent resets
        print("🔧 Creating convex hull envelopment environment...")
        env = ConvexHullOverlapEnv(num_envs=1, vis=False, max_steps=500)  # Reduced from 1000

        # Create data logger
        data_logger = DataLogger(LOG_DIR)
        env.set_data_logger(data_logger)

        # Test initial volume generation and print baseline metrics
        print("\n🔍 Initial Volume Check:")
        test_obs = env._get_observations()
        test_reward, test_info = env._calculate_reward(test_obs)

        if test_info:
            overlap_cm3 = test_info.get('overlap_volume', 0.0) * 1e6
            hand_cm3 = test_info.get('hand_hull_volume', 0.0) * 1e6
            object_cm3 = test_info.get('object_hull_volume', 0.0) * 1e6
            overlap_pct = (test_info.get('overlap_volume', 0.0) / test_info.get('object_hull_volume', 1e-10)) * 100

            print(f"   Initial Overlap:     {overlap_cm3:.4f} cm³ ({overlap_pct:.2f}% of object)")
            print(f"   Initial Hand Hull:   {hand_cm3:.4f} cm³")
            print(f"   Initial Object Hull: {object_cm3:.4f} cm³")
            print(f"   Initial Reward:      {test_reward:.4f}")
            print(f"   Hull Valid:          Hand={test_info.get('hand_hull_valid', False)}, Object={test_info.get('object_hull_valid', False)}")
        else:
            print("   ⚠️ Could not calculate initial volumes")

        # Create PPO model with optimized hyperparameters for 3-phase curriculum
        print("🧠 Creating PPO model...")
        model = PPO(
            "MlpPolicy",
            env,
            learning_rate=3e-4,
            n_steps=2048,
            batch_size=64,
            n_epochs=10,
            gamma=0.99,
            gae_lambda=0.95,
            clip_range=0.2,
            ent_coef=0.01,
            vf_coef=0.5,
            max_grad_norm=0.5,
            device="cuda" if torch.cuda.is_available() else "cpu",
            verbose=1
        )

        # Set up logging
        model.set_logger(configure(LOG_DIR, ["csv", "tensorboard"]))

        # Create curriculum callback for automatic phase progression
        curriculum_callback = CurriculumCallback(
            check_freq=10000,   # Check progress every 10K steps
            verbose=1
        )

        # Create enhanced WandB callback for comprehensive logging with visualization upload
        wandb_callback = ConvexHullWandBCallback(
            log_freq=50,        # Log metrics every 50 steps (more frequent)
            vis_freq=500,       # Upload visualizations every 500 steps
            verbose=1           # Enable progress updates
        )

        # Training parameters - 1M timesteps with curriculum progression
        checkpoint_freq = TOTAL_TIMESTEPS // 10  # 10 checkpoints for better granularity

        print(f"🏋️  Starting 3-phase curriculum training with {model.device} device...")
        print(f"💾 Checkpoints every {checkpoint_freq:,} steps")
        print(f"📚 Curriculum phases: {len(curriculum_callback.phase_configs)}")
        print(f"   Phase 1 (0-300K): Approach & Formation")
        print(f"   Phase 2 (300K-650K): Envelopment with Quality")
        print(f"   Phase 3 (650K-1M): Precision Soft-Capture")
        print(f"\n📊 Volume Tracking:")
        print(f"   Console output: Every 1,000 steps")
        print(f"   WandB metrics: Every 50 steps in 'hull_cm3/' group")
        print(f"   Watch these WandB metrics for best visibility:")
        print(f"     - hull_cm3/overlap_volume (target: 0.01-1.0 cm³)")
        print(f"     - hull_cm3/hand_volume (typical: 0.01-0.1 cm³)")
        print(f"     - hull_ratios/overlap_percentage (target: 10-70%)")

        # Training with automatic curriculum progression
        print(f"\n📈 Starting comprehensive training for {TOTAL_TIMESTEPS:,} timesteps...")

        model.learn(
            total_timesteps=TOTAL_TIMESTEPS,
            callback=[curriculum_callback, wandb_callback],  # Both callbacks
            reset_num_timesteps=False,
            progress_bar=True
        )

        # Save final model
        final_model_path = os.path.join(LOG_DIR, f"{MODEL_NAME}.zip")
        model.save(final_model_path)
        print(f"🎯 Final model saved: {final_model_path}")

        # Save training data
        data_csv_path = data_logger.save_to_csv(filename="v5_3phase_training_data.csv")
        print(f"📊 Training data saved: {data_csv_path}")

        # Log final metrics to WandB
        wandb.log({
            "training_completed": True,
            "final_timesteps": model.num_timesteps,
            "final_model_path": final_model_path,
            "final_phase": curriculum_callback.current_phase,
            "total_phase_transitions": len([p for p in range(1, 4) if curriculum_callback.current_phase >= p]),
        })

        print("\n" + "=" * 80)
        print("✅ V5 3-PHASE CONVEX HULL ENVELOPMENT TRAINING COMPLETED!")
        print("=" * 80)
        print(f"🎯 Final model: {final_model_path}")
        print(f"📊 Training data: {data_csv_path}")
        print(f"📸 Visualizations: Available via reward calculator")
        print(f"📈 TensorBoard: {LOG_DIR}")
        print(f"🎓 Final curriculum phase: {curriculum_callback.current_phase}")
        print(f"🏆 Best overlap achieved: {wandb_callback.best_overlap:.6f} m³")

        return model, env, LOG_DIR

    except Exception as e:
        print(f"❌ Training failed: {e}")
        import traceback
        traceback.print_exc()
        raise
    finally:
        wandb.finish()
        if 'env' in locals():
            env.close()


if __name__ == "__main__":
    # Run convex hull envelopment training with 3-phase curriculum
    model, env, log_dir = create_convex_hull_envelopment_training()
    print(f"\n🎉 Training complete! Check results in: {log_dir}")