#!/usr/bin/env python3
"""
V2_SC-1_Fixed_V3.py - SIMPLIFIED REWARD FUNCTION VERSION
MAJOR REDESIGN: Complete reward function overhaul to fix negative reward issues and enable soft-capture learning.

V3 CRITICAL CHANGES:
- SIMPLIFIED 3-COMPONENT REWARD: Distance + Staged Success + Tactile Engagement (vs previous 10+ components)
- POSITIVE REWARD STRUCTURE: No harsh contact penalties, distance progress always positive
- TACTILE ENCOURAGEMENT: Rewards gentle contact when close (soft-capture goal)
- EXTENDED 500K TRAINING: More time for thorough learning with simplified reward structure
- FIXED TERMINATION LOGIC: No premature episode endings on contact

TRAINING APPROACH:
500K timesteps with static targets and simplified reward function to achieve 15-25% success rate.
Focus on positive learning signals and proper incentive structure for soft-capture task.
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

# Import convex hull envelopment reward function
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

    def save_to_csv(self, filename="v2_sc1_fixed_training_data.csv"):
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


class WandBCallback(BaseCallback):
    """FIXED WandB logging callback without timestep conflicts"""

    def __init__(self, data_logger, log_freq=100, verbose=0):
        super(WandBCallback, self).__init__(verbose)
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

        # FIXED: Track last logged step to avoid conflicts
        self.last_logged_step = 0

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

                        # FIXED: Only log episode data without explicit step parameter
                        # Let WandB handle step incrementing automatically
                        wandb.log({
                            'episode/reward': ep_reward,
                            'episode/length': ep_length,
                            'episode/success': info.get('success', False),
                            'training/timesteps': self.num_timesteps,
                            'training/episodes': len(self.episode_rewards),
                        })

                    # Curriculum progress logging with transition detection
                    if 'reward_curriculum_phase' in info:
                        new_reward_phase = info['reward_curriculum_phase']
                        new_target_phase = info.get('target_curriculum_phase', 1)

                        # Detect phase transitions
                        if new_reward_phase != self.current_reward_phase:
                            self.current_reward_phase = new_reward_phase
                            self.phase_transition_timesteps.append(self.num_timesteps)
                            wandb.log({
                                'curriculum/reward_phase_transition': new_reward_phase,
                                'curriculum/transition_timestep': self.num_timesteps
                            })
                            # FIXED: Also print to console for visibility
                            print(f"\n🎓 CURRICULUM TRANSITION: Reward Phase → {new_reward_phase} at {self.num_timesteps} timesteps")

                        if new_target_phase != self.current_target_phase:
                            self.current_target_phase = new_target_phase
                            wandb.log({
                                'curriculum/target_phase_transition': new_target_phase,
                                'curriculum/transition_timestep': self.num_timesteps
                            })

                        wandb.log({
                            'curriculum/reward_phase': new_reward_phase,
                            'curriculum/target_phase': new_target_phase
                        })

                    # Success tracking
                    if 'success' in info and 'episode' in info:
                        self.episode_successes.append(float(info['success']))

                    # Step-wise metrics collection
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

            # FIXED: Enhanced aggregated metrics logging with proper step handling
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
                        'training/learning_progress': self.num_timesteps / 500000.0  # Progress to 500K
                    })

                # Curriculum effectiveness
                log_dict.update({
                    'curriculum/current_reward_phase': self.current_reward_phase,
                    'curriculum/current_target_phase': self.current_target_phase,
                    'curriculum/total_transitions': len(self.phase_transition_timesteps)
                })

                # FIXED: Log all metrics without explicit step parameter
                # Let WandB auto-increment the step counter
                if log_dict:
                    wandb.log(log_dict)
                    self.last_logged_step = self.num_timesteps

        except Exception as e:
            if self.verbose > 0:
                print(f"WandB logging error: {e}")

        return True


class V2AllegroReachingEnvFixed(VecEnv):
    """
    V2 Enhanced environment with FIXED dual curriculum learning and improved hand shape rewards
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

        # V4: CONVEX HULL ENVELOPMENT REWARD - Replace distance-based with spatial containment reward
        print("🔧 Initializing Convex Hull Envelopment Reward Function...")
        self.reward_calculator = ConvexHullEnvelopmentReward(config={
            # Use default configuration for convex hull envelopment task
        })

        # Legacy curriculum parameters (kept for compatibility but simplified reward doesn't use phases)
        self.training_timesteps = 0  # This will be updated properly
        self.reward_curriculum_phase = 1  # Not used by simplified reward but kept for logging
        self.target_curriculum_phase = 1  # Static targets only for 500K training

        # FIXED: Thresholds for 500K training - curriculum disabled for simplified reward approach
        self.REWARD_PHASE_THRESHOLDS = [0, 500000, 600000, 700000, 800000]  # Curriculum disabled
        self.TARGET_PHASE_THRESHOLDS = [500000, 600000, 700000]  # Static targets only for full 500K

        # Initialize PyBullet
        self._init_pybullet()

        self.hand = None
        self.target_sphere = None
        self.tendon_controller = None
        self.hand_spawned = False

        # Action space: 6 DOF base movement + 4 tendon forces
        action_dim = 10
        self.action_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(action_dim,), dtype=np.float32)

        # Observation: base_pos(3) + target_pos(3) + base_vel(3) + finger_positions(12) + binary_tactile(4) + hand_metrics(2)
        obs_dim = 27  # Enhanced with hand shape metrics instead of convex hull
        self.observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32)

        super().__init__(num_envs, self.observation_space, self.action_space)

        self.step_counts = np.zeros(num_envs, dtype=np.int32)
        self.episode_rewards = np.zeros(num_envs)
        self.episode_lengths = np.zeros(num_envs, dtype=np.int32)
        self.prev_distance = np.zeros(num_envs, dtype=np.float32)

        self.data_logger = None

        # Target management for curriculum
        self.current_target_pos = np.array([0.25, 0.15, 0.35])  # Static start
        self.target_velocity = np.array([0.0, 0.0, 0.0])

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

    def set_data_logger(self, data_logger):
        """Set the data logger for this environment"""
        self.data_logger = data_logger

    def update_curriculum(self, timesteps):
        """FIXED: Update curriculum learning phases based on timesteps"""
        self.training_timesteps = timesteps

        # Update reward curriculum phase
        old_reward_phase = self.reward_curriculum_phase
        for i, threshold in enumerate(self.REWARD_PHASE_THRESHOLDS):
            if timesteps >= threshold:
                self.reward_curriculum_phase = i + 1

        # Update target curriculum phase (only after reward curriculum is complete)
        old_target_phase = self.target_curriculum_phase
        if timesteps >= self.REWARD_PHASE_THRESHOLDS[-2]:  # After reward curriculum
            for i, threshold in enumerate(self.TARGET_PHASE_THRESHOLDS):
                if timesteps >= threshold:
                    self.target_curriculum_phase = i + 2  # Start from phase 2

        # Log phase changes
        if old_reward_phase != self.reward_curriculum_phase:
            print(f"🎓 Reward Curriculum Phase {old_reward_phase} → {self.reward_curriculum_phase} at {timesteps} timesteps")

        if old_target_phase != self.target_curriculum_phase:
            print(f"🎯 Target Curriculum Phase {old_target_phase} → {self.target_curriculum_phase} at {timesteps} timesteps")

    def _setup_simulation(self):
        """Setup simulation environment"""
        try:
            # Clean up existing bodies
            if self.hand_spawned or self.hand is not None:
                if self.hand is not None:
                    try:
                        p.removeBody(self.hand)
                    except:
                        pass
                    self.hand = None
                    self.hand_spawned = False

                if self.target_sphere is not None:
                    try:
                        p.removeBody(self.target_sphere)
                    except:
                        pass
                    self.target_sphere = None

            # Load hand
            if os.path.exists(self.urdf_hand):
                self.hand = p.loadURDF(
                    self.urdf_hand,
                    basePosition=[0, 0, 0.2],
                    baseOrientation=p.getQuaternionFromEuler([0, 0, 0]),
                    useFixedBase=False
                )
                self.hand_spawned = True
            else:
                # Fallback box hand
                hand_collision = p.createCollisionShape(p.GEOM_BOX, halfExtents=[0.1, 0.05, 0.02])
                hand_visual = p.createVisualShape(p.GEOM_BOX, halfExtents=[0.1, 0.05, 0.02],
                                                rgbaColor=[0.8, 0.6, 0.4, 1])
                self.hand = p.createMultiBody(baseMass=1.0, baseCollisionShapeIndex=hand_collision,
                                            baseVisualShapeIndex=hand_visual, basePosition=[0, 0, 0.2])
                self.hand_spawned = True

            # Setup target with curriculum-based properties
            target_collision = p.createCollisionShape(p.GEOM_SPHERE, radius=0.02)
            target_visual = p.createVisualShape(p.GEOM_SPHERE, radius=0.02, rgbaColor=[1, 0, 0, 1])

            # Target position based on curriculum
            self._update_target_position()

            self.target_sphere = p.createMultiBody(baseMass=0.5, baseCollisionShapeIndex=target_collision,
                                                 baseVisualShapeIndex=target_visual,
                                                 basePosition=self.current_target_pos)

            # Set target velocity based on curriculum
            p.resetBaseVelocity(self.target_sphere, linearVelocity=self.target_velocity,
                              angularVelocity=[0, 0, 0])

            # Setup joint control
            joint_inds, joint_names = [], []
            num_joints = p.getNumJoints(self.hand)
            for i in range(num_joints):
                joint_info = p.getJointInfo(self.hand, i)
                joint_type = joint_info[2]
                if joint_type in (p.JOINT_REVOLUTE, p.JOINT_PRISMATIC):
                    joint_inds.append(i)
                    joint_names.append(joint_info[1].decode())

            # Initialize tendon controller
            self.tendon_controller = TendonController(self.hand, joint_names, joint_inds)

            # Initialize fingertip links
            self.fingertip_links = []
            tip_labels = ["joint_15.0_tip", "joint_11.0_tip", "joint_7.0_tip", "joint_3.0_tip"]

            for i in range(num_joints):
                joint_info = p.getJointInfo(self.hand, i)
                joint_name = joint_info[1].decode()
                if joint_name in tip_labels:
                    self.fingertip_links.append(i)

            # Settle simulation
            for _ in range(50):
                p.stepSimulation()

        except Exception as e:
            print(f"Error setting up simulation: {e}")
            raise

    def _update_target_position(self):
        """Update target position and velocity based on curriculum"""
        # Base static position
        base_pos = np.array([0.25, 0.15, 0.35])

        if self.target_curriculum_phase == 1:
            # Phase 1: Static target
            self.current_target_pos = base_pos
            self.target_velocity = np.array([0.0, 0.0, 0.0])
        elif self.target_curriculum_phase == 2:
            # Phase 2: Slow moving target
            self.current_target_pos = base_pos + np.random.uniform(-0.05, 0.05, 3)
            self.target_velocity = np.array([0.01, 0.0, 0.0])  # Slow X movement
        else:
            # Phase 3: Dynamic target
            self.current_target_pos = base_pos + np.random.uniform(-0.1, 0.1, 3)
            self.target_velocity = np.random.uniform(-0.02, 0.02, 3)  # Random movement

    def _get_finger_positions(self):
        """Get positions of all fingertips"""
        finger_positions = []

        if self.hand is None:
            return np.zeros(12)

        tip_labels = ["joint_15.0_tip", "joint_11.0_tip", "joint_7.0_tip", "joint_3.0_tip"]
        num_joints = p.getNumJoints(self.hand)

        for tip_label in tip_labels:
            found = False
            for i in range(num_joints):
                joint_info = p.getJointInfo(self.hand, i)
                joint_name = joint_info[1].decode()

                if joint_name == tip_label:
                    try:
                        link_state = p.getLinkState(self.hand, i)
                        finger_positions.extend(link_state[0])
                        found = True
                        break
                    except:
                        continue

            if not found:
                base_pos, _ = p.getBasePositionAndOrientation(self.hand)
                finger_positions.extend(base_pos)

        return np.array(finger_positions)

    def _get_palm_position(self):
        """Get palm center position for convex hull calculations"""
        if self.hand is None:
            return np.zeros(3)

        try:
            # Get base position as approximation of palm center
            # In a more detailed implementation, this could be a specific palm link
            base_pos, _ = p.getBasePositionAndOrientation(self.hand)
            return np.array(base_pos)
        except:
            # Fallback to zeros if getting base position fails
            return np.zeros(3)

    def _get_binary_tactile_feedback(self):
        """Get binary tactile contact feedback"""
        if self.hand is None or not self.fingertip_links:
            return np.zeros(4)

        binary_contacts = []

        for link_idx in self.fingertip_links:
            contact_points = p.getContactPoints(bodyA=self.hand, linkIndexA=link_idx)
            has_contact = 1.0 if len(contact_points) > 0 else 0.0
            binary_contacts.append(has_contact)

        while len(binary_contacts) < 4:
            binary_contacts.append(0.0)

        return np.array(binary_contacts[:4])

    def _compute_hand_shape_features(self, finger_positions):
        """FIXED: Compute improved hand shape features instead of problematic convex hull"""
        try:
            # Reshape finger positions to 4x3 array
            points = finger_positions.reshape(4, 3)

            # Compute hand spread (distance between extreme fingers)
            distances = []
            for i in range(len(points)):
                for j in range(i+1, len(points)):
                    distances.append(np.linalg.norm(points[i] - points[j]))

            hand_spread = np.max(distances) if distances else 0.0

            # Compute hand compactness (how close fingers are to center)
            center = np.mean(points, axis=0)
            compactness = np.mean([np.linalg.norm(point - center) for point in points])

            return hand_spread, compactness
        except:
            # Return default values if computation fails
            return 0.15, 0.05  # Reasonable default values

    def _get_observation(self):
        """Get current observation with improved hand shape features"""
        try:
            if self.hand is None:
                obs = np.zeros(self.observation_space.shape[0])
                return np.expand_dims(obs.astype(np.float32), axis=0)

            # Get basic states
            base_pos, _ = p.getBasePositionAndOrientation(self.hand)
            base_vel, _ = p.getBaseVelocity(self.hand)
            target_pos, _ = p.getBasePositionAndOrientation(self.target_sphere)

            base_pos = np.array(base_pos)
            base_vel = np.array(base_vel)
            target_pos = np.array(target_pos)

            # Get finger positions and tactile feedback
            finger_positions = self._get_finger_positions()
            binary_tactile = self._get_binary_tactile_feedback()

            # Compute hand shape features
            hand_spread, hand_compactness = self._compute_hand_shape_features(finger_positions)

            # Combine into observation (27D)
            obs = np.concatenate([
                base_pos,           # 3D
                target_pos,         # 3D
                base_vel,           # 3D
                finger_positions,   # 12D
                binary_tactile,     # 4D
                [hand_spread],      # 1D - hand spread metric
                [hand_compactness]  # 1D - hand compactness metric
            ])

            return np.expand_dims(obs.astype(np.float32), axis=0)

        except Exception as e:
            print(f"Error getting observation: {e}")
            return np.zeros((1, self.observation_space.shape[0]), dtype=np.float32)

    def _compute_improved_reward(self, base_pos, target_pos, tendon_forces, binary_tactile,
                                base_vel, hand_spread, hand_compactness):
        """
        V4 CONVEX HULL ENVELOPMENT REWARD: Use convex hull spatial containment for soft-capture task

        PARADIGM SHIFT: Replace distance-based rewards with spatial envelopment rewards:
        1. Hull Formation Reward - Encourages creating valid convex hull with fingers + palm
        2. Target Proximity Reward - Rewards positioning hand center near target (not touching!)
        3. Envelopment Reward - MASSIVE bonus for target being inside convex hull
        4. Sustained Envelopment Bonus - SUCCESS when target stays inside hull for consecutive steps

        Expected range: [0.0, 105.0] - Pure spatial containment task
        """
        # Get finger positions with error handling
        try:
            finger_positions = self._get_finger_positions()

            # Ensure finger_positions is the right shape and type
            if finger_positions.size != 12:
                print(f"Warning: Expected 12 finger position values, got {finger_positions.size}")
                # Use zeros as fallback
                finger_positions = np.zeros(12)

            finger_positions_2d = finger_positions.reshape(4, 3)

            # Get palm position for convex hull
            palm_position = self._get_palm_position()

            # Ensure palm_position is correct shape
            if palm_position.size != 3:
                print(f"Warning: Expected 3 palm position values, got {palm_position.size}")
                palm_position = np.zeros(3)

            # Get hand center for proximity calculation
            hand_center = np.array(base_pos)  # Ensure it's a numpy array

            # Prepare observation dictionary for convex hull reward function
            obs_dict = {
                'finger_positions': finger_positions_2d,  # (4, 3) array
                'palm_position': palm_position,           # (3,) array
                'target_position': np.array(target_pos),  # (3,) array - ensure numpy array
                'hand_center': hand_center,               # (3,) array
            }

            # Calculate reward using convex hull envelopment reward function
            total_reward, reward_info = self.reward_calculator.calculate_reward(obs_dict)

        except Exception as reward_error:
            print(f"Error in convex hull reward calculation: {reward_error}")
            # Return fallback values
            total_reward = 0.0
            reward_info = {
                'hull_formation_reward': 0.0,
                'proximity_reward': 0.0,
                'envelopment_reward': 0.0,
                'sustained_envelopment_reward': 0.0,
                'hull_valid': False,
                'hull_volume': 0.0,
                'is_enveloped': False,
                'clearance': 0.0,
                'consecutive_steps': 0,
                'distance_to_hull_center': 0.0,
            }

        # Extract individual components for logging (maintaining compatibility with existing logging)
        hull_formation_reward = reward_info['hull_formation_reward']
        proximity_reward = reward_info['proximity_reward']
        envelopment_reward = reward_info['envelopment_reward']
        sustained_envelopment_reward = reward_info['sustained_envelopment_reward']

        # Map to legacy component names for logging compatibility
        distance_reward = proximity_reward  # Proximity is similar to distance reward
        success_bonus = envelopment_reward + sustained_envelopment_reward  # Combine envelopment rewards

        # Legacy components set to 0 (no longer used in convex hull paradigm)
        tendon_efficiency_reward = 0.0
        movement_penalty = 0.0
        hand_shape_reward = hull_formation_reward  # Hull formation is related to hand shape

        return {
            'total_reward': total_reward,  # From convex hull reward calculation
            'distance_reward': distance_reward,  # Mapped to proximity reward
            'tendon_efficiency_reward': tendon_efficiency_reward,  # 0.0 (legacy)
            'tactile_contact_reward': 0.0,  # Not used in convex hull paradigm
            'movement_penalty': movement_penalty,  # 0.0 (legacy)
            'hand_shape_reward': hand_shape_reward,  # Hull formation reward
            'success_bonus': success_bonus,  # Combined envelopment rewards

            # Additional info from convex hull reward for enhanced logging
            'hull_formation_reward': hull_formation_reward,
            'proximity_reward': proximity_reward,
            'envelopment_reward': envelopment_reward,
            'sustained_envelopment_reward': sustained_envelopment_reward,
            'hull_valid': reward_info['hull_valid'],
            'hull_volume': reward_info['hull_volume'],
            'is_enveloped': reward_info['is_enveloped'],
            'clearance': reward_info['clearance'],
            'consecutive_envelopment_steps': reward_info['consecutive_steps'],
            'distance_to_hull_center': reward_info['distance_to_hull_center'],

            # Legacy compatibility
            'consecutive_success_steps': reward_info['consecutive_steps'],
            'success_stage': 0,  # Not applicable in convex hull paradigm
            'in_success_zone': reward_info['is_enveloped'],  # Map envelopment to success zone
        }

    def step_wait(self):
        """Execute one step with FIXED curriculum learning and improved reward"""
        self.step_counts += 1
        self.episode_lengths += 1

        try:
            # DEBUG: Check actions shape
            if len(self.actions) == 0 or self.actions[0] is None:
                obs = self._get_observation()
                return obs, np.array([-1.0], dtype=np.float32), np.array([True]), [{"error": "No actions provided"}]

            actions = self.actions[0]

            # DEBUG: Ensure actions is the right shape
            if not isinstance(actions, np.ndarray):
                actions = np.array(actions)
            if actions.size < 10:
                obs = self._get_observation()
                return obs, np.array([-1.0], dtype=np.float32), np.array([True]), [{"error": f"Actions wrong size: {actions.size}"}]

            if self.hand is None:
                obs = self._get_observation()
                return obs, np.array([-1.0], dtype=np.float32), np.array([True]), [{"error": "Hand not loaded"}]

            # Split actions
            base_actions = actions[:6]
            tendon_actions = actions[6:10]

            # Apply Butterworth filtering to base actions
            filtered_base_actions = self.tendon_controller.apply_control_filtering(base_actions)

            # Apply base movement with INCREASED gains for faster learning
            linear_vel = filtered_base_actions[:3] * 0.5  # FIXED: Increased from 0.25 to 0.5
            angular_vel = filtered_base_actions[3:6] * 1.0  # FIXED: Increased from 0.6 to 1.0

            p.resetBaseVelocity(self.hand, linearVelocity=linear_vel, angularVelocity=angular_vel)

            # Apply tendon forces
            tendon_forces = (tendon_actions + 1.0) / 2.0
            self.current_tendon_forces = tendon_forces

            tendon_force_dict = {
                "index": tendon_forces[0],
                "middle": tendon_forces[1],
                "ring": tendon_forces[2],
                "thumb": tendon_forces[3]
            }

            torques = self.tendon_controller.compute_tendon_torques(tendon_force_dict)

            if len(torques) > 0:
                p.setJointMotorControlArray(bodyUniqueId=self.hand,
                                          jointIndices=self.tendon_controller.joint_indices,
                                          controlMode=p.TORQUE_CONTROL, forces=torques.tolist())

            p.stepSimulation()

            # Get current states with error checking
            try:
                base_pos, base_orn = p.getBasePositionAndOrientation(self.hand)
                base_pos = np.array(base_pos)
            except Exception as e:
                print(f"Error getting base position: {e}")
                base_pos = np.zeros(3)

            try:
                base_vel, base_ang_vel = p.getBaseVelocity(self.hand)
                base_vel = np.array(base_vel)
                base_ang_vel = np.array(base_ang_vel)
                base_vel_combined = np.array(list(base_vel) + list(base_ang_vel))
            except Exception as e:
                print(f"Error getting base velocity: {e}")
                base_vel = np.zeros(3)
                base_ang_vel = np.zeros(3)
                base_vel_combined = np.zeros(6)

            try:
                target_pos, _ = p.getBasePositionAndOrientation(self.target_sphere)
                target_pos = np.array(target_pos)
            except Exception as e:
                print(f"Error getting target position: {e}")
                target_pos = np.zeros(3)

            # Get additional states with error checking
            try:
                finger_positions = self._get_finger_positions()
                if finger_positions.size != 12:
                    print(f"Warning: Finger positions wrong size: {finger_positions.size}")
                    finger_positions = np.zeros(12)
            except Exception as e:
                print(f"Error getting finger positions: {e}")
                finger_positions = np.zeros(12)

            try:
                binary_tactile = self._get_binary_tactile_feedback()
                if binary_tactile.size != 4:
                    print(f"Warning: Binary tactile wrong size: {binary_tactile.size}")
                    binary_tactile = np.zeros(4)
            except Exception as e:
                print(f"Error getting binary tactile: {e}")
                binary_tactile = np.zeros(4)

            try:
                hand_spread, hand_compactness = self._compute_hand_shape_features(finger_positions)
            except Exception as e:
                print(f"Error computing hand shape: {e}")
                hand_spread, hand_compactness = 0.0, 0.0

            # Compute improved reward with curriculum
            reward_components = self._compute_improved_reward(
                base_pos, target_pos, tendon_forces, binary_tactile,
                base_vel_combined, hand_spread, hand_compactness
            )

            total_reward = reward_components['total_reward']
            distance = np.linalg.norm(base_pos - target_pos)

            # V4 CONVEX HULL SUCCESS CRITERIA: Use envelopment-based success requirements
            # Get success criteria from convex hull reward function
            success_criteria = self.reward_calculator.get_success_criteria()
            consecutive_steps = reward_components.get('consecutive_envelopment_steps', 0)
            is_enveloped = reward_components.get('is_enveloped', False)

            # Success only when target is enveloped for required consecutive steps
            success = (is_enveloped and
                      consecutive_steps >= success_criteria['min_consecutive_steps'])

            self.episode_rewards[0] += total_reward

            # V4 CONVEX HULL TERMINATION CONDITIONS: Focus on spatial containment, not distance
            # PARADIGM SHIFT: Terminate based on envelopment success, not distance thresholds
            dones = np.array([
                # Only terminate on: max steps, sustained envelopment success, or clear failure
                self.step_counts[0] >= self.max_steps or
                success or  # Only sustained envelopment (not immediate spatial contact)
                distance > 5.0 or  # Extremely far from target = impossible to envelop
                base_pos[2] < 0.0   # Hand fell through ground = clear failure
                # REMOVED: distance-based termination (not relevant for spatial containment)
                # FOCUS: Allow agent to learn spatial positioning and hull formation
            ])

            # Log detailed data if logger is available
            if self.data_logger is not None:
                finger_pos_reshaped = finger_positions.reshape(4, 3)
                self.step_data = {
                    "base_pos_x": base_pos[0], "base_pos_y": base_pos[1], "base_pos_z": base_pos[2],
                    "base_vel_x": base_vel[0], "base_vel_y": base_vel[1], "base_vel_z": base_vel[2],
                    "base_ang_vel_x": base_ang_vel[0], "base_ang_vel_y": base_ang_vel[1], "base_ang_vel_z": base_ang_vel[2],
                    "finger_pos_1_x": finger_pos_reshaped[0,0], "finger_pos_1_y": finger_pos_reshaped[0,1], "finger_pos_1_z": finger_pos_reshaped[0,2],
                    "finger_pos_2_x": finger_pos_reshaped[1,0], "finger_pos_2_y": finger_pos_reshaped[1,1], "finger_pos_2_z": finger_pos_reshaped[1,2],
                    "finger_pos_3_x": finger_pos_reshaped[2,0], "finger_pos_3_y": finger_pos_reshaped[2,1], "finger_pos_3_z": finger_pos_reshaped[2,2],
                    "finger_pos_4_x": finger_pos_reshaped[3,0], "finger_pos_4_y": finger_pos_reshaped[3,1], "finger_pos_4_z": finger_pos_reshaped[3,2],
                    "hand_spread": hand_spread, "hand_compactness": hand_compactness,
                    "target_x": target_pos[0], "target_y": target_pos[1], "target_z": target_pos[2],
                    "distance_to_target": distance,
                    "tendon_force_index": tendon_forces[0], "tendon_force_middle": tendon_forces[1],
                    "tendon_force_ring": tendon_forces[2], "tendon_force_thumb": tendon_forces[3],
                    "control_linear_x": base_actions[0], "control_linear_y": base_actions[1], "control_linear_z": base_actions[2],
                    "control_angular_x": base_actions[3], "control_angular_y": base_actions[4], "control_angular_z": base_actions[5],
                    "filtered_linear_x": filtered_base_actions[0], "filtered_linear_y": filtered_base_actions[1], "filtered_linear_z": filtered_base_actions[2],
                    "filtered_angular_x": filtered_base_actions[3], "filtered_angular_y": filtered_base_actions[4], "filtered_angular_z": filtered_base_actions[5],
                    "binary_tactile_1": binary_tactile[0], "binary_tactile_2": binary_tactile[1],
                    "binary_tactile_3": binary_tactile[2], "binary_tactile_4": binary_tactile[3],
                    "num_active_fingers": np.sum(binary_tactile),
                    "reward": total_reward,
                    "distance_reward": reward_components['distance_reward'],
                    "tendon_efficiency_reward": reward_components['tendon_efficiency_reward'],
                    "tactile_contact_reward": reward_components['tactile_contact_reward'],
                    "movement_penalty": reward_components['movement_penalty'],
                    "hand_shape_reward": reward_components['hand_shape_reward'],
                    "success_bonus": reward_components['success_bonus'],
                    "reward_curriculum_phase": self.reward_curriculum_phase,
                    "target_curriculum_phase": self.target_curriculum_phase,
                    "training_timesteps": self.training_timesteps,
                    "success": success,
                }

                self.data_logger.log_step(self.step_data)

            # Info for callbacks (FIXED: Added curriculum tracking)
            infos = [{
                "success": success,
                "distance": distance,
                "reward_curriculum_phase": self.reward_curriculum_phase,
                "target_curriculum_phase": self.target_curriculum_phase,
                "training_timesteps": self.training_timesteps,  # FIXED: Added timestep tracking
                "reward_components": reward_components,
                "tendon_forces": tendon_force_dict,
                "num_active_fingers": np.sum(binary_tactile),
                "hand_spread": hand_spread,
                "hand_compactness": hand_compactness
            }]

            if dones[0]:
                infos[0]["episode"] = {
                    "r": float(self.episode_rewards[0]),
                    "l": int(self.episode_lengths[0]),
                    "t": time.time()
                }

            obs = self.reset_(dones) if dones[0] else self._get_observation()
            return obs, np.array([total_reward], dtype=np.float32), dones, infos

        except Exception as e:
            print(f"Error in step: {e}")
            obs = self._get_observation()
            return obs, np.array([-1.0], dtype=np.float32), np.array([True]), [{"error": str(e)}]

    def reset_(self, dones):
        """Reset environment"""
        if np.any(dones):
            # V3: Reset simplified reward function for new episode
            self.reward_calculator.reset()

            # Clean up and setup new simulation
            if self.hand is not None:
                try:
                    p.removeBody(self.hand)
                except:
                    pass
                self.hand = None
                self.hand_spawned = False

            if self.target_sphere is not None:
                try:
                    p.removeBody(self.target_sphere)
                except:
                    pass
                self.target_sphere = None

            # Reset tendon controller filters
            if self.tendon_controller is not None:
                self.tendon_controller.reset_filters()

            # Setup new simulation
            self._setup_simulation()

            # Reset counters
            self.step_counts[dones] = 0
            self.episode_rewards[dones] = 0
            self.episode_lengths[dones] = 0
            self.current_tendon_forces = np.zeros(4)

        return self._get_observation()

    def reset(self):
        return self.reset_(np.ones(self.num_envs, dtype=bool))

    def step_async(self, actions):
        self.actions = actions

    def close(self):
        try:
            if self.hand is not None:
                p.removeBody(self.hand)
            if self.target_sphere is not None:
                p.removeBody(self.target_sphere)
            p.disconnect(self.client_id)
        except:
            pass

    def seed(self, seed=None):
        if seed is not None:
            np.random.seed(seed)
        return [seed]

    # VecEnv required methods
    def get_attr(self, attr_name, indices=None):
        return [getattr(self, attr_name, None)]

    def set_attr(self, attr_name, value, indices=None):
        setattr(self, attr_name, value)

    def env_method(self, method_name, *method_args, indices=None, **method_kwargs):
        if hasattr(self, method_name):
            method = getattr(self, method_name)
            return [method(*method_args, **method_kwargs)]
        return [None]

    def env_is_wrapped(self, wrapper_class, indices=None):
        return [False] * self.num_envs


class CheckpointCallback(BaseCallback):
    """Enhanced checkpoint callback with curriculum tracking - FIXED for custom VecEnv"""

    def __init__(self, save_freq=60000, save_path="./checkpoints/", verbose=1):
        super(CheckpointCallback, self).__init__(verbose)
        self.save_freq = save_freq
        self.save_path = Path(save_path)
        self.save_path.mkdir(parents=True, exist_ok=True)

    def _on_step(self) -> bool:
        if self.n_calls % self.save_freq == 0:
            checkpoint_path = self.save_path / f"checkpoint_{self.num_timesteps}.zip"
            self.model.save(str(checkpoint_path))
            if self.verbose > 0:
                print(f"📦 Checkpoint saved at {self.num_timesteps} timesteps: {checkpoint_path}")

            # FIXED: Directly call update_curriculum on the custom VecEnv
            # The environment IS the VecEnv, not wrapped
            if hasattr(self.training_env, 'update_curriculum'):
                self.training_env.update_curriculum(self.num_timesteps)
                if self.verbose > 0:
                    print(f"✓ Curriculum updated: Reward Phase {self.training_env.reward_curriculum_phase}, Target Phase {self.training_env.target_curriculum_phase}")
            else:
                if self.verbose > 0:
                    print("⚠️ Warning: Environment does not have update_curriculum method")

        return True


def create_comprehensive_plots(csv_file, test_scenarios=None):
    """Create comprehensive analysis plots from training/test data"""
    print(f"Creating comprehensive plots from {csv_file}...")

    try:
        df = pd.read_csv(csv_file)
        print(f"Loaded data with {len(df)} rows and {len(df.columns)} columns")

        if len(df) == 0:
            print("No data to plot")
            return

        plot_dir = csv_file.parent / "plots"
        plot_dir.mkdir(exist_ok=True)

        plt.style.use('default')
        plt.rcParams['figure.figsize'] = (15, 10)

        # Sample data for plotting performance
        if len(df) > 10000:
            step_size = len(df) // 10000
            df_plot = df.iloc[::step_size].copy()
        else:
            df_plot = df.copy()

        step_vals = df_plot['step'].values

        # 1. Fixed Performance Overview
        fig, axes = plt.subplots(3, 2, figsize=(18, 15))
        fig.suptitle('V2 SC-1 FIXED Comprehensive Performance Analysis', fontsize=16, fontweight='bold')

        # Reward components over time
        if all(col in df_plot.columns for col in ['distance_reward', 'tendon_efficiency_reward', 'tactile_contact_reward', 'movement_penalty', 'hand_shape_reward']):
            axes[0,0].plot(step_vals, df_plot['reward'], label='Total Reward', linewidth=1.5, color='black')
            axes[0,0].plot(step_vals, df_plot['distance_reward'], label='Distance', alpha=0.8)
            axes[0,0].plot(step_vals, df_plot['tendon_efficiency_reward'], label='Tendon Efficiency', alpha=0.8)
            axes[0,0].plot(step_vals, df_plot['tactile_contact_reward'], label='Tactile Contact', alpha=0.8)
            axes[0,0].plot(step_vals, df_plot['movement_penalty'], label='Movement Penalty', alpha=0.8)
            axes[0,0].plot(step_vals, df_plot['hand_shape_reward'], label='Hand Shape', alpha=0.8)
            axes[0,0].set_title('FIXED Reward Components (V2)')
            axes[0,0].set_xlabel('Step')
            axes[0,0].set_ylabel('Reward')
            axes[0,0].legend()
            axes[0,0].grid(True, alpha=0.3)

        # Fixed Curriculum progression
        if 'reward_curriculum_phase' in df_plot.columns:
            axes[0,1].plot(step_vals, df_plot['reward_curriculum_phase'], 'o-', markersize=2, label='Reward Phase')
            if 'target_curriculum_phase' in df_plot.columns:
                axes[0,1].plot(step_vals, df_plot['target_curriculum_phase'], 's-', markersize=2, label='Target Phase')
            axes[0,1].set_title('FIXED Dual Curriculum Learning Progress')
            axes[0,1].set_xlabel('Step')
            axes[0,1].set_ylabel('Curriculum Phase')
            axes[0,1].legend()
            axes[0,1].grid(True, alpha=0.3)

        # Distance and success tracking
        axes[1,0].plot(step_vals, df_plot['distance_to_target'], alpha=0.7, linewidth=0.8, color='blue', label='Distance')
        axes[1,0].axhline(y=0.08, color='red', linestyle='--', label='Success Threshold (0.08m)')
        axes[1,0].set_title('Distance to Target Over Time')
        axes[1,0].set_ylabel('Distance (m)')
        axes[1,0].set_xlabel('Step')
        axes[1,0].legend()
        axes[1,0].grid(True, alpha=0.3)

        # Hand shape analysis
        if 'hand_spread' in df_plot.columns:
            axes[1,1].plot(step_vals, df_plot['hand_spread'], alpha=0.7, linewidth=0.8, color='green', label='Hand Spread')
            if 'hand_compactness' in df_plot.columns:
                axes[1,1].plot(step_vals, df_plot['hand_compactness'], alpha=0.7, linewidth=0.8, color='orange', label='Hand Compactness')
            axes[1,1].axhline(y=0.12, color='red', linestyle='--', alpha=0.5, label='Target Spread')
            axes[1,1].axhline(y=0.06, color='orange', linestyle='--', alpha=0.5, label='Target Compactness')
            axes[1,1].set_title('FIXED Hand Shape Metrics')
            axes[1,1].set_ylabel('Distance (m)')
            axes[1,1].set_xlabel('Step')
            axes[1,1].legend()
            axes[1,1].grid(True, alpha=0.3)

        # Control smoothing analysis (filtered vs raw)
        if all(col in df_plot.columns for col in ['control_linear_x', 'filtered_linear_x']):
            sample_steps = step_vals[-1000:] if len(step_vals) > 1000 else step_vals
            sample_raw = df_plot['control_linear_x'].values[-1000:] if len(step_vals) > 1000 else df_plot['control_linear_x'].values
            sample_filtered = df_plot['filtered_linear_x'].values[-1000:] if len(step_vals) > 1000 else df_plot['filtered_linear_x'].values

            axes[2,0].plot(sample_steps, sample_raw, alpha=0.7, label='Raw Control', linewidth=0.8)
            axes[2,0].plot(sample_steps, sample_filtered, alpha=0.7, label='Butterworth Filtered', linewidth=0.8)
            axes[2,0].set_title('IMPROVED Control Smoothing')
            axes[2,0].set_xlabel('Step')
            axes[2,0].set_ylabel('Linear X Control')
            axes[2,0].legend()
            axes[2,0].grid(True, alpha=0.3)

        # Tactile engagement analysis
        if 'num_active_fingers' in df_plot.columns:
            axes[2,1].plot(step_vals, df_plot['num_active_fingers'], alpha=0.7, linewidth=0.8, color='purple')
            axes[2,1].axhline(y=2, color='red', linestyle='--', label='Success Threshold (2 fingers)')
            axes[2,1].set_title('Tactile Engagement (Active Fingers)')
            axes[2,1].set_ylabel('Number of Active Fingers')
            axes[2,1].set_xlabel('Step')
            axes[2,1].legend()
            axes[2,1].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(plot_dir / 'v2_fixed_comprehensive_analysis.png', dpi=150, bbox_inches='tight')
        plt.close()

        print(f"✅ V2 FIXED comprehensive plots saved to: {plot_dir}")

        # Summary statistics
        print("\n" + "=" * 80)
        print("V2 SC-1 FIXED TRAINING SUMMARY")
        print("=" * 80)
        print(f"Total steps: {len(df)}")

        if len(df) > 0:
            print(f"Final distance to target: {df['distance_to_target'].iloc[-1]:.4f} m")
            print(f"Average total reward: {df['reward'].mean():.4f}")

            if 'hand_spread' in df.columns:
                print(f"Final hand spread: {df['hand_spread'].iloc[-1]:.4f}")
                print(f"Final hand compactness: {df['hand_compactness'].iloc[-1]:.4f}")

            if 'num_active_fingers' in df.columns:
                print(f"Average active fingers: {df['num_active_fingers'].mean():.2f}")

            if 'success' in df.columns:
                print(f"Overall success rate: {df['success'].mean()*100:.1f}%")

            if 'reward_curriculum_phase' in df.columns:
                final_reward_phase = df['reward_curriculum_phase'].iloc[-1]
                print(f"Final reward curriculum phase: {final_reward_phase}")

            if 'target_curriculum_phase' in df.columns:
                final_target_phase = df['target_curriculum_phase'].iloc[-1]
                print(f"Final target curriculum phase: {final_target_phase}")

        print("=" * 80)

    except Exception as e:
        print(f"❌ Error creating plots: {e}")
        import traceback
        traceback.print_exc()


def run_intuitive_testing(model_path, log_dir, visualize=False, episodes_per_scenario=3):
    """Run intuitive testing scenarios with clear success criteria"""
    print("\n" + "=" * 80)
    print("🧪 V2 SC-1 FIXED INTUITIVE MODEL TESTING")
    print("=" * 80)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    test_dir = Path(log_dir) / f"V2_Fixed_Test_{timestamp}"
    test_dir.mkdir(parents=True, exist_ok=True)

    # Load model
    try:
        print(f"📂 Loading V2 Fixed model: {model_path.name}")
        dummy_env = V2AllegroReachingEnvFixed(vis=False)
        model = PPO.load(str(model_path), env=dummy_env)
        dummy_env.close()
        print("✅ V2 Fixed Model loaded successfully!")
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        return

    # Intuitive test scenarios
    test_scenarios = {
        "close_easy": {
            "target_pos": [0.20, 0.15, 0.30],
            "description": "Close & Easy - Should succeed easily",
            "expected_success_rate": 0.7  # UPDATED: Was 0.8, adjusted for 0.12m threshold
        },
        "medium_standard": {
            "target_pos": [0.25, 0.15, 0.35],
            "description": "Standard Distance - Baseline performance",
            "expected_success_rate": 0.4  # UPDATED: Was 0.5, adjusted for new criteria
        },
        "far_challenging": {
            "target_pos": [0.35, 0.15, 0.40],
            "description": "Far & Challenging - Tests reach limits",
            "expected_success_rate": 0.15  # UPDATED: Was 0.2, more realistic
        },
        "side_reach": {
            "target_pos": [0.25, 0.25, 0.35],
            "description": "Side Reach - Tests lateral dexterity",
            "expected_success_rate": 0.25  # UPDATED: Was 0.3, adjusted
        },
        "precise_grasp": {
            "target_pos": [0.22, 0.12, 0.32],
            "description": "Precise Grasp - Tests fine control",
            "expected_success_rate": 0.3  # UPDATED: Was 0.4, adjusted
        }
    }

    all_results = []
    scenario_summaries = []

    for scenario_name, scenario_config in test_scenarios.items():
        print(f"\n🎯 Testing scenario: {scenario_name}")
        print(f"   {scenario_config['description']}")
        print(f"   Target: {scenario_config['target_pos']}")
        print(f"   Expected Success Rate: {scenario_config['expected_success_rate']:.1%}")

        # Create test environment
        env_test = V2AllegroReachingEnvFixed(vis=visualize)
        test_data_logger = DataLogger(test_dir / f"scenario_{scenario_name}")
        env_test.set_data_logger(test_data_logger)

        # Set specific scenario properties
        env_test.current_target_pos = np.array(scenario_config['target_pos'])

        scenario_results = []

        for episode in range(episodes_per_scenario):
            print(f"  Episode {episode+1}/{episodes_per_scenario}...", end=" ")

            obs = env_test.reset()
            episode_reward = 0
            episode_steps = 0
            max_fingers = 0
            min_distance = float('inf')
            final_hand_spread = 0
            final_hand_compactness = 0

            while episode_steps < 500:  # Reduced max steps for faster testing
                action, _ = model.predict(obs, deterministic=True)
                obs, reward, done, info = env_test.step(action)
                episode_reward += reward[0]
                episode_steps += 1

                # Track metrics
                fingers = info[0].get('num_active_fingers', 0)
                distance = info[0].get('distance', float('inf'))
                hand_spread = info[0].get('hand_spread', 0)
                hand_compactness = info[0].get('hand_compactness', 0)

                max_fingers = max(max_fingers, fingers)
                min_distance = min(min_distance, distance)
                final_hand_spread = hand_spread
                final_hand_compactness = hand_compactness

                if done[0]:
                    break

                if visualize:
                    time.sleep(0.02)

            test_data_logger.new_episode()

            # Extract final results
            final_distance = info[0].get('distance', float('inf'))
            success = info[0].get('success', False)

            result = {
                'scenario': scenario_name,
                'episode': episode + 1,
                'success': success,
                'final_distance': final_distance,
                'min_distance': min_distance,
                'max_fingers': max_fingers,
                'final_hand_spread': final_hand_spread,
                'final_hand_compactness': final_hand_compactness,
                'reward': episode_reward,
                'steps': episode_steps
            }

            scenario_results.append(result)
            all_results.append(result)

            status = "✅ SUCCESS" if success else "❌ FAILED"
            print(f"{status} | Dist: {final_distance:.3f}m | Fingers: {max_fingers} | Steps: {episode_steps}")

        # Save scenario data and create plots
        test_csv = test_data_logger.save_to_csv(f"{scenario_name}_test_data.csv")
        create_comprehensive_plots(test_csv)

        env_test.close()

        # Scenario summary
        successes = sum(r['success'] for r in scenario_results)
        success_rate = successes / len(scenario_results)
        avg_final_distance = np.mean([r['final_distance'] for r in scenario_results])
        avg_max_fingers = np.mean([r['max_fingers'] for r in scenario_results])

        expected_rate = scenario_config['expected_success_rate']
        performance_vs_expected = success_rate - expected_rate

        scenario_summary = {
            'scenario': scenario_name,
            'success_rate': success_rate,
            'expected_rate': expected_rate,
            'performance_vs_expected': performance_vs_expected,
            'avg_distance': avg_final_distance,
            'avg_fingers': avg_max_fingers
        }
        scenario_summaries.append(scenario_summary)

        print(f"  📊 {scenario_name} Results:")
        print(f"     Success Rate: {success_rate:.1%} (Expected: {expected_rate:.1%})")
        print(f"     Performance: {'+' if performance_vs_expected >= 0 else ''}{performance_vs_expected:.1%}")
        print(f"     Avg Final Distance: {avg_final_distance:.3f}m")

    # Overall analysis with intuitive summary
    print(f"\n{'='*80}")
    print("📈 V2 SC-1 FIXED INTUITIVE TEST RESULTS")
    print(f"{'='*80}")

    total_episodes = len(all_results)
    total_successes = sum(r['success'] for r in all_results)
    overall_success_rate = total_successes / total_episodes

    print(f"🎯 Overall Success Rate: {total_successes}/{total_episodes} ({overall_success_rate:.1%})")

    # Scenario-by-scenario breakdown
    print(f"\n📋 Scenario Performance Breakdown:")
    for summary in scenario_summaries:
        name = summary['scenario']
        rate = summary['success_rate']
        expected = summary['expected_rate']
        vs_expected = summary['performance_vs_expected']

        performance_icon = "🟢" if vs_expected >= 0.1 else "🟡" if vs_expected >= -0.1 else "🔴"
        print(f"   {performance_icon} {name:15s} - {rate:.1%} (vs {expected:.1%} expected)")

    # Overall assessment
    avg_performance_delta = np.mean([s['performance_vs_expected'] for s in scenario_summaries])

    if avg_performance_delta >= 0.1:
        assessment = "🟢 EXCELLENT - Exceeds expectations across scenarios"
    elif avg_performance_delta >= 0.0:
        assessment = "🟡 GOOD - Meets expectations with room for improvement"
    elif avg_performance_delta >= -0.2:
        assessment = "🟠 FAIR - Below expectations but functional"
    else:
        assessment = "🔴 POOR - Significantly below expectations"

    print(f"\n🏆 Overall Assessment: {assessment}")
    print(f"📁 All test results saved to: {test_dir}")
    print("=" * 80)

    return all_results, scenario_summaries


def main():
    """Main V2 SC-1 FIXED training function with dual curriculum learning"""

    # Initialize WandB with comprehensive configuration
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = f"V4_SC1_ConvexHull_{timestamp}"

    # Device detection for config
    device = "cuda" if torch.cuda.is_available() else "cpu"
    gpu_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "N/A"

    wandb_config = {
        # Training Configuration
        "algorithm": "PPO",
        "environment": "V2AllegroReachingEnvFixed_ConvexHull",
        "total_timesteps": 500000,
        "learning_rate": 3e-4,
        "batch_size": 64,
        "n_steps": 2048,
        "n_epochs": 10,
        "gamma": 0.99,
        "gae_lambda": 0.95,
        "clip_range": 0.2,
        "ent_coef": 0.01,
        "vf_coef": 0.5,
        "max_grad_norm": 0.5,

        # Hardware Configuration
        "device": device,
        "gpu_name": gpu_name,
        "pytorch_version": torch.__version__,

        # Environment Configuration
        "observation_space": 27,  # 27D with hand shape metrics
        "action_space": 10,
        "max_episode_steps": 1000,
        "simulation_frequency": 240,

        # V4 CONVEX HULL ENVELOPMENT Reward Function Configuration
        "reward_components": 4,
        "reward_structure": "convex_hull_envelopment",
        "reward_weights": {
            "hull_formation": "0-10 (curriculum-based)",
            "target_proximity": "0-15 (exponential decay from hull center)",
            "envelopment": "0-30 (spatial containment bonus)",
            "sustained_envelopment": "0-50 (consecutive containment)",
        },

        # Convex Hull Success Criteria
        "success_criteria": {
            "envelopment_required": True,        # Target must be inside convex hull
            "min_consecutive_steps": 50,         # Sustained envelopment required
            "min_clearance": 0.03,              # 3cm clearance to hull surface
            "safe_clearance": 0.05,             # 5cm optimal clearance
        },

        # Convex Hull Parameters
        "convex_hull_config": {
            "min_hull_volume": 0.0001,          # 0.1ml minimum viable hull
            "optimal_hull_volume": 0.001,       # 1ml optimal spread
            "max_hull_volume": 0.01,            # 10ml too spread out
            "hull_points": "4_fingers_1_palm",   # 5-point convex hull
        },

        "action_scaling": {
            "linear_velocity": 0.5,
            "angular_velocity": 1.0
        },

        # Convex Hull Curriculum Learning Configuration
        "curriculum_type": "convex_hull_formation",
        "hull_curriculum_phases": {
            "stage_1_any_valid_hull": "0-150K timesteps - Learn basic finger spreading",
            "stage_2_optimal_hull_size": "150K-350K timesteps - Optimize hull volume",
            "stage_3_precise_envelopment": "350K-500K timesteps - Master spatial containment"
        },
        "target_curriculum_phases": {
            "phase_1_static_targets_only": "0-500K timesteps - Master envelopment with static targets",
        },

        # Improved Control Configuration
        "butterworth_filter": {
            "cutoff_frequency": 8.0,
            "sampling_frequency": 240.0,
            "order": 2
        },

        # Fixed Tendon Control Configuration
        "tendon_control": {
            "force_gain": 10.0,
            "damping": 1.2,
            "max_force": 50.0,
            "fingers": ["index", "middle", "ring", "thumb"]
        },

        # V4 Features
        "paradigm_shift_features": [
            "spatial_containment_focus",
            "convex_hull_geometry",
            "delaunay_triangulation_inside_testing",
            "clearance_distance_calculation",
            "sustained_envelopment_success",
            "hull_formation_curriculum"
        ],

        # Experiment Metadata
        "experiment_version": "V4.0_CONVEX_HULL_ENVELOPMENT",
        "base_script": "V2_SC-1_Fixed_V3.py",
        "paradigm_description": "Complete paradigm shift from distance-based grasping to spatial envelopment using convex hull geometry",
        "task_definition": "Position 4 fingertips + palm to create convex hull that spatially contains target object for sustained period",
        "timestamp": timestamp
    }

    # FIXED: Initialize WandB properly with global scope handling
    wandb_available = True
    try:
        # FIXED: Initialize with sync_tensorboard=False to avoid conflicts
        wandb.init(
            project="space-touch-v4-convex-hull-envelopment",  # V4 project name
            name=run_name,
            config=wandb_config,
            tags=[
                "v4-convex-hull",
                "spatial-envelopment",
                "500k-training",
                "4-component-reward",
                "hull-formation",
                "delaunay-triangulation",
                "sustained-containment",
                "geometric-approach",
                "soft-capture-redefined"
            ],
            notes=f"V4.0 SC-1 CONVEX HULL ENVELOPMENT - Complete paradigm shift from distance-based grasping to spatial containment using convex hull geometry. Task: Position fingertips + palm to create hull that spatially contains target for 50+ consecutive steps. Expected hull formation learning at 100-250K, envelopment mastery at 350-500K steps.",
            save_code=True,
            sync_tensorboard=False  # FIXED: Disable tensorboard sync to avoid conflicts
        )
        print("✅ WandB initialized with comprehensive FIXED configuration")
        print(f"🌐 WandB Run: {wandb.run.url}")
    except Exception as e:
        print(f"⚠️  WandB initialization failed: {e}")
        print("   Continuing training without WandB logging...")
        wandb_available = False

    # Create organized training directory
    base_training_dir = Path("./SC1_Training_Runs/")
    base_training_dir.mkdir(parents=True, exist_ok=True)

    log_dir = base_training_dir / f"Run_{timestamp}_V4_SC1_ConvexHull_Envelopment"
    log_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print(f"🚀 V4 SC-1 CONVEX HULL ENVELOPMENT TRAINING")
    print("=" * 80)
    print(f"Run Name: {run_name}")
    print(f"Log Directory: {log_dir}")
    print(f"Total Timesteps: 500,000")
    print("V4 PARADIGM SHIFT - SPATIAL CONTAINMENT APPROACH:")
    print("  ✓ CONVEX HULL geometry-based reward function")
    print("  ✓ SPATIAL ENVELOPMENT task (surround, don't touch)")
    print("  ✓ 4-component reward: Hull + Proximity + Envelopment + Sustained")
    print("  ✓ Target must be INSIDE convex hull formed by fingers + palm")
    print("  ✓ SUCCESS = 50+ consecutive steps of spatial containment")
    print("  ✓ Delaunay triangulation for precise inside/outside testing")
    print("  ✓ Hull formation curriculum (any valid → optimal size → precise control)")
    print("=" * 80)

    try:
        # Create data logger
        data_logger = DataLogger(log_dir)

        # Create V2 FIXED environment
        print("\n🏗️  Creating V2 FIXED tendon-controlled environment...")
        env = V2AllegroReachingEnvFixed(vis=False)
        env.set_data_logger(data_logger)

        # Create callbacks
        checkpoint_callback = CheckpointCallback(
            save_freq=60000,
            save_path=log_dir / "checkpoints",
            verbose=1
        )

        wandb_callback = WandBCallback(data_logger, log_freq=100) if wandb_available else None
        callbacks = [checkpoint_callback]
        if wandb_callback:
            callbacks.append(wandb_callback)

        # Configure logger
        tb_log_dir = log_dir / "tensorboard"
        tb_log_dir.mkdir(parents=True, exist_ok=True)
        custom_logger = configure(str(tb_log_dir), ["stdout", "tensorboard"])

        # Create PPO model
        print("🧠 Creating V2 FIXED PPO model...")

        device = "cuda" if torch.cuda.is_available() else "cpu"
        if torch.cuda.is_available():
            print(f"✓ GPU acceleration: {torch.cuda.get_device_name(0)}")
        else:
            print("⚠️  Using CPU (no GPU found)")

        model = PPO(
            "MlpPolicy",
            env,
            verbose=1,
            tensorboard_log=str(tb_log_dir),
            n_steps=2048,
            learning_rate=3e-4,
            n_epochs=10,
            batch_size=64,
            gamma=0.99,
            gae_lambda=0.95,
            clip_range=0.2,
            ent_coef=0.01,
            vf_coef=0.5,
            max_grad_norm=0.5,
            device=device
        )

        model.set_logger(custom_logger)

        print("\n" + "=" * 80)
        print("V4 CONVEX HULL ENVELOPMENT TRAINING CONFIGURATION")
        print("=" * 80)
        print("V4 Convex Hull 4-Component Reward Function:")
        print("  Component 1: Hull Formation (0-10) - Valid convex hull with fingers + palm")
        print("  Component 2: Target Proximity (0-15) - Position hull center near target")
        print("  Component 3: Envelopment (0-30) - Target INSIDE convex hull volume")
        print("  Component 4: Sustained Envelopment (0-50) - Maintain containment 50+ steps")
        print("\n🎯 V4 CONVEX HULL KEY FEATURES:")
        print("  ✓ SPATIAL CONTAINMENT: Target must be inside 3D convex hull")
        print("  ✓ GEOMETRY-BASED: Uses scipy ConvexHull + Delaunay triangulation")
        print("  ✓ NO CONTACT REQUIRED: Pure spatial positioning task")
        print("  ✓ HULL FORMATION CURRICULUM: Any valid → optimal size → precise control")
        print("  ✓ CLEARANCE TRACKING: 3-5cm safety margin to hull surfaces")
        print("  ✓ SUSTAINED SUCCESS: 50 consecutive steps of spatial containment")
        print("  ✓ REWARD RANGE: [0.0, 105.0] - No negative penalties")
        print("  ✓ 5-POINT GEOMETRY: 4 fingertips + 1 palm center = convex hull")
        print("\nExpected Learning Timeline (CONVEX HULL ENVELOPMENT):")
        print("  0K-150K:    Hull formation basics - learn finger spreading")
        print("  150K-350K:  Hull size optimization - achieve 0.1-1ml volumes")
        print("  350K-500K:  Envelopment mastery - 10-25% sustained containment rate")
        print("\nTask Definition: 'Surround and engulf target using hand geometry, NOT contact!'")
        print("Success Metric: Target spatially contained within convex hull for 50+ steps")
        print("Observation Space: 27D (with hand shape metrics)")
        print("Action Space: 10D (6 DOF base + 4 tendons)")
        print("Control Filtering: Butterworth 8Hz low-pass")
        print("Checkpoints: Every 60K timesteps")
        print("=" * 80)

        # Train the model
        print("\n🏃‍♂️ Starting V4 SC-1 Convex Hull Envelopment training...")

        model.learn(
            total_timesteps=500000,
            callback=callbacks,
            log_interval=10,
            tb_log_name=f"V4_SC1_ConvexHull_{timestamp}",
            reset_num_timesteps=True,
            progress_bar=True
        )

        # Save final model
        final_model_path = log_dir / f"v4_sc1_convex_hull_final_model_{timestamp}"
        model.save(str(final_model_path))
        print(f"\n✅ V4 CONVEX HULL ENVELOPMENT Training Complete! Final model saved: {final_model_path}.zip")

        # Run INTUITIVE testing
        print(f"\n🧪 Running INTUITIVE testing with clear scenarios...")
        test_results, scenario_summaries = run_intuitive_testing(
            final_model_path.with_suffix('.zip'),
            log_dir,
            visualize=False,
            episodes_per_scenario=5
        )

        # Enhanced final results logging to WandB
        if wandb_available:
            # Calculate final metrics from intuitive testing
            overall_success_rate = np.mean([r['success'] for r in test_results])
            avg_distance = np.mean([r['final_distance'] for r in test_results])
            avg_reward = np.mean([r['reward'] for r in test_results])

            # Scenario-specific metrics
            scenario_metrics = {}
            for summary in scenario_summaries:
                scenario_name = summary['scenario']
                scenario_metrics.update({
                    f"test_scenarios/{scenario_name}_success_rate": summary['success_rate'],
                    f"test_scenarios/{scenario_name}_vs_expected": summary['performance_vs_expected'],
                    f"test_scenarios/{scenario_name}_avg_distance": summary['avg_distance']
                })

            final_metrics = {
                "final/overall_success_rate": overall_success_rate,
                "final/avg_distance": avg_distance,
                "final/avg_reward": avg_reward,
                "final/total_test_episodes": len(test_results),
                "final/scenarios_tested": len(scenario_summaries),
                **scenario_metrics
            }

            # Performance assessment
            avg_performance_vs_expected = np.mean([s['performance_vs_expected'] for s in scenario_summaries])
            if avg_performance_vs_expected >= 0.1:
                performance_grade = "Excellent"
            elif avg_performance_vs_expected >= 0.0:
                performance_grade = "Good"
            elif avg_performance_vs_expected >= -0.1:
                performance_grade = "Fair"
            else:
                performance_grade = "Needs Improvement"

            final_metrics["final/performance_grade"] = performance_grade
            final_metrics["final/avg_performance_vs_expected"] = avg_performance_vs_expected

            # Log all final metrics
            wandb.log(final_metrics)

            # Create WandB summary
            wandb.run.summary.update({
                "best_success_rate": overall_success_rate,
                "best_avg_distance": avg_distance,
                "performance_assessment": performance_grade,
                "curriculum_phases_completed": 5,
                "architecture_version": "V2.1_FIXED",
                "key_improvements": "Fixed curriculum learning, hand shape rewards, improved stability"
            })

            print("✅ Enhanced results logged to WandB")

        print("\n" + "=" * 80)
        print("🎉 V4 SC-1 CONVEX HULL ENVELOPMENT TRAINING COMPLETE!")
        print("=" * 80)
        print(f"📁 All results: {log_dir}")
        print(f"🤖 Final model: {final_model_path}.zip")
        print(f"📊 Test results: {log_dir}/V4_ConvexHull_Test_*/")
        print(f"📈 TensorBoard: tensorboard --logdir {tb_log_dir}")
        if wandb_available:
            print(f"🌐 WandB: {wandb.run.url}")
        print("\n🎯 V4 PARADIGM SHIFT ACCOMPLISHED:")
        print("   • COMPLETE TASK REDEFINITION: Distance-based → Spatial envelopment")
        print("   • CONVEX HULL GEOMETRY: 4 fingertips + palm = 5-point containment volume")
        print("   • DELAUNAY TRIANGULATION: Precise inside/outside spatial testing")
        print("   • SUSTAINED ENVELOPMENT: 50+ consecutive steps of target containment")
        print("   • HULL FORMATION CURRICULUM: Progressive complexity (any → optimal → precise)")
        print("   • NO CONTACT PARADIGM: Surround and engulf, don't touch!")
        print("   • CLEARANCE MONITORING: 3-5cm safety margins to hull surfaces")
        print("=" * 80)

    except KeyboardInterrupt:
        print("\n⏹️  Training interrupted by user")

    except Exception as e:
        print(f"\n❌ Error occurred: {e}")
        import traceback
        traceback.print_exc()

    finally:
        try:
            if 'env' in locals():
                env.close()
            if wandb_available and wandb.run is not None:
                wandb.finish()
        except:
            pass


if __name__ == "__main__":
    main()