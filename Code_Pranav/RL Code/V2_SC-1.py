#!/usr/bin/env python3
"""
V2_SC-1.py - Advanced Tendon-Based Allegro Hand Training with Dual Curriculum Learning
Combines SC-1 and WandB Enhanced Checkpointing with simplified reward and dual curriculum approach.

Features:
- Simplified 5-component reward function (no smoothness penalty)
- Butterworth low-pass filtering for control smoothing
- Dual curriculum learning: Reward complexity → Target dynamics
- Convex hull integration in reward function
- Comprehensive testing with full plotting
- 200K timesteps training with checkpointing
- WandB integration for experiment tracking
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

    def __init__(self, cutoff_freq=10.0, sampling_freq=240.0, order=2):
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

            # Finger positions and convex hull
            'finger_pos_1_x': [], 'finger_pos_1_y': [], 'finger_pos_1_z': [],
            'finger_pos_2_x': [], 'finger_pos_2_y': [], 'finger_pos_2_z': [],
            'finger_pos_3_x': [], 'finger_pos_3_y': [], 'finger_pos_3_z': [],
            'finger_pos_4_x': [], 'finger_pos_4_y': [], 'finger_pos_4_z': [],
            'convex_hull_volume': [], 'convex_hull_area': [],

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
            'convex_hull_reward': [],

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

    def save_to_csv(self, filename="v2_sc1_training_data.csv"):
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
    """Enhanced tendon controller with Butterworth filtering"""

    def __init__(self, hand_id, joint_names, joint_indices):
        self.hand_id = hand_id
        self.joint_names = joint_names
        self.joint_indices = joint_indices

        # Tendon control parameters
        self.TENDON_FORCE_GAIN = 15.0
        self.TENDON_DAMPING = 0.8
        self.MAX_TENDON_FORCE = 60.0

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
    """Enhanced WandB logging callback with comprehensive curriculum tracking"""

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
        self.recent_convex_hull_volumes = []
        self.reward_components_buffer = {
            'distance': [], 'tendon_efficiency': [], 'tactile_contact': [],
            'movement_penalty': [], 'convex_hull': [], 'success_bonus': []
        }

        # Curriculum phase tracking
        self.current_reward_phase = 1
        self.current_target_phase = 1
        self.phase_transition_timesteps = []

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

                        # Enhanced episode logging to WandB
                        wandb.log({
                            'episode/reward': ep_reward,
                            'episode/length': ep_length,
                            'episode/success': info.get('success', False),
                            'training/timesteps': self.num_timesteps,
                            'training/episodes': len(self.episode_rewards),
                            'training/fps': self.log_freq / (time.time() - getattr(self, '_last_log_time', time.time()))
                        })
                        self._last_log_time = time.time()

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
                            if key.replace('_', '') in str(components).lower():
                                component_key = None
                                if key == 'distance':
                                    component_key = 'distance_reward'
                                elif key == 'tendon_efficiency':
                                    component_key = 'tendon_efficiency_reward'
                                elif key == 'tactile_contact':
                                    component_key = 'tactile_contact_reward'
                                elif key == 'movement_penalty':
                                    component_key = 'movement_penalty'
                                elif key == 'convex_hull':
                                    component_key = 'convex_hull_reward'
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

                        # Convex hull tracking
                        if 'convex_hull_volume' in info:
                            self.recent_convex_hull_volumes.append(info['convex_hull_volume'])

            # Enhanced aggregated metrics logging
            if self.num_timesteps % self.log_freq == 0:
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

                # Convex hull metrics
                if self.recent_convex_hull_volumes:
                    log_dict.update({
                        'convex_hull/avg_volume': np.mean(self.recent_convex_hull_volumes),
                        'convex_hull/max_volume': np.max(self.recent_convex_hull_volumes),
                        'convex_hull/volume_std': np.std(self.recent_convex_hull_volumes)
                    })
                    self.recent_convex_hull_volumes = []

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
                        'training/learning_progress': self.num_timesteps / 200000.0  # Progress to 200K
                    })

                # Curriculum effectiveness
                log_dict.update({
                    'curriculum/current_reward_phase': self.current_reward_phase,
                    'curriculum/current_target_phase': self.current_target_phase,
                    'curriculum/total_transitions': len(self.phase_transition_timesteps)
                })

                # Log all metrics
                if log_dict:
                    wandb.log(log_dict, step=self.num_timesteps)

        except Exception as e:
            if self.verbose > 0:
                print(f"WandB logging error: {e}")

        return True


class V2AllegroReachingEnv(VecEnv):
    """
    V2 Enhanced environment with dual curriculum learning and simplified reward function
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

        # Curriculum learning parameters
        self.training_timesteps = 0
        self.reward_curriculum_phase = 1  # Phase 1-4: Simple to complex reward components
        self.target_curriculum_phase = 1  # Phase 1-3: Static to dynamic targets

        # Reward curriculum thresholds (based on timesteps)
        self.REWARD_PHASE_THRESHOLDS = [0, 40000, 80000, 120000, 200000]  # 5 phases
        self.TARGET_PHASE_THRESHOLDS = [160000, 180000, 200000]  # Start after reward curriculum

        # Initialize PyBullet
        self._init_pybullet()

        self.hand = None
        self.target_sphere = None
        self.tendon_controller = None
        self.hand_spawned = False

        # Action space: 6 DOF base movement + 4 tendon forces
        action_dim = 10
        self.action_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(action_dim,), dtype=np.float32)

        # Observation: base_pos(3) + target_pos(3) + base_vel(3) + finger_positions(12) + binary_tactile(4) + convex_hull(1)
        obs_dim = 26  # Enhanced with convex hull
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
        """Update curriculum learning phases based on timesteps"""
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

    def _compute_convex_hull_features(self, finger_positions):
        """Compute convex hull volume and area from finger positions"""
        try:
            # Reshape finger positions to 4x3 array
            points = finger_positions.reshape(4, 3)

            #Add one point from the palm center
            #Train just for convex hull 

            # Compute convex hull
            hull = ConvexHull(points)

            return hull.volume, hull.area
        except:
            # Return default values if hull computation fails
            return 0.0, 0.0

    def _get_observation(self):
        """Get current observation with convex hull features"""
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

            # Compute convex hull features
            hull_volume, hull_area = self._compute_convex_hull_features(finger_positions)

            # Combine into observation (26D)
            obs = np.concatenate([
                base_pos,           # 3D
                target_pos,         # 3D
                base_vel,           # 3D
                finger_positions,   # 12D
                binary_tactile,     # 4D
                [hull_volume]       # 1D - convex hull feature
            ])

            return np.expand_dims(obs.astype(np.float32), axis=0)

        except Exception as e:
            print(f"Error getting observation: {e}")
            return np.zeros((1, self.observation_space.shape[0]), dtype=np.float32)

    def _compute_simplified_reward(self, base_pos, target_pos, tendon_forces, binary_tactile,
                                  base_vel, hull_volume, hull_area):
        """
        Compute simplified 5-component reward with curriculum progression
        """
        distance = np.linalg.norm(base_pos - target_pos)
        num_active_fingers = np.sum(binary_tactile)

        # ============ REWARD COMPONENT 1: Distance Reward ============
        distance_reward = 0.0
        if self.reward_curriculum_phase >= 1:
            distance_reward = np.exp(-8.0 * distance)

        # ============ REWARD COMPONENT 2: Tendon Efficiency Reward ============
        tendon_efficiency_reward = 0.0
        if self.reward_curriculum_phase >= 2:
            tendon_efficiency = 1.0 - 0.5 * np.mean(tendon_forces)
            tendon_efficiency_reward = 0.2 * tendon_efficiency

        # ============ REWARD COMPONENT 3: Tactile Contact Reward ============
        tactile_contact_reward = 0.0
        if self.reward_curriculum_phase >= 3:
            tactile_contact_reward = 0.15 * num_active_fingers

        # ============ REWARD COMPONENT 4: Movement Penalty ============
        movement_penalty = 0.0
        if self.reward_curriculum_phase >= 4:
            linear_vel_magnitude = np.linalg.norm(base_vel[:3])
            angular_vel_magnitude = np.linalg.norm(base_vel[3:]) if len(base_vel) > 3 else 0
            movement_penalty = -0.01 * (linear_vel_magnitude + angular_vel_magnitude)

        # ============ REWARD COMPONENT 5: Convex Hull Reward ============
        convex_hull_reward = 0.0
        if self.reward_curriculum_phase >= 5:
            # Reward for appropriate hand configuration
            if hull_volume > 1e-6:  # Valid hull
                # Encourage moderate hull volume (not too collapsed, not too spread)
                target_volume = 0.001  # Target hull volume
                hull_reward = np.exp(-abs(hull_volume - target_volume) * 1000)
                convex_hull_reward = 0.1 * hull_reward

        # ============ SUCCESS BONUS ============
        success_bonus = 10.0 if distance < 0.1 and num_active_fingers >= 2 else 0.0

        # Combine all components
        total_reward = (distance_reward + tendon_efficiency_reward + tactile_contact_reward +
                       movement_penalty + convex_hull_reward + success_bonus)

        return {
            'total_reward': total_reward,
            'distance_reward': distance_reward,
            'tendon_efficiency_reward': tendon_efficiency_reward,
            'tactile_contact_reward': tactile_contact_reward,
            'movement_penalty': movement_penalty,
            'convex_hull_reward': convex_hull_reward,
            'success_bonus': success_bonus
        }

    def step_wait(self):
        """Execute one step with curriculum learning and simplified reward"""
        self.step_counts += 1
        self.episode_lengths += 1
        actions = self.actions[0]

        try:
            if self.hand is None:
                obs = self._get_observation()
                return obs, np.array([-1.0], dtype=np.float32), np.array([True]), [{"error": "Hand not loaded"}]

            # Split actions
            base_actions = actions[:6]
            tendon_actions = actions[6:10]

            # Apply Butterworth filtering to base actions
            filtered_base_actions = self.tendon_controller.apply_control_filtering(base_actions)

            # Apply base movement
            linear_vel = filtered_base_actions[:3] * 0.3
            angular_vel = filtered_base_actions[3:6] * 0.8

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

            # Get current states
            base_pos, base_orn = p.getBasePositionAndOrientation(self.hand)
            base_vel, base_ang_vel = p.getBaseVelocity(self.hand)
            target_pos, _ = p.getBasePositionAndOrientation(self.target_sphere)

            base_pos = np.array(base_pos)
            base_vel_combined = np.array(list(base_vel) + list(base_ang_vel))
            target_pos = np.array(target_pos)

            # Get additional states
            finger_positions = self._get_finger_positions()
            binary_tactile = self._get_binary_tactile_feedback()
            hull_volume, hull_area = self._compute_convex_hull_features(finger_positions)

            # Compute simplified reward with curriculum
            reward_components = self._compute_simplified_reward(
                base_pos, target_pos, tendon_forces, binary_tactile,
                base_vel_combined, hull_volume, hull_area
            )

            total_reward = reward_components['total_reward']
            distance = np.linalg.norm(base_pos - target_pos)
            success = distance < 0.1 and np.sum(binary_tactile) >= 2

            self.episode_rewards[0] += total_reward

            # Termination conditions
            dones = np.array([
                self.step_counts[0] >= self.max_steps or
                success or
                distance > 3.0 or
                base_pos[2] < 0.01
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
                    "convex_hull_volume": hull_volume, "convex_hull_area": hull_area,
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
                    "convex_hull_reward": reward_components['convex_hull_reward'],
                    "success_bonus": reward_components['success_bonus'],
                    "reward_curriculum_phase": self.reward_curriculum_phase,
                    "target_curriculum_phase": self.target_curriculum_phase,
                    "training_timesteps": self.training_timesteps,
                    "success": success,
                }

                self.data_logger.log_step(self.step_data)

            # Info for callbacks
            infos = [{
                "success": success,
                "distance": distance,
                "reward_curriculum_phase": self.reward_curriculum_phase,
                "target_curriculum_phase": self.target_curriculum_phase,
                "reward_components": reward_components
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
    """Enhanced checkpoint callback with curriculum tracking"""

    def __init__(self, save_freq=40000, save_path="./checkpoints/", verbose=1):
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

            # Update environment curriculum
            if hasattr(self.training_env, 'envs') and len(self.training_env.envs) > 0:
                self.training_env.envs[0].update_curriculum(self.num_timesteps)

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

        # 1. V2 Performance Overview
        fig, axes = plt.subplots(3, 2, figsize=(18, 15))
        fig.suptitle('V2 SC-1 Comprehensive Performance Analysis', fontsize=16, fontweight='bold')

        # Reward components over time
        if all(col in df_plot.columns for col in ['distance_reward', 'tendon_efficiency_reward', 'tactile_contact_reward', 'movement_penalty', 'convex_hull_reward']):
            axes[0,0].plot(step_vals, df_plot['reward'], label='Total Reward', linewidth=1.5, color='black')
            axes[0,0].plot(step_vals, df_plot['distance_reward'], label='Distance', alpha=0.8)
            axes[0,0].plot(step_vals, df_plot['tendon_efficiency_reward'], label='Tendon Efficiency', alpha=0.8)
            axes[0,0].plot(step_vals, df_plot['tactile_contact_reward'], label='Tactile Contact', alpha=0.8)
            axes[0,0].plot(step_vals, df_plot['movement_penalty'], label='Movement Penalty', alpha=0.8)
            axes[0,0].plot(step_vals, df_plot['convex_hull_reward'], label='Convex Hull', alpha=0.8)
            axes[0,0].set_title('Simplified Reward Components (V2)')
            axes[0,0].set_xlabel('Step')
            axes[0,0].set_ylabel('Reward')
            axes[0,0].legend()
            axes[0,0].grid(True, alpha=0.3)

        # Curriculum progression
        if 'reward_curriculum_phase' in df_plot.columns:
            axes[0,1].plot(step_vals, df_plot['reward_curriculum_phase'], 'o-', markersize=2, label='Reward Phase')
            if 'target_curriculum_phase' in df_plot.columns:
                axes[0,1].plot(step_vals, df_plot['target_curriculum_phase'], 's-', markersize=2, label='Target Phase')
            axes[0,1].set_title('Dual Curriculum Learning Progress')
            axes[0,1].set_xlabel('Step')
            axes[0,1].set_ylabel('Curriculum Phase')
            axes[0,1].legend()
            axes[0,1].grid(True, alpha=0.3)

        # Distance and success tracking
        axes[1,0].plot(step_vals, df_plot['distance_to_target'], alpha=0.7, linewidth=0.8, color='blue', label='Distance')
        axes[1,0].axhline(y=0.1, color='red', linestyle='--', label='Success Threshold (0.1m)')
        axes[1,0].set_title('Distance to Target Over Time')
        axes[1,0].set_ylabel('Distance (m)')
        axes[1,0].set_xlabel('Step')
        axes[1,0].legend()
        axes[1,0].grid(True, alpha=0.3)

        # Convex hull analysis
        if 'convex_hull_volume' in df_plot.columns:
            axes[1,1].plot(step_vals, df_plot['convex_hull_volume'], alpha=0.7, linewidth=0.8, color='green')
            axes[1,1].set_title('Convex Hull Volume Evolution')
            axes[1,1].set_ylabel('Hull Volume')
            axes[1,1].set_xlabel('Step')
            axes[1,1].grid(True, alpha=0.3)

        # Control smoothing analysis (filtered vs raw)
        if all(col in df_plot.columns for col in ['control_linear_x', 'filtered_linear_x']):
            sample_steps = step_vals[-1000:] if len(step_vals) > 1000 else step_vals
            sample_raw = df_plot['control_linear_x'].values[-1000:] if len(step_vals) > 1000 else df_plot['control_linear_x'].values
            sample_filtered = df_plot['filtered_linear_x'].values[-1000:] if len(step_vals) > 1000 else df_plot['filtered_linear_x'].values

            axes[2,0].plot(sample_steps, sample_raw, alpha=0.7, label='Raw Control', linewidth=0.8)
            axes[2,0].plot(sample_steps, sample_filtered, alpha=0.7, label='Butterworth Filtered', linewidth=0.8)
            axes[2,0].set_title('Control Smoothing (Butterworth Filter)')
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
        plt.savefig(plot_dir / 'v2_comprehensive_analysis.png', dpi=150, bbox_inches='tight')
        plt.close()

        # 2. Tendon and Hand Configuration Analysis
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('V2 Tendon Control and Hand Configuration Analysis', fontsize=16)

        # Individual tendon forces
        if all(col in df_plot.columns for col in ['tendon_force_index', 'tendon_force_middle', 'tendon_force_ring', 'tendon_force_thumb']):
            axes[0,0].plot(step_vals, df_plot['tendon_force_index'], label='Index', alpha=0.8)
            axes[0,0].plot(step_vals, df_plot['tendon_force_middle'], label='Middle', alpha=0.8)
            axes[0,0].plot(step_vals, df_plot['tendon_force_ring'], label='Ring', alpha=0.8)
            axes[0,0].plot(step_vals, df_plot['tendon_force_thumb'], label='Thumb', alpha=0.8)
            axes[0,0].set_title('Individual Tendon Forces')
            axes[0,0].set_xlabel('Step')
            axes[0,0].set_ylabel('Normalized Force')
            axes[0,0].legend()
            axes[0,0].grid(True, alpha=0.3)

            # Tendon coordination (correlation matrix)
            tendon_data = np.array([
                df_plot['tendon_force_index'].values,
                df_plot['tendon_force_middle'].values,
                df_plot['tendon_force_ring'].values,
                df_plot['tendon_force_thumb'].values
            ])
            correlation_matrix = np.corrcoef(tendon_data)

            im = axes[0,1].imshow(correlation_matrix, cmap='coolwarm', vmin=-1, vmax=1)
            axes[0,1].set_title('Tendon Force Coordination')
            axes[0,1].set_xticks(range(4))
            axes[0,1].set_yticks(range(4))
            axes[0,1].set_xticklabels(['Index', 'Middle', 'Ring', 'Thumb'])
            axes[0,1].set_yticklabels(['Index', 'Middle', 'Ring', 'Thumb'])
            plt.colorbar(im, ax=axes[0,1])

        # Hand configuration in 3D space (finger positions)
        if all(col in df_plot.columns for col in ['finger_pos_1_x', 'finger_pos_2_x', 'finger_pos_3_x', 'finger_pos_4_x']):
            # Plot finger trajectories
            axes[0,2].plot(df_plot['finger_pos_1_x'], df_plot['finger_pos_1_y'], 'o', markersize=1, alpha=0.6, label='Finger 1')
            axes[0,2].plot(df_plot['finger_pos_2_x'], df_plot['finger_pos_2_y'], 'o', markersize=1, alpha=0.6, label='Finger 2')
            axes[0,2].plot(df_plot['finger_pos_3_x'], df_plot['finger_pos_3_y'], 'o', markersize=1, alpha=0.6, label='Finger 3')
            axes[0,2].plot(df_plot['finger_pos_4_x'], df_plot['finger_pos_4_y'], 'o', markersize=1, alpha=0.6, label='Finger 4')
            if 'target_x' in df_plot.columns:
                axes[0,2].scatter(df_plot['target_x'].iloc[0], df_plot['target_y'].iloc[0],
                                color='red', s=100, marker='*', label='Target')
            axes[0,2].set_title('Finger Trajectories (XY Plane)')
            axes[0,2].set_xlabel('X Position (m)')
            axes[0,2].set_ylabel('Y Position (m)')
            axes[0,2].legend()
            axes[0,2].grid(True, alpha=0.3)
            axes[0,2].axis('equal')

        # Convex hull volume vs distance relationship
        if all(col in df_plot.columns for col in ['convex_hull_volume', 'distance_to_target']):
            axes[1,0].scatter(df_plot['distance_to_target'], df_plot['convex_hull_volume'],
                            alpha=0.5, s=1, c=step_vals, cmap='viridis')
            axes[1,0].set_title('Convex Hull Volume vs Distance')
            axes[1,0].set_xlabel('Distance to Target (m)')
            axes[1,0].set_ylabel('Convex Hull Volume')
            axes[1,0].grid(True, alpha=0.3)

        # Success analysis by episode
        if 'episode' in df.columns and 'success' in df.columns:
            try:
                episode_success = df.groupby('episode')['success'].max()
                if len(episode_success) > 1:
                    axes[1,1].plot(episode_success.index, episode_success.values, 'o-', alpha=0.7, markersize=3)

                    # Rolling success rate
                    if len(episode_success) > 20:
                        window = min(20, len(episode_success) // 5)
                        rolling_success = episode_success.rolling(window=window, center=True).mean()
                        axes[1,1].plot(episode_success.index, rolling_success.values, 'red',
                                     linewidth=2, label=f'Rolling Success Rate ({window})')
                        axes[1,1].legend()

                    axes[1,1].set_title('Success Rate by Episode')
                    axes[1,1].set_xlabel('Episode')
                    axes[1,1].set_ylabel('Success (0/1)')
                    axes[1,1].grid(True, alpha=0.3)
            except Exception as e:
                print(f"Skipping episode success plot: {e}")

        # Curriculum phase effectiveness
        if all(col in df_plot.columns for col in ['reward_curriculum_phase', 'reward']):
            phase_rewards = df.groupby('reward_curriculum_phase')['reward'].agg(['mean', 'std']).reset_index()
            axes[1,2].bar(phase_rewards['reward_curriculum_phase'], phase_rewards['mean'],
                         yerr=phase_rewards['std'], alpha=0.7, capsize=5)
            axes[1,2].set_title('Reward Performance by Curriculum Phase')
            axes[1,2].set_xlabel('Reward Curriculum Phase')
            axes[1,2].set_ylabel('Average Reward')
            axes[1,2].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(plot_dir / 'v2_tendon_configuration_analysis.png', dpi=150, bbox_inches='tight')
        plt.close()

        # 3. Test Scenarios Analysis (if test data)
        if test_scenarios:
            fig, axes = plt.subplots(2, 2, figsize=(15, 10))
            fig.suptitle('V2 Test Scenarios Performance Analysis', fontsize=16)

            # This would be populated during testing phase
            # Placeholder for test scenario analysis
            axes[0,0].text(0.5, 0.5, 'Test Scenario Results\n(Generated during testing)',
                         ha='center', va='center', transform=axes[0,0].transAxes, fontsize=12)
            axes[0,1].text(0.5, 0.5, 'Success Rate by Scenario\n(Generated during testing)',
                         ha='center', va='center', transform=axes[0,1].transAxes, fontsize=12)
            axes[1,0].text(0.5, 0.5, 'Distance Performance\n(Generated during testing)',
                         ha='center', va='center', transform=axes[1,0].transAxes, fontsize=12)
            axes[1,1].text(0.5, 0.5, 'Curriculum Effectiveness\n(Generated during testing)',
                         ha='center', va='center', transform=axes[1,1].transAxes, fontsize=12)

            plt.tight_layout()
            plt.savefig(plot_dir / 'v2_test_scenarios_analysis.png', dpi=150, bbox_inches='tight')
            plt.close()

        print(f"✅ V2 comprehensive plots saved to: {plot_dir}")

        # Summary statistics
        print("\n" + "=" * 80)
        print("V2 SC-1 TRAINING SUMMARY")
        print("=" * 80)
        print(f"Total steps: {len(df)}")

        if len(df) > 0:
            print(f"Final distance to target: {df['distance_to_target'].iloc[-1]:.4f} m")
            print(f"Average total reward: {df['reward'].mean():.4f}")

            if 'convex_hull_volume' in df.columns:
                print(f"Final convex hull volume: {df['convex_hull_volume'].iloc[-1]:.6f}")

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


def run_comprehensive_testing(model_path, log_dir, visualize=True, episodes_per_scenario=5):
    """Run comprehensive testing with all scenarios and full plotting"""
    print("\n" + "=" * 80)
    print("🧪 V2 SC-1 COMPREHENSIVE MODEL TESTING")
    print("=" * 80)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    test_dir = Path(log_dir) / f"V2_Test_{timestamp}"
    test_dir.mkdir(parents=True, exist_ok=True)

    # Load model
    try:
        print(f"📂 Loading V2 model: {model_path.name}")
        dummy_env = V2AllegroReachingEnv(vis=False)
        model = PPO.load(str(model_path), env=dummy_env)
        dummy_env.close()
        print("✅ V2 Model loaded successfully!")
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        return

    # Test scenarios for V2
    test_scenarios = {
        "static_close": {"distance": 0.15, "description": "Static Close Target"},
        "static_medium": {"distance": 0.25, "description": "Static Medium Target"},
        "static_far": {"distance": 0.35, "description": "Static Far Target"},
        "curriculum_test": {"distance": 0.2, "description": "Curriculum Learning Test"},
        "convex_hull_test": {"distance": 0.2, "description": "Convex Hull Reward Test"}
    }

    all_results = []

    for scenario_name, scenario_config in test_scenarios.items():
        print(f"\n🎯 Testing V2 scenario: {scenario_name}")
        print(f"   {scenario_config['description']}")

        # Create test environment
        env_test = V2AllegroReachingEnv(vis=visualize)
        test_data_logger = DataLogger(test_dir / f"scenario_{scenario_name}")
        env_test.set_data_logger(test_data_logger)

        # Set specific scenario properties
        env_test.current_target_pos = np.array([scenario_config['distance'], 0.15, 0.35])

        scenario_results = []

        for episode in range(episodes_per_scenario):
            print(f"  Episode {episode+1}/{episodes_per_scenario}...", end=" ")

            obs = env_test.reset()
            episode_reward = 0
            episode_steps = 0
            max_fingers = 0
            min_distance = float('inf')
            max_hull_volume = 0

            while episode_steps < 1000:
                action, _ = model.predict(obs, deterministic=True)
                obs, reward, done, info = env_test.step(action)
                episode_reward += reward[0]
                episode_steps += 1

                # Track V2 specific metrics
                fingers = info[0].get('num_active_fingers', 0) if hasattr(info[0], 'get') else 0
                distance = info[0].get('distance', float('inf')) if hasattr(info[0], 'get') else float('inf')
                hull_volume = info[0].get('convex_hull_volume', 0) if hasattr(info[0], 'get') else 0

                max_fingers = max(max_fingers, fingers)
                min_distance = min(min_distance, distance)
                max_hull_volume = max(max_hull_volume, hull_volume)

                if done[0]:
                    break

                if visualize:
                    time.sleep(0.01)

            test_data_logger.new_episode()

            # Extract final results
            final_distance = info[0].get('distance', float('inf')) if hasattr(info[0], 'get') else float('inf')
            success = info[0].get('success', False) if hasattr(info[0], 'get') else False

            result = {
                'scenario': scenario_name,
                'episode': episode + 1,
                'success': success,
                'final_distance': final_distance,
                'min_distance': min_distance,
                'max_fingers': max_fingers,
                'max_hull_volume': max_hull_volume,
                'reward': episode_reward,
                'steps': episode_steps
            }

            scenario_results.append(result)
            all_results.append(result)

            status = "✅ SUCCESS" if success else "❌ FAILED"
            print(f"{status} | Dist: {final_distance:.3f}m | Fingers: {max_fingers} | Hull: {max_hull_volume:.6f}")

        # Save scenario data and create plots
        test_csv = test_data_logger.save_to_csv(f"{scenario_name}_test_data.csv")
        create_comprehensive_plots(test_csv, test_scenarios)

        env_test.close()

        # Scenario summary
        successes = sum(r['success'] for r in scenario_results)
        success_rate = successes / len(scenario_results)
        avg_final_distance = np.mean([r['final_distance'] for r in scenario_results])
        avg_max_fingers = np.mean([r['max_fingers'] for r in scenario_results])

        print(f"  📊 {scenario_name} Results:")
        print(f"     Success Rate: {success_rate:.1%} | Avg Final Dist: {avg_final_distance:.3f}m | Max Fingers: {avg_max_fingers:.1f}")

    # Overall V2 analysis
    print(f"\n{'='*80}")
    print("📈 V2 SC-1 COMPREHENSIVE TEST RESULTS")
    print(f"{'='*80}")

    total_episodes = len(all_results)
    total_successes = sum(r['success'] for r in all_results)
    overall_success_rate = total_successes / total_episodes

    overall_avg_distance = np.mean([r['final_distance'] for r in all_results])
    overall_avg_fingers = np.mean([r['max_fingers'] for r in all_results])
    overall_avg_hull = np.mean([r['max_hull_volume'] for r in all_results])

    print(f"🎯 Overall Success Rate: {total_successes}/{total_episodes} ({overall_success_rate:.1%})")
    print(f"📏 Average Final Distance: {overall_avg_distance:.3f}m")
    print(f"👆 Average Max Fingers: {overall_avg_fingers:.1f}")
    print(f"🔷 Average Max Hull Volume: {overall_avg_hull:.6f}")

    print(f"\n📁 All V2 test results saved to: {test_dir}")
    print("=" * 80)

    return all_results


def main():
    """Main V2 SC-1 training function with dual curriculum learning"""

    # Initialize WandB with comprehensive configuration
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = f"V2_SC1_{timestamp}"

    # Device detection for config
    device = "cuda" if torch.cuda.is_available() else "cpu"
    gpu_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "N/A"

    wandb_config = {
        # Training Configuration
        "algorithm": "PPO",
        "environment": "V2AllegroReachingEnv",
        "total_timesteps": 200000,
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
        "observation_space": 26,
        "action_space": 10,
        "max_episode_steps": 1000,
        "simulation_frequency": 240,

        # Reward Function Configuration
        "reward_components": 5,
        "reward_weights": {
            "distance": "exponential(-8*d)",
            "tendon_efficiency": 0.2,
            "tactile_contact": 0.15,
            "movement_penalty": -0.01,
            "convex_hull": 0.1,
            "success_bonus": 10.0
        },

        # Curriculum Learning Configuration
        "curriculum_type": "dual_reward_target",
        "reward_curriculum_phases": {
            "phase_1_distance": "0-40K timesteps",
            "phase_2_tendon": "40K-80K timesteps",
            "phase_3_tactile": "80K-120K timesteps",
            "phase_4_movement": "120K-160K timesteps",
            "phase_5_convex_hull": "160K-200K timesteps"
        },
        "target_curriculum_phases": {
            "phase_1_static": "160K+ timesteps",
            "phase_2_slow_moving": "180K+ timesteps",
            "phase_3_dynamic": "190K+ timesteps"
        },

        # Control Configuration
        "butterworth_filter": {
            "cutoff_frequency": 10.0,
            "sampling_frequency": 240.0,
            "order": 2
        },

        # Tendon Control Configuration
        "tendon_control": {
            "force_gain": 15.0,
            "damping": 0.8,
            "max_force": 60.0,
            "fingers": ["index", "middle", "ring", "thumb"]
        },

        # Checkpointing Configuration
        "checkpoint_frequency": 40000,
        "save_final_model": True,

        # Testing Configuration
        "test_scenarios": 5,
        "episodes_per_scenario": 10,

        # Architecture Details
        "architecture": "V2_simplified_enhanced",
        "removed_features": ["smoothness_penalty"],
        "added_features": ["convex_hull_reward", "butterworth_filtering", "dual_curriculum"],

        # Experiment Metadata
        "experiment_version": "V2.0",
        "base_scripts": ["SC-1.py", "Wandb_SC-1_Enhanced_Checkpointing.py"],
        "timestamp": timestamp
    }

    try:
        wandb.init(
            project="space-touch-v2-sc1",
            name=run_name,
            config=wandb_config,
            tags=["v2", "dual-curriculum", "butterworth-filter", "convex-hull", "tendon-control"],
            notes=f"V2 SC-1 training with dual curriculum learning, Butterworth filtering, and convex hull integration. 200K timesteps with enhanced WandB logging.",
            save_code=True  # Save the script code to WandB
        )
        print("✅ WandB initialized with comprehensive configuration")
        print(f"🌐 WandB Run: {wandb.run.url}")
    except Exception as e:
        print(f"⚠️  WandB initialization failed: {e}")
        print("   Continuing training without WandB logging...")
        wandb = None

    # Create organized training directory
    base_training_dir = Path("./SC1_Training_Runs/")
    base_training_dir.mkdir(parents=True, exist_ok=True)

    log_dir = base_training_dir / f"Run_{timestamp}_V2_SC1_DualCurriculum"
    log_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print(f"🚀 V2 SC-1 ENHANCED TRAINING WITH DUAL CURRICULUM LEARNING")
    print("=" * 80)
    print(f"Run Name: {run_name}")
    print(f"Log Directory: {log_dir}")
    print(f"Total Timesteps: 200,000")
    print("Features:")
    print("  ✓ Simplified 5-component reward function")
    print("  ✓ Butterworth low-pass filtering (10Hz)")
    print("  ✓ Dual curriculum learning (Reward → Target)")
    print("  ✓ Convex hull integration")
    print("  ✓ Comprehensive testing & plotting")
    print("  ✓ WandB experiment tracking")
    print("=" * 80)

    try:
        # Create data logger
        data_logger = DataLogger(log_dir)

        # Create V2 environment
        print("\n🏗️  Creating V2 tendon-controlled environment with curriculum learning...")
        env = V2AllegroReachingEnv(vis=False)

        # Create callbacks
        checkpoint_callback = CheckpointCallback(
            save_freq=40000,
            save_path=log_dir / "checkpoints",
            verbose=1
        )

        wandb_callback = WandBCallback(data_logger, log_freq=100) if wandb else None
        callbacks = [checkpoint_callback]
        if wandb_callback:
            callbacks.append(wandb_callback)

        # Configure logger
        tb_log_dir = log_dir / "tensorboard"
        tb_log_dir.mkdir(parents=True, exist_ok=True)
        custom_logger = configure(str(tb_log_dir), ["stdout", "tensorboard"])

        # Create PPO model
        print("🧠 Creating V2 PPO model...")

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
        print("V2 TRAINING CONFIGURATION")
        print("=" * 80)
        print("Reward Function (Curriculum):")
        print("  Phase 1: Distance reward only")
        print("  Phase 2: + Tendon efficiency reward")
        print("  Phase 3: + Tactile contact reward")
        print("  Phase 4: + Movement penalty")
        print("  Phase 5: + Convex hull reward")
        print("\nTarget Curriculum (After reward curriculum):")
        print("  Phase 1: Static targets")
        print("  Phase 2: Slow moving targets")
        print("  Phase 3: Dynamic targets")
        print("\nObservation Space: 26D (with convex hull)")
        print("Action Space: 10D (6 DOF base + 4 tendons)")
        print("Control Filtering: Butterworth 10Hz low-pass")
        print("Checkpoints: Every 40K timesteps")
        print("=" * 80)

        # Train the model
        print("\n🏃‍♂️ Starting V2 SC-1 training with dual curriculum...")

        model.learn(
            total_timesteps=200000,
            callback=callbacks,
            log_interval=10,
            tb_log_name=f"V2_SC1_{timestamp}",
            reset_num_timesteps=True,
            progress_bar=True
        )

        # Save final model
        final_model_path = log_dir / f"v2_sc1_final_model_{timestamp}"
        model.save(str(final_model_path))
        print(f"\n✅ V2 Training Complete! Final model saved: {final_model_path}.zip")

        # Run comprehensive testing with detailed logging
        print(f"\n🧪 Running comprehensive V2 testing...")
        env_test = V2AllegroReachingEnv(vis=False)
        test_data_logger = DataLogger(log_dir / "test_data")
        env_test.set_data_logger(test_data_logger)

        # Test for multiple episodes
        test_results = []
        for episode in range(10):
            print(f"  Test Episode {episode+1}/10... ", end="")

            obs = env_test.reset()
            episode_reward = 0
            episode_steps = 0

            while episode_steps < 1000:
                action, _ = model.predict(obs, deterministic=True)
                obs, reward, done, info = env_test.step(action)
                episode_reward += reward[0]
                episode_steps += 1

                if done[0]:
                    break

            test_data_logger.new_episode()

            distance = info[0].get('distance', float('inf'))
            success = info[0].get('success', False)

            test_results.append({
                'episode': episode + 1,
                'success': success,
                'distance': distance,
                'reward': episode_reward,
                'steps': episode_steps
            })

            status = "✅ SUCCESS" if success else "❌ FAILED"
            print(f"{status} | Distance: {distance:.3f}m | Reward: {episode_reward:.1f}")

        env_test.close()

        # Save test data and create comprehensive plots
        test_csv = test_data_logger.save_to_csv("v2_sc1_test_data.csv")
        print(f"\n✅ Test data saved: {test_csv}")

        # Create comprehensive plots
        print(f"📊 Generating V2 comprehensive analysis plots...")
        create_comprehensive_plots(test_csv)

        # Test summary
        successes = sum(r['success'] for r in test_results)
        success_rate = successes / len(test_results)
        avg_distance = np.mean([r['distance'] for r in test_results])
        avg_reward = np.mean([r['reward'] for r in test_results])

        print("\n" + "=" * 80)
        print("V2 SC-1 TEST RESULTS SUMMARY")
        print("=" * 80)
        print(f"Success Rate: {successes}/{len(test_results)} ({success_rate:.1%})")
        print(f"Average Distance: {avg_distance:.3f}m")
        print(f"Average Reward: {avg_reward:.1f}")
        print("=" * 80)

        # Enhanced final results logging to WandB
        if wandb:
            # Calculate additional final metrics
            final_metrics = {
                "final/success_rate": success_rate,
                "final/avg_distance": avg_distance,
                "final/avg_reward": avg_reward,
                "final/total_episodes_tested": len(test_results),
                "final/successful_episodes": successes,
                "final/training_timesteps": 200000,
                "final/checkpoint_count": 5,  # Every 40K timesteps
            }

            # Add per-scenario analysis if available
            if hasattr(env_test, 'reward_curriculum_phase'):
                final_metrics.update({
                    "final/reward_curriculum_phase": env_test.reward_curriculum_phase,
                    "final/target_curriculum_phase": env_test.target_curriculum_phase,
                })

            # Log curriculum effectiveness
            if len(test_results) > 0:
                distances = [r['distance'] for r in test_results]
                rewards = [r['reward'] for r in test_results]
                final_metrics.update({
                    "final/distance_std": np.std(distances),
                    "final/reward_std": np.std(rewards),
                    "final/distance_min": np.min(distances),
                    "final/distance_max": np.max(distances),
                    "final/reward_min": np.min(rewards),
                    "final/reward_max": np.max(rewards),
                })

            # Performance assessment
            if success_rate >= 0.3:
                performance_grade = "Excellent"
            elif success_rate >= 0.2:
                performance_grade = "Good"
            elif success_rate >= 0.1:
                performance_grade = "Fair"
            else:
                performance_grade = "Needs Improvement"

            final_metrics["final/performance_grade"] = performance_grade

            # Log all final metrics
            wandb.log(final_metrics)

            # Create WandB summary
            wandb.run.summary.update({
                "best_success_rate": success_rate,
                "best_avg_distance": avg_distance,
                "total_training_time_hours": (time.time() - getattr(wandb.run, '_start_time', time.time())) / 3600,
                "performance_assessment": performance_grade,
                "curriculum_phases_completed": 5,  # All reward phases
                "architecture_version": "V2.0"
            })

            # Log training artifacts
            try:
                # Log the final model as an artifact
                model_artifact = wandb.Artifact(
                    name=f"v2_sc1_final_model_{timestamp}",
                    type="model",
                    description="V2 SC-1 final trained model with dual curriculum learning"
                )
                model_artifact.add_file(str(final_model_path) + ".zip")
                wandb.log_artifact(model_artifact)

                # Log test data as artifact
                if test_csv.exists():
                    data_artifact = wandb.Artifact(
                        name=f"v2_sc1_test_data_{timestamp}",
                        type="dataset",
                        description="V2 SC-1 comprehensive test results and metrics"
                    )
                    data_artifact.add_file(str(test_csv))
                    wandb.log_artifact(data_artifact)

                print("✅ Training artifacts logged to WandB")

            except Exception as e:
                print(f"⚠️  Could not log artifacts to WandB: {e}")

        print("\n" + "=" * 80)
        print("🎉 V2 SC-1 TRAINING & TESTING COMPLETE!")
        print("=" * 80)
        print(f"📁 All results: {log_dir}")
        print(f"🤖 Final model: {final_model_path}.zip")
        print(f"📊 Test plots: {log_dir}/test_data/plots/")
        print(f"📈 TensorBoard: tensorboard --logdir {tb_log_dir}")
        if wandb:
            print(f"🌐 WandB: {wandb.run.url}")
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
            if wandb:
                wandb.finish()
        except:
            pass


if __name__ == "__main__":
    main()