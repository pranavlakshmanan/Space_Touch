#!/usr/bin/env python3
import time
import numpy as np
import cv2
import pybullet as p
import pybullet_data
import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import VecEnv
from stable_baselines3.common.callbacks import BaseCallback
import os
import pandas as pd
from datetime import datetime
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
import torch
import wandb
import sys
from scipy.signal import butter, lfilter

# Fix for attrdict Python 3.13 compatibility
import collections.abc
import collections
collections.Mapping = collections.abc.Mapping
collections.MutableMapping = collections.abc.MutableMapping
collections.Sequence = collections.abc.Sequence
collections.MutableSequence = collections.abc.MutableSequence


class LowPassFilter:
    """Low pass filter for smoothing control inputs"""

    def __init__(self, cutoff_freq=10.0, sampling_freq=240.0, order=2):
        """
        Initialize low pass filter

        Args:
            cutoff_freq: Cutoff frequency in Hz
            sampling_freq: Sampling frequency in Hz (simulation frequency)
            order: Filter order
        """
        self.cutoff_freq = cutoff_freq
        self.sampling_freq = sampling_freq
        self.order = order

        # Calculate filter coefficients
        nyquist_freq = 0.5 * sampling_freq
        normalized_cutoff = cutoff_freq / nyquist_freq
        self.b, self.a = butter(order, normalized_cutoff, btype='low', analog=False)

        # Initialize filter history
        self.reset()

    def reset(self):
        """Reset filter internal state"""
        # Initialize with zeros - will be updated as we receive data
        self.x_history = []
        self.y_history = []

    def filter(self, signal):
        """
        Apply low pass filter to signal

        Args:
            signal: Input signal (numpy array)

        Returns:
            filtered_signal: Low-pass filtered signal
        """
        if len(self.x_history) == 0:
            # First call - initialize with current signal
            self.x_history = [signal.copy() for _ in range(self.order + 1)]
            self.y_history = [signal.copy() for _ in range(self.order)]
            return signal

        # Update input history
        self.x_history = [signal.copy()] + self.x_history[:self.order]

        # Apply filter equation: y[n] = b0*x[n] + b1*x[n-1] + ... - a1*y[n-1] - a2*y[n-2] - ...
        filtered = np.zeros_like(signal)

        # Add input terms
        for i in range(min(len(self.b), len(self.x_history))):
            filtered += self.b[i] * self.x_history[i]

        # Subtract output terms (skip a[0] which is always 1.0)
        for i in range(1, min(len(self.a), len(self.y_history) + 1)):
            if i-1 < len(self.y_history):
                filtered -= self.a[i] * self.y_history[i-1]

        # Update output history
        self.y_history = [filtered.copy()] + self.y_history[:self.order-1]

        return filtered


class DataLogger:
    """Handles data logging for analysis and plotting"""

    def __init__(self, log_dir):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)

        # Initialize data storage
        self.data = {
            'timestamp': [], 'step': [], 'episode': [],
            'base_pos_x': [], 'base_pos_y': [], 'base_pos_z': [],
            'base_vel_x': [], 'base_vel_y': [], 'base_vel_z': [],
            'base_ang_vel_x': [], 'base_ang_vel_y': [], 'base_ang_vel_z': [],
            'ee1_pos_x': [], 'ee1_pos_y': [], 'ee1_pos_z': [],
            'ee2_pos_x': [], 'ee2_pos_y': [], 'ee2_pos_z': [],
            'ee3_pos_x': [], 'ee3_pos_y': [], 'ee3_pos_z': [],
            'ee4_pos_x': [], 'ee4_pos_y': [], 'ee4_pos_z': [],
            'target_x': [], 'target_y': [], 'target_z': [],
            'distance_to_target': [], 'ee_target_distances': [],
            'tendon_force_index': [], 'tendon_force_middle': [],
            'tendon_force_ring': [], 'tendon_force_thumb': [],
            'control_linear_x': [], 'control_linear_y': [], 'control_linear_z': [],
            'control_angular_x': [], 'control_angular_y': [], 'control_angular_z': [],
            'control_linear_x_filtered': [], 'control_linear_y_filtered': [], 'control_linear_z_filtered': [],
            'control_angular_x_filtered': [], 'control_angular_y_filtered': [], 'control_angular_z_filtered': [],
            'tactile_contact_finger1': [], 'tactile_contact_finger2': [],
            'tactile_contact_finger3': [], 'tactile_contact_finger4': [],
            'reward': [], 'distance_reward': [], 'tendon_efficiency_reward': [],
            'acceleration_penalty': [], 'jerk_penalty': [], 'success': [],
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

    def save_to_csv(self, filename="sc1_training_data.csv"):
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
    """Implements tendon-based control with reference axis and torque-based actuation"""

    def __init__(self, hand_id, joint_names, joint_indices):
        self.hand_id = hand_id
        self.joint_names = joint_names
        self.joint_indices = joint_indices

        self.TENDON_FORCE_GAIN = 15.0
        self.TENDON_DAMPING = 0.8
        self.MAX_TENDON_FORCE = 60.0

        self.FINGER_CHAINS = {
            "index": ["joint_8.0", "joint_9.0", "joint_10.0", "joint_11.0"],
            "middle": ["joint_4.0", "joint_5.0", "joint_6.0", "joint_7.0"],
            "ring": ["joint_0.0", "joint_1.0", "joint_2.0", "joint_3.0"],
            "thumb": ["joint_12.0", "joint_13.0", "joint_14.0", "joint_15.0"]
        }

        self.FINGER_REFERENCE_AXES = {
            "index": {"start": np.array([0.0, -0.02, 0.0]), "end": np.array([0.0, -0.02, 0.08])},
            "middle": {"start": np.array([0.0, -0.01, 0.0]), "end": np.array([0.0, -0.01, 0.08])},
            "ring": {"start": np.array([0.0, 0.01, 0.0]), "end": np.array([0.0, 0.01, 0.08])},
            "thumb": {"start": np.array([-0.02, 0.0, 0.0]), "end": np.array([-0.02, 0.0, 0.06])}
        }

        self.name_to_idx = {name: idx for name, idx in zip(joint_names, joint_indices)}

        self.finger_joints = {}
        for finger, chain in self.FINGER_CHAINS.items():
            self.finger_joints[finger] = []
            for joint_name in chain:
                if joint_name in self.name_to_idx:
                    self.finger_joints[finger].append(self.name_to_idx[joint_name])

        self.joint_moment_arms = {}
        self._compute_moment_arms()

    def _compute_moment_arms(self):
        """Pre-compute moment arms for each joint relative to its finger's reference axis"""
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
        """Compute perpendicular distance from point to line (moment arm)"""
        point_vec = point - axis_start
        projection_length = np.dot(point_vec, axis_direction)
        projection_point = axis_start + projection_length * axis_direction
        moment_arm_vec = point - projection_point
        moment_arm = np.linalg.norm(moment_arm_vec)
        return max(moment_arm, 0.005)

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


class CheckpointCallback(BaseCallback):
    """Callback for saving model checkpoints at regular intervals"""

    def __init__(self, checkpoint_freq=200000, save_path="./checkpoints/", verbose=0):
        super(CheckpointCallback, self).__init__(verbose)
        self.checkpoint_freq = checkpoint_freq
        self.save_path = Path(save_path)
        self.save_path.mkdir(parents=True, exist_ok=True)

    def _on_step(self) -> bool:
        if self.n_calls % self.checkpoint_freq == 0:
            checkpoint_path = self.save_path / f"checkpoint_{self.n_calls}"
            self.model.save(str(checkpoint_path))
            if self.verbose > 0:
                print(f"✓ Checkpoint saved at {self.n_calls} timesteps: {checkpoint_path}.zip")
        return True


class WandBCallback(BaseCallback):
    """WandB callback for comprehensive logging"""

    def __init__(self, data_logger, log_freq=100, verbose=0):
        super(WandBCallback, self).__init__(verbose)
        self.data_logger = data_logger
        self.log_freq = log_freq
        self.episode_rewards = []
        self.episode_lengths = []
        self.episode_successes = []

        self.recent_distances = []
        self.recent_tendon_forces = []
        self.recent_tactile_contacts = []
        self.recent_accel_penalties = []
        self.recent_jerk_penalties = []

    def _on_step(self) -> bool:
        if self.locals.get('infos'):
            for info in self.locals['infos']:
                if 'episode' in info:
                    ep_reward = info['episode']['r']
                    ep_length = info['episode']['l']

                    self.episode_rewards.append(ep_reward)
                    self.episode_lengths.append(ep_length)
                    self.data_logger.new_episode()

                    # Log episode metrics to WandB
                    wandb.log({
                        'episode/reward': ep_reward,
                        'episode/length': ep_length,
                        'episode/count': len(self.episode_rewards)
                    }, step=self.num_timesteps)

                    if len(self.episode_rewards) >= 100:
                        wandb.log({
                            'episode/reward_mean_100': np.mean(self.episode_rewards[-100:]),
                            'episode/length_mean_100': np.mean(self.episode_lengths[-100:])
                        }, step=self.num_timesteps)

                if 'distance' in info:
                    self.recent_distances.append(info['distance'])

                if 'success' in info and 'episode' in info:
                    self.episode_successes.append(float(info['success']))

                if 'tendon_forces' in info:
                    avg_tendon_force = np.mean(list(info['tendon_forces'].values()))
                    self.recent_tendon_forces.append(avg_tendon_force)

                if 'tactile_contacts' in info:
                    total_contacts = sum(info['tactile_contacts'])
                    self.recent_tactile_contacts.append(total_contacts)

                if 'acceleration_penalty' in info:
                    self.recent_accel_penalties.append(info['acceleration_penalty'])

                if 'jerk_penalty' in info:
                    self.recent_jerk_penalties.append(info['jerk_penalty'])

        # Log aggregated metrics at regular intervals
        if self.num_timesteps % self.log_freq == 0:
            log_dict = {'training/timesteps': self.num_timesteps}

            if self.recent_distances:
                log_dict['metrics/distance_mean'] = np.mean(self.recent_distances)
                log_dict['metrics/distance_min'] = np.min(self.recent_distances)
                self.recent_distances = []

            if self.recent_tendon_forces:
                log_dict['tendon/average_force'] = np.mean(self.recent_tendon_forces)
                log_dict['tendon/max_force'] = np.max(self.recent_tendon_forces)
                self.recent_tendon_forces = []

            if self.recent_tactile_contacts:
                log_dict['tactile/contact_rate'] = np.mean(self.recent_tactile_contacts)
                self.recent_tactile_contacts = []

            if self.recent_accel_penalties:
                log_dict['penalty/acceleration_mean'] = np.mean(self.recent_accel_penalties)
                self.recent_accel_penalties = []

            if self.recent_jerk_penalties:
                log_dict['penalty/jerk_mean'] = np.mean(self.recent_jerk_penalties)
                self.recent_jerk_penalties = []

            if self.episode_successes:
                success_rate = np.mean(self.episode_successes[-100:]) if len(self.episode_successes) >= 100 else np.mean(self.episode_successes)
                log_dict['performance/success_rate'] = success_rate
                log_dict['performance/total_successes'] = sum(self.episode_successes)

            wandb.log(log_dict, step=self.num_timesteps)

        return True


class TendonAllegroReachingEnv(VecEnv):
    """Enhanced environment with delta-distance reward, adaptive episode length, and control smoothing"""

    def __init__(self,
                 num_envs=1,
                 vis=False,
                 max_steps=500,
                 urdf_hand="/home/pralak/Space_Touch/examples/allegro_hand_description/allegro_hand_description_left_digit_fixed.urdf",
                 target_range=0.3,
                 control_smoothing=True,
                 filter_cutoff=15.0):

        self.num_envs = num_envs
        self.vis = vis
        self.base_max_steps = max_steps
        self.max_steps = max_steps  # Will be adapted per episode
        self.urdf_hand = urdf_hand
        self.target_range = target_range
        self.control_smoothing = control_smoothing

        self.target_pos = np.array([0.25, 0.15, 0.35])

        # Set simulation frequency first (needed by _init_pybullet)
        self.sim_freq = 240.0

        self._init_pybullet()

        self.hand = None
        self.target_sphere = None
        self.tendon_controller = None
        self.hand_spawned = False

        # Initialize low-pass filters for control smoothing
        if self.control_smoothing:
            self.linear_filter = LowPassFilter(cutoff_freq=filter_cutoff, sampling_freq=self.sim_freq)
            self.angular_filter = LowPassFilter(cutoff_freq=filter_cutoff, sampling_freq=self.sim_freq)
            if not hasattr(self, '_filter_init_printed'):
                print(f"✓ Control smoothing enabled (cutoff: {filter_cutoff} Hz)")
                self._filter_init_printed = True

        # Action space: 6 DOF base movement + 4 tendon forces
        action_dim = 10
        self.action_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(action_dim,), dtype=np.float32)

        # Observation: base_pos(3) + target_pos(3) + base_vel(3) + finger_positions(12) + binary_tactile(4)
        obs_dim = 3 + 3 + 3 + 12 + 4  # = 25
        self.observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32)

        super().__init__(num_envs, self.observation_space, self.action_space)

        self.step_counts = np.zeros(num_envs, dtype=np.int32)
        self.episode_rewards = np.zeros(num_envs)
        self.episode_lengths = np.zeros(num_envs, dtype=np.int32)

        # Enhanced action tracking for jerk penalty
        self.prev_actions = np.zeros((num_envs, action_dim), dtype=np.float32)
        self.prev_prev_actions = np.zeros((num_envs, action_dim), dtype=np.float32)
        self.prev_prev_prev_actions = np.zeros((num_envs, action_dim), dtype=np.float32)

        # Track filtered actions
        self.filtered_actions = np.zeros((num_envs, action_dim), dtype=np.float32)

        # Track previous distance for delta reward calculation
        self.prev_distance = np.zeros(num_envs, dtype=np.float32)

        # Track initial distance for adaptive episode length
        self.initial_distance = np.zeros(num_envs, dtype=np.float32)

        self.data_logger = None

        self.reset()

    def _init_pybullet(self):
        """Initialize PyBullet connection"""
        self.client_id = p.connect(p.DIRECT)

        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.setGravity(0, 0, -9.81)
        p.setRealTimeSimulation(0)
        p.setTimeStep(1.0/self.sim_freq)

        p.loadURDF("plane.urdf", [0, 0, 0], useFixedBase=True)

    def set_data_logger(self, data_logger):
        """Set the data logger for this environment"""
        self.data_logger = data_logger

    def _setup_simulation(self):
        """Setup the simulation environment"""
        try:
            # CHECK: Prevent multiple hand spawning
            if self.hand_spawned and self.hand is not None:
                try:
                    p.getBasePositionAndOrientation(self.hand)
                    print("⚠ Hand already exists, skipping spawn")
                    return
                except:
                    self.hand_spawned = False

            # Load hand
            if os.path.exists(self.urdf_hand):
                self.hand = p.loadURDF(
                    self.urdf_hand,
                    basePosition=[0, 0, 0.2],
                    baseOrientation=p.getQuaternionFromEuler([0, 0, 0]),
                    useFixedBase=False
                )
                self.hand_spawned = True
                if not hasattr(self, '_hand_loaded_printed'):
                    print(f"✓ Hand loaded from: {self.urdf_hand}")
                    self._hand_loaded_printed = True
            else:
                print(f"Hand URDF not found: {self.urdf_hand}")
                hand_collision = p.createCollisionShape(p.GEOM_BOX, halfExtents=[0.1, 0.05, 0.02])
                hand_visual = p.createVisualShape(p.GEOM_BOX, halfExtents=[0.1, 0.05, 0.02], rgbaColor=[0.8, 0.6, 0.4, 1])
                self.hand = p.createMultiBody(
                    baseMass=1.0,
                    baseCollisionShapeIndex=hand_collision,
                    baseVisualShapeIndex=hand_visual,
                    basePosition=[0, 0, 0.2]
                )
                self.hand_spawned = True

            # Create target sphere
            target_collision = p.createCollisionShape(p.GEOM_SPHERE, radius=0.02)
            target_visual = p.createVisualShape(p.GEOM_SPHERE, radius=0.02, rgbaColor=[1, 0, 0, 1])
            self.target_sphere = p.createMultiBody(
                baseMass=0.1,
                baseCollisionShapeIndex=target_collision,
                baseVisualShapeIndex=target_visual,
                basePosition=self.target_pos
            )

            # Collect actuated joints
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

            # Initialize fingertip link indices for tactile sensing
            self.fingertip_links = []
            tip_labels = ["joint_15.0_tip", "joint_11.0_tip", "joint_7.0_tip", "joint_3.0_tip"]

            for i in range(num_joints):
                joint_info = p.getJointInfo(self.hand, i)
                joint_name = joint_info[1].decode()
                if joint_name in tip_labels:
                    self.fingertip_links.append(i)

            if not hasattr(self, '_tactile_init_printed'):
                print(f"✓ Binary tactile sensing enabled on {len(self.fingertip_links)} fingertips")
                self._tactile_init_printed = True

            # Let simulation settle
            for _ in range(50):
                p.stepSimulation()

        except Exception as e:
            print(f"Error setting up simulation: {e}")
            raise

    def _get_finger_positions(self):
        """Get positions of all fingertips"""
        finger_positions = []

        if self.hand is None:
            return np.zeros(12)

        num_joints = p.getNumJoints(self.hand)
        tip_labels = ["joint_15.0_tip", "joint_11.0_tip", "joint_7.0_tip", "joint_3.0_tip"]

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
        """Get binary tactile contact feedback (0 or 1 per fingertip)"""
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

    def _get_observation(self):
        """Get current observation"""
        try:
            if self.hand is None:
                obs = np.zeros(self.observation_space.shape[0])
                return np.expand_dims(obs.astype(np.float32), axis=0)

            base_pos, _ = p.getBasePositionAndOrientation(self.hand)
            base_vel, _ = p.getBaseVelocity(self.hand)

            base_pos = np.array(base_pos)
            base_vel = np.array(base_vel)

            finger_positions = self._get_finger_positions()
            binary_tactile = self._get_binary_tactile_feedback()

            obs = np.concatenate([
                base_pos,           # 3D
                self.target_pos,    # 3D
                base_vel,           # 3D
                finger_positions,   # 12D
                binary_tactile,     # 4D
            ])

            return np.expand_dims(obs.astype(np.float32), axis=0)

        except Exception as e:
            print(f"Error getting observation: {e}")
            return np.zeros((1, self.observation_space.shape[0]), dtype=np.float32)

    def _compute_adaptive_max_steps(self, initial_distance):
        """Compute adaptive episode length based on initial distance"""
        # Estimate steps needed to reach 80% of initial distance
        avg_velocity = 0.15  # Conservative estimate
        time_to_80_percent = (0.8 * initial_distance) / avg_velocity

        # Convert to steps (240 Hz simulation)
        steps_to_80_percent = int(time_to_80_percent * self.sim_freq)

        # Add 20% buffer
        adaptive_steps = int(steps_to_80_percent * 1.2)

        # Clamp between reasonable bounds
        min_steps = 50
        max_steps = self.base_max_steps
        adaptive_steps = np.clip(adaptive_steps, min_steps, max_steps)

        return adaptive_steps

    def step_wait(self):
        """Execute one step of the environment - Enhanced with control smoothing and jerk penalty"""
        self.step_counts += 1
        self.episode_lengths += 1
        actions = self.actions[0]

        try:
            if self.hand is None:
                obs = self._get_observation()
                return obs, np.array([-1.0], dtype=np.float32), np.array([True]), [{"error": "Hand not loaded"}]

            base_actions = actions[:6]
            tendon_actions = actions[6:10]

            # Apply control smoothing if enabled
            if self.control_smoothing:
                filtered_linear = self.linear_filter.filter(base_actions[:3])
                filtered_angular = self.angular_filter.filter(base_actions[3:6])
                filtered_base_actions = np.concatenate([filtered_linear, filtered_angular])
            else:
                filtered_base_actions = base_actions

            # Store filtered actions for logging
            self.filtered_actions[0, :6] = filtered_base_actions
            self.filtered_actions[0, 6:10] = tendon_actions

            # Apply base movement with filtered actions
            linear_vel = filtered_base_actions[:3] * 0.3
            angular_vel = filtered_base_actions[3:6] * 0.8

            p.resetBaseVelocity(
                self.hand,
                linearVelocity=linear_vel,
                angularVelocity=angular_vel
            )

            # Apply tendon forces (unfiltered - physical forces don't need smoothing)
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
                p.setJointMotorControlArray(
                    bodyUniqueId=self.hand,
                    jointIndices=self.tendon_controller.joint_indices,
                    controlMode=p.TORQUE_CONTROL,
                    forces=torques.tolist()
                )

            p.stepSimulation()

            # Calculate reward components
            base_pos, base_orn = p.getBasePositionAndOrientation(self.hand)
            base_vel, base_ang_vel = p.getBaseVelocity(self.hand)
            base_pos = np.array(base_pos)
            distance = np.linalg.norm(base_pos - self.target_pos)

            finger_positions = self._get_finger_positions()
            binary_tactile = self._get_binary_tactile_feedback()

            # 1. Delta-distance reward (reward for reducing distance)
            distance_delta = self.prev_distance[0] - distance
            distance_reward = 10.0 * distance_delta

            # Update previous distance for next step
            self.prev_distance[0] = distance

            # 2. Tendon efficiency reward
            tendon_efficiency = 1.0 - 0.5 * np.mean(tendon_forces)
            tendon_efficiency_reward = 0.2 * tendon_efficiency

            # 3. Tactile contact penalty (penalize accidental contact)
            tactile_contact_penalty = -0.5 * np.sum(binary_tactile)

            # 4. Movement efficiency penalty
            linear_vel_magnitude = np.linalg.norm(linear_vel)
            angular_vel_magnitude = np.linalg.norm(angular_vel)
            movement_penalty = -0.01 * (linear_vel_magnitude + angular_vel_magnitude)

            # 5. Enhanced acceleration penalty (smoothness)
            current_vel = actions[:6]
            prev_vel = self.prev_actions[0][:6]
            accel = current_vel - prev_vel
            accel_magnitude = np.linalg.norm(accel)
            acceleration_penalty = -0.15 * accel_magnitude  # Increased from 0.1

            # 6. NEW: Jerk penalty (rate of change of acceleration) - ENHANCED
            prev_accel = prev_vel - self.prev_prev_actions[0][:6]
            jerk = accel - prev_accel
            jerk_magnitude = np.linalg.norm(jerk)
            jerk_penalty = -0.25 * jerk_magnitude  # Heavy penalty for jerky movements

            # 7. Success bonus
            success_bonus = 10.0 if distance < 0.1 else 0.0

            # Combine all reward components
            total_reward = (distance_reward + tendon_efficiency_reward + tactile_contact_penalty +
                           movement_penalty + acceleration_penalty + jerk_penalty + success_bonus)

            # Update action history for jerk computation
            self.prev_prev_prev_actions[0] = self.prev_prev_actions[0].copy()
            self.prev_prev_actions[0] = self.prev_actions[0].copy()
            self.prev_actions[0] = actions.copy()

            self.episode_rewards[0] += total_reward

            success = distance < 0.1

            # Termination conditions with adaptive max_steps
            dones = np.array([
                self.step_counts[0] >= self.max_steps or
                success or
                distance > 2.0 or
                base_pos[2] < 0.05
            ])

            # Store data for logging
            self.step_data = {
                "base_pos_x": base_pos[0], "base_pos_y": base_pos[1], "base_pos_z": base_pos[2],
                "base_vel_x": base_vel[0], "base_vel_y": base_vel[1], "base_vel_z": base_vel[2],
                "base_ang_vel_x": base_ang_vel[0], "base_ang_vel_y": base_ang_vel[1], "base_ang_vel_z": base_ang_vel[2],
                "target_x": self.target_pos[0], "target_y": self.target_pos[1], "target_z": self.target_pos[2],
                "distance_to_target": distance,
                "tendon_force_index": tendon_forces[0], "tendon_force_middle": tendon_forces[1],
                "tendon_force_ring": tendon_forces[2], "tendon_force_thumb": tendon_forces[3],
                "control_linear_x": base_actions[0], "control_linear_y": base_actions[1], "control_linear_z": base_actions[2],
                "control_angular_x": base_actions[3], "control_angular_y": base_actions[4], "control_angular_z": base_actions[5],
                "control_linear_x_filtered": self.filtered_actions[0, 0], "control_linear_y_filtered": self.filtered_actions[0, 1],
                "control_linear_z_filtered": self.filtered_actions[0, 2],
                "control_angular_x_filtered": self.filtered_actions[0, 3], "control_angular_y_filtered": self.filtered_actions[0, 4],
                "control_angular_z_filtered": self.filtered_actions[0, 5],
                "tactile_contact_finger1": binary_tactile[0], "tactile_contact_finger2": binary_tactile[1],
                "tactile_contact_finger3": binary_tactile[2], "tactile_contact_finger4": binary_tactile[3],
                "reward": total_reward,
                "distance_reward": distance_reward,
                "tendon_efficiency_reward": tendon_efficiency_reward,
                "acceleration_penalty": acceleration_penalty,
                "jerk_penalty": jerk_penalty,
                "success": success,
            }

            if self.data_logger is not None:
                self.data_logger.log_step(self.step_data)

            # Info for callbacks
            infos = [{
                "success": success,
                "distance": distance,
                "distance_reward": distance_reward,
                "distance_delta": distance_delta,
                "tendon_efficiency_reward": tendon_efficiency_reward,
                "acceleration_penalty": acceleration_penalty,
                "jerk_penalty": jerk_penalty,
                "tendon_forces": tendon_force_dict,
                "tactile_contacts": binary_tactile.tolist(),
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
        """Reset environment - ENSURES proper reset during testing"""
        if np.any(dones):
            # Clean up old bodies
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

            # Reset filters on environment reset
            if self.control_smoothing:
                self.linear_filter.reset()
                self.angular_filter.reset()

            # Setup new simulation
            self._setup_simulation()

            # Calculate initial distance and set adaptive max_steps
            if self.hand is not None:
                base_pos, _ = p.getBasePositionAndOrientation(self.hand)
                initial_distance = np.linalg.norm(np.array(base_pos) - self.target_pos)
                self.initial_distance[0] = initial_distance
                self.prev_distance[0] = initial_distance

                # Set adaptive episode length
                self.max_steps = self._compute_adaptive_max_steps(initial_distance)

                if not hasattr(self, '_adaptive_steps_printed'):
                    print(f"✓ Adaptive episode length enabled")
                    print(f"  Initial distance: {initial_distance:.3f}m")
                    print(f"  Adaptive max steps: {self.max_steps}")
                    self._adaptive_steps_printed = True

            # Reset counters and action history
            self.step_counts[dones] = 0
            self.episode_rewards[dones] = 0
            self.episode_lengths[dones] = 0
            self.prev_actions[dones] = 0
            self.prev_prev_actions[dones] = 0
            self.prev_prev_prev_actions[dones] = 0
            self.filtered_actions[dones] = 0
            self.current_tendon_forces = np.zeros(4)

        return self._get_observation()

    def reset(self):
        """Full environment reset - ENSURES clean state"""
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


def load_latest_checkpoint(checkpoint_dir, model_class=PPO, env=None):
    """Load the latest checkpoint from directory"""
    checkpoint_path = Path(checkpoint_dir)
    if not checkpoint_path.exists():
        return None, 0

    # Find all checkpoint files
    checkpoint_files = list(checkpoint_path.glob("checkpoint_*.zip"))
    if not checkpoint_files:
        return None, 0

    # Sort by timestep number
    def extract_timestep(path):
        try:
            return int(path.stem.split('_')[1])
        except:
            return 0

    checkpoint_files.sort(key=extract_timestep)
    latest_checkpoint = checkpoint_files[-1]
    latest_timestep = extract_timestep(latest_checkpoint)

    print(f"Loading checkpoint: {latest_checkpoint}")
    print(f"Resume from timestep: {latest_timestep}")

    # Load the model
    model = model_class.load(str(latest_checkpoint.with_suffix('')), env=env)

    return model, latest_timestep


def create_plots(csv_file):
    """Create comprehensive plots from the logged data"""
    print(f"Creating plots from {csv_file}...")

    try:
        df = pd.read_csv(csv_file)
        print(f"Loaded data with {len(df)} rows and {len(df.columns)} columns")

        if len(df) == 0:
            print("No data to plot")
            return

        if 'episode' in df.columns:
            df['episode'] = pd.to_numeric(df['episode'], errors='coerce')
            df['episode'] = df['episode'].ffill().fillna(0).astype(int)

        max_plot_points = 10000
        if len(df) > max_plot_points:
            print(f"Sampling {max_plot_points} points from {len(df)} for plotting performance")
            step_size = len(df) // max_plot_points
            df_plot = df.iloc[::step_size].copy()
        else:
            df_plot = df.copy()

        plot_dir = csv_file.parent / "plots"
        plot_dir.mkdir(exist_ok=True)

        plt.style.use('default')
        plt.rcParams['agg.path.chunksize'] = 10000
        plt.rcParams['figure.max_open_warning'] = 0

        step_vals = df_plot['step'].values

        print(f"Creating plots with {len(df_plot)} sampled data points...")

        # 1. Control Smoothing Analysis (NEW PLOT)
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('Enhanced Control Analysis with Smoothing', fontsize=16)

        # Raw vs Filtered Control Inputs
        if 'control_linear_x_filtered' in df_plot.columns:
            axes[0, 0].plot(step_vals, df_plot['control_linear_x'].values, label='Raw X', alpha=0.6, linewidth=0.8)
            axes[0, 0].plot(step_vals, df_plot['control_linear_x_filtered'].values, label='Filtered X', linewidth=1.2)
            axes[0, 0].plot(step_vals, df_plot['control_linear_y'].values, label='Raw Y', alpha=0.6, linewidth=0.8)
            axes[0, 0].plot(step_vals, df_plot['control_linear_y_filtered'].values, label='Filtered Y', linewidth=1.2)
            axes[0, 0].set_title('Linear Control: Raw vs Low-Pass Filtered')
        else:
            axes[0, 0].plot(step_vals, df_plot['control_linear_x'].values, label='X', linewidth=0.8)
            axes[0, 0].plot(step_vals, df_plot['control_linear_y'].values, label='Y', linewidth=0.8)
            axes[0, 0].plot(step_vals, df_plot['control_linear_z'].values, label='Z', linewidth=0.8)
            axes[0, 0].set_title('Linear Control Inputs')
        axes[0, 0].set_xlabel('Step')
        axes[0, 0].set_ylabel('Control Input')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)

        # Control input magnitude comparison
        control_magnitude_raw = np.sqrt(df_plot['control_linear_x']**2 + df_plot['control_linear_y']**2 + df_plot['control_linear_z']**2)
        axes[0, 1].plot(step_vals, control_magnitude_raw.values, alpha=0.7, linewidth=0.8, label='Raw Magnitude')
        if 'control_linear_x_filtered' in df_plot.columns:
            control_magnitude_filtered = np.sqrt(df_plot['control_linear_x_filtered']**2 + df_plot['control_linear_y_filtered']**2 + df_plot['control_linear_z_filtered']**2)
            axes[0, 1].plot(step_vals, control_magnitude_filtered.values, alpha=0.9, linewidth=1.2, label='Filtered Magnitude')
            axes[0, 1].legend()
        axes[0, 1].set_title('Control Input Magnitude')
        axes[0, 1].set_xlabel('Step')
        axes[0, 1].set_ylabel('Control Magnitude')
        axes[0, 1].grid(True, alpha=0.3)

        # Enhanced penalty analysis
        if 'jerk_penalty' in df_plot.columns:
            axes[1, 0].plot(step_vals, df_plot['acceleration_penalty'].values, alpha=0.8, linewidth=0.8, color='red', label='Accel Penalty')
            axes[1, 0].plot(step_vals, df_plot['jerk_penalty'].values, alpha=0.8, linewidth=0.8, color='darkred', label='Jerk Penalty')
            axes[1, 0].set_title('Enhanced Smoothness Penalties')
            axes[1, 0].legend()
        else:
            axes[1, 0].plot(step_vals, df_plot['acceleration_penalty'].values, alpha=0.8, linewidth=0.8, color='red')
            axes[1, 0].set_title('Acceleration Penalty')
        axes[1, 0].set_xlabel('Step')
        axes[1, 0].set_ylabel('Penalty')
        axes[1, 0].grid(True, alpha=0.3)

        # Distance performance
        axes[1, 1].plot(step_vals, df_plot['distance_to_target'].values, label='Distance to Target', linewidth=1.2)
        axes[1, 1].set_title('Distance to Target (Enhanced Control)')
        axes[1, 1].set_xlabel('Step')
        axes[1, 1].set_ylabel('Distance (m)')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(plot_dir / 'enhanced_control_analysis.png', dpi=150, bbox_inches='tight')
        plt.close()
        print("✓ Enhanced control analysis plot created")

        # 2. Reward Components Analysis (Enhanced)
        fig, axes = plt.subplots(2, 3, figsize=(18, 10))
        fig.suptitle('Enhanced Reward Components Analysis', fontsize=16)

        axes[0, 0].plot(step_vals, df_plot['reward'].values, alpha=0.7, linewidth=0.8)
        window_size = min(500, len(df_plot) // 20)
        if window_size > 1:
            rolling_reward = df_plot['reward'].rolling(window=window_size, center=True).mean()
            axes[0, 0].plot(step_vals, rolling_reward.values, 'red', linewidth=2, label=f'Rolling Mean ({window_size})')
            axes[0, 0].legend()
        axes[0, 0].set_title('Total Reward vs Time')
        axes[0, 0].set_xlabel('Step')
        axes[0, 0].set_ylabel('Reward')
        axes[0, 0].grid(True, alpha=0.3)

        axes[0, 1].plot(step_vals, df_plot['distance_reward'].values, alpha=0.8, linewidth=0.8, color='green')
        axes[0, 1].axhline(y=0, color='black', linestyle='--', alpha=0.3)
        axes[0, 1].set_title('Delta-Distance Reward')
        axes[0, 1].set_xlabel('Step')
        axes[0, 1].set_ylabel('Delta Reward')
        axes[0, 1].grid(True, alpha=0.3)

        axes[0, 2].plot(step_vals, df_plot['tendon_efficiency_reward'].values, alpha=0.8, linewidth=0.8, color='blue')
        axes[0, 2].set_title('Tendon Efficiency Reward')
        axes[0, 2].set_xlabel('Step')
        axes[0, 2].set_ylabel('Efficiency Reward')
        axes[0, 2].grid(True, alpha=0.3)

        axes[1, 0].plot(step_vals, df_plot['acceleration_penalty'].values, alpha=0.8, linewidth=0.8, color='red', label='Acceleration')
        if 'jerk_penalty' in df_plot.columns:
            axes[1, 0].plot(step_vals, df_plot['jerk_penalty'].values, alpha=0.8, linewidth=0.8, color='darkred', label='Jerk')
            axes[1, 0].legend()
        axes[1, 0].set_title('Smoothness Penalties')
        axes[1, 0].set_xlabel('Step')
        axes[1, 0].set_ylabel('Penalty')
        axes[1, 0].grid(True, alpha=0.3)

        # Tendon force usage over time
        axes[1, 1].plot(step_vals, df_plot['tendon_force_index'].values, label='Index', alpha=0.8, linewidth=0.8)
        axes[1, 1].plot(step_vals, df_plot['tendon_force_middle'].values, label='Middle', alpha=0.8, linewidth=0.8)
        axes[1, 1].plot(step_vals, df_plot['tendon_force_ring'].values, label='Ring', alpha=0.8, linewidth=0.8)
        axes[1, 1].plot(step_vals, df_plot['tendon_force_thumb'].values, label='Thumb', alpha=0.8, linewidth=0.8)
        axes[1, 1].set_title('Tendon Force Usage')
        axes[1, 1].set_xlabel('Step')
        axes[1, 1].set_ylabel('Normalized Force')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)

        # Success analysis
        if 'episode' in df_plot.columns:
            try:
                episode_success = df.groupby('episode')['success'].max()
                episode_indices = np.array(episode_success.index)
                success_values = np.array(episode_success.values)

                if len(episode_indices) > 1000:
                    step_ep = len(episode_indices) // 1000
                    episode_indices = episode_indices[::step_ep]
                    success_values = success_values[::step_ep]

                axes[1, 2].plot(episode_indices, success_values, 'o-', alpha=0.7, markersize=2)
                if len(episode_success) > 10:
                    window = min(50, len(episode_success) // 10)
                    rolling_success = episode_success.rolling(window=window, center=True).mean()
                    if len(episode_indices) > 1000:
                        rolling_vals = np.array(rolling_success.values)[::step_ep]
                    else:
                        rolling_vals = np.array(rolling_success.values)
                    axes[1, 2].plot(episode_indices, rolling_vals, 'red', linewidth=2,
                                   label=f'Rolling Success Rate ({window})')
                    axes[1, 2].legend()
                axes[1, 2].set_title('Success Rate per Episode')
                axes[1, 2].set_xlabel('Episode')
                axes[1, 2].set_ylabel('Success (0/1)')
                axes[1, 2].grid(True, alpha=0.3)
            except Exception as e:
                print(f"Skipping episode success plot: {e}")
                axes[1, 2].text(0.5, 0.5, 'Episode data unavailable', ha='center', va='center', transform=axes[1, 2].transAxes)

        plt.tight_layout()
        plt.savefig(plot_dir / 'enhanced_reward_analysis.png', dpi=150, bbox_inches='tight')
        plt.close()
        print("✓ Enhanced reward analysis plot created")

    except Exception as e:
        print(f"Error in plotting: {e}")
        import traceback
        traceback.print_exc()
        plt.close('all')

    print(f"All plots saved to: {plot_dir}")


def main(argv):
    """Main training function with checkpointing system"""

    # Parse command line arguments for checkpoint loading
    checkpoint_path = None
    run_number = "1"

    if "-c" in argv:
        argument = argv[argv.index("-c") + 1]
        if argument.isdigit():
            run_number = argument
        else:
            checkpoint_path = argument
    else:
        if len(argv) > 1:
            run_number = argv[1]

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    base_training_dir = Path("./SC1_Training_Runs/")
    base_training_dir.mkdir(parents=True, exist_ok=True)

    log_dir = base_training_dir / f"Run_{timestamp}_SC1_Enhanced_Checkpointing"
    log_dir.mkdir(parents=True, exist_ok=True)

    # Setup checkpoint directory
    checkpoint_dir = log_dir / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print(f"SC-1 ENHANCED WITH CHECKPOINTING - RUN #{run_number}")
    print("=" * 80)

    try:
        # Initialize WandB
        wandb.init(
            project="space-touch-sc1-enhanced",
            name=f"SC1_Enhanced_Checkpointing_run{run_number}",
            config={
                "algorithm": "PPO",
                "total_timesteps": 1000000,  # Scale to 1M timesteps
                "checkpoint_frequency": 200000,  # Save every 200K timesteps
                "learning_rate": 3e-4,
                "n_steps": 2048,
                "batch_size": 64,
                "n_epochs": 10,
                "gamma": 0.99,
                "target_position": [0.25, 0.15, 0.35],
                "max_episode_steps": "adaptive",
                "action_space": "10D (6 base + 4 tendons)",
                "observation_space": "25D",
                "tactile_type": "binary_contact",
                "reward_type": "delta_distance + enhanced_penalties",
                "control_smoothing": True,
                "filter_cutoff_hz": 15.0,
                "jerk_penalty_weight": 0.25,
                "acceleration_penalty_weight": 0.15,
                "notes": "Enhanced with checkpointing, control smoothing, and jerk penalty"
            },
            tags=["tendon-control", "checkpointing", "control-smoothing", "jerk-penalty", "ppo"]
        )

        data_logger = DataLogger(log_dir)

        print("\nCreating enhanced environment with control smoothing...")
        env = TendonAllegroReachingEnv(vis=False, control_smoothing=True, filter_cutoff=15.0)
        env.set_data_logger(data_logger)

        # Setup callbacks
        wandb_callback = WandBCallback(data_logger, log_freq=100)
        checkpoint_callback = CheckpointCallback(
            checkpoint_freq=200000,
            save_path=checkpoint_dir,
            verbose=1
        )

        print("Setting up PPO model...")

        if torch.cuda.is_available():
            device = "cuda"
            print(f"✓ GPU acceleration enabled: {torch.cuda.get_device_name(0)}")
        else:
            device = "cpu"
            print("⚠ No GPU found, using CPU")

        # Try to load existing checkpoint
        model = None
        start_timestep = 0

        if checkpoint_path:
            # Load specific checkpoint
            try:
                model = PPO.load(checkpoint_path, env=env, device=device)
                start_timestep = int(Path(checkpoint_path).stem.split('_')[1])
                print(f"✓ Loaded checkpoint: {checkpoint_path}")
                print(f"✓ Resuming from timestep: {start_timestep}")
            except Exception as e:
                print(f"⚠ Failed to load checkpoint {checkpoint_path}: {e}")
                model = None
        else:
            # Try to find latest checkpoint in directory
            model, start_timestep = load_latest_checkpoint(checkpoint_dir, PPO, env)
            if model is not None:
                model.set_env(env)
                print(f"✓ Resuming from existing checkpoint at timestep {start_timestep}")

        # Create new model if no checkpoint found
        if model is None:
            print("✓ Creating new model (no checkpoint found)")
            model = PPO(
                "MlpPolicy",
                env,
                verbose=1,
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

        print("\n" + "=" * 80)
        print("ENHANCED SC-1 TRAINING CONFIGURATION")
        print("=" * 80)
        print(f"Log Directory: {log_dir}")
        print(f"Checkpoint Directory: {checkpoint_dir}")
        print(f"WandB Project: space-touch-sc1-enhanced")
        print(f"Target Position: {env.target_pos} (STATIC)")
        print(f"Total Training Timesteps: 1,000,000")
        print(f"Checkpoint Frequency: Every 200,000 timesteps")
        print(f"Starting Timestep: {start_timestep}")
        print("Action Space: 10D (6 base movement + 4 tendon forces)")
        print("Observation Space: 25D (base + target + fingers + BINARY_TACTILE)")
        print("\nENHANCEMENTS:")
        print("  ✓ Model checkpointing system (save/resume every 200K)")
        print("  ✓ Low-pass filter for control smoothing (15 Hz cutoff)")
        print("  ✓ Enhanced jerk penalty (0.25x weight)")
        print("  ✓ Enhanced acceleration penalty (0.15x weight)")
        print("  ✓ Proper environment resetting during testing")
        print("  ✓ Scalable to 1M timesteps with automatic resuming")
        print("=" * 80)

        # Training loop with checkpointing
        TIMESTEPS_PER_ITERATION = 200000
        TOTAL_TIMESTEPS = 1000000

        current_timestep = start_timestep
        iteration = current_timestep // TIMESTEPS_PER_ITERATION

        print(f"\nStarting training from iteration {iteration + 1}")
        print(f"Current timestep: {current_timestep}")
        print(f"Target timesteps: {TOTAL_TIMESTEPS}")

        while current_timestep < TOTAL_TIMESTEPS:
            iteration += 1
            next_checkpoint = min(current_timestep + TIMESTEPS_PER_ITERATION, TOTAL_TIMESTEPS)
            timesteps_this_iteration = next_checkpoint - current_timestep

            print(f"\n" + "="*60)
            print(f"ITERATION {iteration}: Training {timesteps_this_iteration:,} timesteps")
            print(f"Progress: {current_timestep:,} -> {next_checkpoint:,} / {TOTAL_TIMESTEPS:,}")
            print(f"Completion: {(next_checkpoint/TOTAL_TIMESTEPS)*100:.1f}%")
            print("="*60)

            # Train for this iteration
            model.learn(
                total_timesteps=timesteps_this_iteration,
                callback=[wandb_callback, checkpoint_callback],
                log_interval=10,
                reset_num_timesteps=False,
                progress_bar=True
            )

            current_timestep = next_checkpoint

            # Save iteration checkpoint
            iteration_checkpoint = checkpoint_dir / f"checkpoint_{current_timestep}"
            model.save(str(iteration_checkpoint))
            print(f"✓ Iteration {iteration} complete! Checkpoint saved: {iteration_checkpoint}.zip")

            # Log progress to WandB
            wandb.log({
                "training/iteration": iteration,
                "training/completion_percent": (current_timestep / TOTAL_TIMESTEPS) * 100,
                "training/remaining_timesteps": TOTAL_TIMESTEPS - current_timestep
            }, step=current_timestep)

            if current_timestep >= TOTAL_TIMESTEPS:
                break

        # Save final model
        final_model_path = log_dir / f"sc1_enhanced_final_model_run{run_number}_{timestamp}"
        model.save(str(final_model_path))
        print(f"\n✓ Training Complete! Final model saved to {final_model_path}.zip")

        # Upload final model to WandB
        wandb.save(str(final_model_path) + ".zip")

        # Test the trained model with proper environment resetting
        print(f"\nTesting trained model for 10 episodes with proper environment resetting...")

        test_data_logger = DataLogger(log_dir / "test_data")
        env_test = TendonAllegroReachingEnv(vis=False, control_smoothing=True, filter_cutoff=15.0)
        env_test.set_data_logger(test_data_logger)

        test_results = []

        for test_episode in range(10):
            # ENSURE proper reset before each test episode
            obs = env_test.reset()
            episode_reward = 0
            episode_steps = 0

            while episode_steps < 1000:  # Max steps per test episode
                action, _ = model.predict(obs, deterministic=True)
                obs, reward, done, info = env_test.step(action)
                episode_reward += reward[0]
                episode_steps += 1

                if done[0]:
                    break

            test_data_logger.new_episode()

            distance = info[0].get('distance', float('inf'))
            success = info[0].get('success', False)
            tendon_forces = info[0].get('tendon_forces', {})
            avg_tendon = np.mean(list(tendon_forces.values())) if tendon_forces else 0
            tactile_contacts = info[0].get('tactile_contacts', [0,0,0,0])
            total_contacts = sum(tactile_contacts)

            test_results.append({
                'episode': test_episode + 1,
                'distance': distance,
                'success': success,
                'reward': episode_reward,
                'steps': episode_steps,
                'avg_tendon': avg_tendon,
                'total_contacts': total_contacts
            })

            status = "SUCCESS" if success else "FAILED"
            print(f"  Test Episode {test_episode + 1:2d}: {status} | Dist={distance:.4f}m | Reward={episode_reward:.1f} | Steps={episode_steps} | Contacts={total_contacts}")

        env_test.close()

        test_csv = test_data_logger.save_to_csv("sc1_enhanced_test_data.csv")
        print(f"\n✓ Test data saved to: {test_csv}")

        # Log test results to WandB
        successes = sum(r['success'] for r in test_results)
        avg_distance = np.mean([r['distance'] for r in test_results]) if test_results else 0
        avg_reward = np.mean([r['reward'] for r in test_results]) if test_results else 0
        avg_steps = np.mean([r['steps'] for r in test_results]) if test_results else 0
        avg_tendon = np.mean([r['avg_tendon'] for r in test_results]) if test_results else 0
        avg_contacts = np.mean([r['total_contacts'] for r in test_results]) if test_results else 0

        wandb.log({
            "test/success_rate": successes / len(test_results) if test_results else 0,
            "test/avg_distance": avg_distance,
            "test/avg_reward": avg_reward,
            "test/avg_steps": avg_steps,
            "test/avg_tendon_usage": avg_tendon,
            "test/avg_accidental_contacts": avg_contacts
        })

        print("\n" + "=" * 80)
        print("SC-1 ENHANCED TEST RESULTS SUMMARY")
        print("=" * 80)
        print(f"Success Rate: {successes}/{len(test_results)} ({successes/len(test_results)*100:.1f}%)")
        print(f"Average Final Distance: {avg_distance:.4f}m")
        print(f"Average Episode Reward: {avg_reward:.1f}")
        print(f"Average Episode Steps: {avg_steps:.1f}")
        print(f"Average Tendon Usage: {avg_tendon:.3f}")
        print(f"Average Accidental Contacts: {avg_contacts:.1f}")
        print("=" * 80)

        print(f"\nGenerating enhanced analysis plots from test data...")
        create_plots(test_csv)

        # Upload plots to WandB
        plot_dir = log_dir / "test_data" / "plots"
        if plot_dir.exists():
            for plot_file in plot_dir.glob("*.png"):
                wandb.log({f"plots/{plot_file.stem}": wandb.Image(str(plot_file))})

        wandb.finish()

        print("\n" + "=" * 80)
        print("SC-1 ENHANCED TRAINING COMPLETE!")
        print("=" * 80)
        print(f"All results saved to: {log_dir}")
        print(f"Checkpoints: {checkpoint_dir}")
        print(f"Test data: {log_dir}/test_data/")
        print(f"Analysis plots: {log_dir}/test_data/plots/")
        print(f"WandB Dashboard: https://wandb.ai/[your-username]/space-touch-sc1-enhanced")
        print("\nKEY ENHANCEMENTS:")
        print("  • Model checkpointing system (200K timestep intervals)")
        print("  • Scalable to 1M+ timesteps with automatic resuming")
        print("  • Low-pass filter control smoothing (reduces jerkiness)")
        print("  • Enhanced jerk penalty (penalizes rate of acceleration change)")
        print("  • Proper environment resetting during testing phase")
        print("  • Comprehensive analysis plots including control smoothing")
        print("=" * 80)

    except KeyboardInterrupt:
        print("\nTraining interrupted by user")
        # Save emergency checkpoint
        if 'model' in locals() and model is not None:
            emergency_path = log_dir / f"emergency_checkpoint_{int(time.time())}"
            model.save(str(emergency_path))
            print(f"Emergency checkpoint saved: {emergency_path}.zip")
        wandb.finish()

    except Exception as e:
        print(f"\nError occurred: {e}")
        import traceback
        traceback.print_exc()
        # Save emergency checkpoint
        if 'model' in locals() and model is not None:
            emergency_path = log_dir / f"emergency_checkpoint_{int(time.time())}"
            model.save(str(emergency_path))
            print(f"Emergency checkpoint saved: {emergency_path}.zip")
        wandb.finish()

    finally:
        try:
            if 'env' in locals():
                env.close()
                print("Environment closed.")
        except:
            pass


if __name__ == "__main__":
    main(sys.argv)