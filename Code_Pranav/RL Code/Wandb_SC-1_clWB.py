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

# Fix for attrdict Python 3.13 compatibility
import collections.abc
import collections
collections.Mapping = collections.abc.Mapping
collections.MutableMapping = collections.abc.MutableMapping
collections.Sequence = collections.abc.Sequence
collections.MutableSequence = collections.abc.MutableSequence


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
            'tactile_contact_finger1': [], 'tactile_contact_finger2': [],
            'tactile_contact_finger3': [], 'tactile_contact_finger4': [],
            'reward': [], 'distance_reward': [], 'tendon_efficiency_reward': [],
            'acceleration_penalty': [], 'success': [],
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
            
            if self.episode_successes:
                success_rate = np.mean(self.episode_successes[-100:]) if len(self.episode_successes) >= 100 else np.mean(self.episode_successes)
                log_dict['performance/success_rate'] = success_rate
                log_dict['performance/total_successes'] = sum(self.episode_successes)
            
            wandb.log(log_dict, step=self.num_timesteps)
            
        return True


class TendonAllegroReachingEnv(VecEnv):
    """Enhanced environment with binary tactile sensors and improved reward function"""
    
    def __init__(self, 
                 num_envs=1,
                 vis=False,
                 max_steps=500,
                 urdf_hand="/home/pralak/Space_Touch/examples/allegro_hand_description/allegro_hand_description_left_digit_fixed.urdf",
                 target_range=0.3):
        
        self.num_envs = num_envs
        self.vis = vis
        self.max_steps = max_steps
        self.sim_freq = 240
        self.urdf_hand = urdf_hand
        self.target_range = target_range
        
        self.target_pos = np.array([0.25, 0.15, 0.35])
        
        self._init_pybullet()
        
        self.hand = None
        self.target_sphere = None
        self.tendon_controller = None
        self.hand_spawned = False  # Track if hand is already spawned
        
        # Action space: 6 DOF base movement + 4 tendon forces
        action_dim = 10
        self.action_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(action_dim,), dtype=np.float32)
        
        # Observation: base_pos(3) + target_pos(3) + base_vel(3) + distance(1) + 
        #              finger_positions(12) + binary_tactile(4) + tendon_forces(4)
        obs_dim = 3 + 3 + 3 + 1 + 12 + 4 + 4  # = 30
        self.observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32)
        
        super().__init__(num_envs, self.observation_space, self.action_space)
        
        self.step_counts = np.zeros(num_envs, dtype=np.int32)
        self.episode_rewards = np.zeros(num_envs)
        self.episode_lengths = np.zeros(num_envs, dtype=np.int32)
        
        # For tracking previous actions (acceleration penalty)
        self.prev_actions = np.zeros((num_envs, action_dim), dtype=np.float32)
        self.prev_prev_actions = np.zeros((num_envs, action_dim), dtype=np.float32)
        
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
                    # Verify hand still exists
                    p.getBasePositionAndOrientation(self.hand)
                    print("⚠ Hand already exists, skipping spawn")
                    return
                except:
                    # Hand was removed, allow respawn
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
            
            # Create target sphere (visible for tactile testing)
            target_collision = p.createCollisionShape(p.GEOM_SPHERE, radius=0.02)
            target_visual = p.createVisualShape(p.GEOM_SPHERE, radius=0.02, rgbaColor=[1, 0, 0, 1])
            self.target_sphere = p.createMultiBody(
                baseMass=0.1,  # Small mass to allow interaction
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
            # Check for contact points on this link
            contact_points = p.getContactPoints(bodyA=self.hand, linkIndexA=link_idx)
            
            # Binary: 1 if any contact, 0 otherwise
            has_contact = 1.0 if len(contact_points) > 0 else 0.0
            binary_contacts.append(has_contact)
        
        # Ensure we have exactly 4 values
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
            
            distance = np.linalg.norm(base_pos - self.target_pos)
            
            finger_positions = self._get_finger_positions()
            binary_tactile = self._get_binary_tactile_feedback()
            
            tendon_forces = getattr(self, 'current_tendon_forces', np.zeros(4))
            
            obs = np.concatenate([
                base_pos,           # 3D
                self.target_pos,    # 3D  
                base_vel,           # 3D
                [distance],         # 1D
                finger_positions,   # 12D
                binary_tactile,     # 4D (binary contact per finger)
                tendon_forces       # 4D
            ])
            
            return np.expand_dims(obs.astype(np.float32), axis=0)
            
        except Exception as e:
            print(f"Error getting observation: {e}")
            return np.zeros((1, self.observation_space.shape[0]), dtype=np.float32)

    def step_wait(self):
        """Execute one step of the environment"""
        self.step_counts += 1
        self.episode_lengths += 1
        actions = self.actions[0]
        
        try:
            if self.hand is None:
                obs = self._get_observation()
                return obs, np.array([-1.0], dtype=np.float32), np.array([True]), [{"error": "Hand not loaded"}]
            
            base_actions = actions[:6]
            tendon_actions = actions[6:10]
            
            # Apply base movement
            linear_vel = base_actions[:3] * 0.3  
            angular_vel = base_actions[3:6] * 0.8  
            
            p.resetBaseVelocity(
                self.hand, 
                linearVelocity=linear_vel, 
                angularVelocity=angular_vel
            )
            
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
            
            # 1. INCREASED Distance reward (main objective)
            distance_reward = 5.0 * np.exp(-8.0 * distance)  # Increased from 1.0 to 5.0
            
            # 2. Tendon efficiency reward
            tendon_efficiency = 1.0 - 0.5 * np.mean(tendon_forces)
            tendon_efficiency_reward = 0.2 * tendon_efficiency
            
            # 3. Tactile contact penalty (penalize accidental contact)
            tactile_contact_penalty = -0.5 * np.sum(binary_tactile)
            
            # 4. Movement efficiency penalty
            linear_vel_magnitude = np.linalg.norm(linear_vel)
            angular_vel_magnitude = np.linalg.norm(angular_vel)
            movement_penalty = -0.01 * (linear_vel_magnitude + angular_vel_magnitude)
            
            # 5. NEW: Acceleration/jerk penalty (penalize rapid changes in control)
            # Compute acceleration as difference between consecutive velocities
            current_vel = actions[:6]  # First 6 actions are velocities
            prev_vel = self.prev_actions[0][:6]
            accel = current_vel - prev_vel
            
            # Penalize large accelerations (jerky behavior)
            accel_magnitude = np.linalg.norm(accel)
            acceleration_penalty = -0.1 * accel_magnitude
            
            # 6. Success bonus
            success_bonus = 10.0 if distance < 0.1 else 0.0
            
            # Combine all reward components
            total_reward = (distance_reward + tendon_efficiency_reward + tactile_contact_penalty + 
                           movement_penalty + acceleration_penalty + success_bonus)
            
            # Update action history
            self.prev_prev_actions[0] = self.prev_actions[0].copy()
            self.prev_actions[0] = actions.copy()
            
            self.episode_rewards[0] += total_reward
            
            success = distance < 0.1
            
            # Termination conditions
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
                "tactile_contact_finger1": binary_tactile[0], "tactile_contact_finger2": binary_tactile[1],
                "tactile_contact_finger3": binary_tactile[2], "tactile_contact_finger4": binary_tactile[3],
                "reward": total_reward,
                "distance_reward": distance_reward,
                "tendon_efficiency_reward": tendon_efficiency_reward,
                "acceleration_penalty": acceleration_penalty,
                "success": success,
            }
            
            if self.data_logger is not None:
                self.data_logger.log_step(self.step_data)
            
            # Info for WandB callback
            infos = [{
                "success": success,
                "distance": distance,
                "distance_reward": distance_reward,
                "tendon_efficiency_reward": tendon_efficiency_reward,
                "acceleration_penalty": acceleration_penalty,
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
        """Reset environment"""
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
            
            # Setup new simulation
            self._setup_simulation()
            
            # Reset counters and action history
            self.step_counts[dones] = 0
            self.episode_rewards[dones] = 0
            self.episode_lengths[dones] = 0
            self.prev_actions[dones] = 0
            self.prev_prev_actions[dones] = 0
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
        
        # 1. Base Position and Movement Analysis
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('Base Position and Movement Analysis', fontsize=16)
        
        axes[0, 0].plot(step_vals, df_plot['base_pos_x'].values, label='X', alpha=0.8, linewidth=0.8)
        axes[0, 0].plot(step_vals, df_plot['base_pos_y'].values, label='Y', alpha=0.8, linewidth=0.8)
        axes[0, 0].plot(step_vals, df_plot['base_pos_z'].values, label='Z', alpha=0.8, linewidth=0.8)
        axes[0, 0].axhline(y=df_plot['target_x'].iloc[0], color='r', linestyle='--', label='Target X', alpha=0.6)
        axes[0, 0].axhline(y=df_plot['target_y'].iloc[0], color='g', linestyle='--', label='Target Y', alpha=0.6)
        axes[0, 0].axhline(y=df_plot['target_z'].iloc[0], color='b', linestyle='--', label='Target Z', alpha=0.6)
        axes[0, 0].set_title('Base Position vs Time')
        axes[0, 0].set_xlabel('Step')
        axes[0, 0].set_ylabel('Position (m)')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        axes[0, 1].plot(step_vals, df_plot['base_vel_x'].values, label='Vel X', alpha=0.8, linewidth=0.8)
        axes[0, 1].plot(step_vals, df_plot['base_vel_y'].values, label='Vel Y', alpha=0.8, linewidth=0.8)
        axes[0, 1].plot(step_vals, df_plot['base_vel_z'].values, label='Vel Z', alpha=0.8, linewidth=0.8)
        axes[0, 1].set_title('Base Velocity vs Time')
        axes[0, 1].set_xlabel('Step')
        axes[0, 1].set_ylabel('Velocity (m/s)')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        axes[1, 0].plot(step_vals, df_plot['distance_to_target'].values, label='Distance to Target', linewidth=1.2)
        axes[1, 0].set_title('Distance to Target vs Time')
        axes[1, 0].set_xlabel('Step')
        axes[1, 0].set_ylabel('Distance (m)')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        
        control_magnitude = np.sqrt(df_plot['control_linear_x']**2 + df_plot['control_linear_y']**2 + df_plot['control_linear_z']**2)
        axes[1, 1].plot(step_vals, control_magnitude.values, alpha=0.7, linewidth=0.8)
        axes[1, 1].set_title('Control Input Magnitude')
        axes[1, 1].set_xlabel('Step')
        axes[1, 1].set_ylabel('Control Magnitude')
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(plot_dir / 'base_movement_analysis.png', dpi=150, bbox_inches='tight')
        plt.close()
        print("✓ Base movement analysis plot created")
        
        # 2. Tactile Sensor Analysis
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('Binary Tactile Sensor Analysis', fontsize=16)
        
        axes[0, 0].plot(step_vals, df_plot['tactile_contact_finger1'].values, label='Finger 1', alpha=0.8)
        axes[0, 0].plot(step_vals, df_plot['tactile_contact_finger2'].values, label='Finger 2', alpha=0.8)
        axes[0, 0].plot(step_vals, df_plot['tactile_contact_finger3'].values, label='Finger 3', alpha=0.8)
        axes[0, 0].plot(step_vals, df_plot['tactile_contact_finger4'].values, label='Finger 4', alpha=0.8)
        axes[0, 0].set_title('Binary Tactile Contact per Finger')
        axes[0, 0].set_xlabel('Step')
        axes[0, 0].set_ylabel('Contact (0/1)')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        total_contacts = (df_plot['tactile_contact_finger1'] + df_plot['tactile_contact_finger2'] + 
                         df_plot['tactile_contact_finger3'] + df_plot['tactile_contact_finger4'])
        axes[0, 1].plot(step_vals, total_contacts.values, linewidth=1.2)
        axes[0, 1].set_title('Total Contact Count')
        axes[0, 1].set_xlabel('Step')
        axes[0, 1].set_ylabel('Number of Fingers in Contact')
        axes[0, 1].grid(True, alpha=0.3)
        
        axes[1, 0].scatter(df_plot['distance_to_target'].values, total_contacts.values, alpha=0.5, s=1)
        axes[1, 0].set_title('Contact vs Distance to Target')
        axes[1, 0].set_xlabel('Distance to Target (m)')
        axes[1, 0].set_ylabel('Number of Fingers in Contact')
        axes[1, 0].grid(True, alpha=0.3)
        
        if 'episode' in df.columns:
            try:
                episode_contact = df.groupby('episode')[['tactile_contact_finger1', 'tactile_contact_finger2', 
                                                          'tactile_contact_finger3', 'tactile_contact_finger4']].sum().sum(axis=1)
                episode_indices = np.array(episode_contact.index)
                contact_values = np.array(episode_contact.values)
                
                if len(episode_indices) > 1000:
                    step_ep = len(episode_indices) // 1000
                    episode_indices = episode_indices[::step_ep]
                    contact_values = contact_values[::step_ep]
                
                axes[1, 1].plot(episode_indices, contact_values, 'o-', alpha=0.7, markersize=2)
                axes[1, 1].set_title('Total Contacts per Episode')
                axes[1, 1].set_xlabel('Episode')
                axes[1, 1].set_ylabel('Total Contact Events')
                axes[1, 1].grid(True, alpha=0.3)
            except Exception as e:
                print(f"Skipping episode contact plot: {e}")
                axes[1, 1].text(0.5, 0.5, 'Episode data unavailable', ha='center', va='center', transform=axes[1, 1].transAxes)
        
        plt.tight_layout()
        plt.savefig(plot_dir / 'tactile_analysis.png', dpi=150, bbox_inches='tight')
        plt.close()
        print("✓ Tactile analysis plot created")
        
        # 3. Reward Components Analysis
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('Reward Components Analysis', fontsize=16)
        
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
        
        axes[0, 1].plot(step_vals, df_plot['distance_reward'].values, alpha=0.8, linewidth=0.8)
        axes[0, 1].set_title('Distance Reward Component (5x weighted)')
        axes[0, 1].set_xlabel('Step')
        axes[0, 1].set_ylabel('Distance Reward')
        axes[0, 1].grid(True, alpha=0.3)
        
        axes[1, 0].plot(step_vals, df_plot['acceleration_penalty'].values, alpha=0.8, linewidth=0.8, color='red')
        axes[1, 0].set_title('Acceleration Penalty (Anti-Jerk)')
        axes[1, 0].set_xlabel('Step')
        axes[1, 0].set_ylabel('Penalty')
        axes[1, 0].grid(True, alpha=0.3)
        
        if 'episode' in df_plot.columns:
            try:
                episode_success = df.groupby('episode')['success'].max()
                episode_indices = np.array(episode_success.index)
                success_values = np.array(episode_success.values)
                
                if len(episode_indices) > 1000:
                    step_ep = len(episode_indices) // 1000
                    episode_indices = episode_indices[::step_ep]
                    success_values = success_values[::step_ep]
                
                axes[1, 1].plot(episode_indices, success_values, 'o-', alpha=0.7, markersize=2)
                if len(episode_success) > 10:
                    window = min(50, len(episode_success) // 10)
                    rolling_success = episode_success.rolling(window=window, center=True).mean()
                    if len(episode_indices) > 1000:
                        rolling_vals = np.array(rolling_success.values)[::step_ep]
                    else:
                        rolling_vals = np.array(rolling_success.values)
                    axes[1, 1].plot(episode_indices, rolling_vals, 'red', linewidth=2, 
                                   label=f'Rolling Success Rate ({window})')
                    axes[1, 1].legend()
                axes[1, 1].set_title('Success Rate per Episode')
                axes[1, 1].set_xlabel('Episode')
                axes[1, 1].set_ylabel('Success (0/1)')
                axes[1, 1].grid(True, alpha=0.3)
            except Exception as e:
                print(f"Skipping episode success plot: {e}")
                axes[1, 1].text(0.5, 0.5, 'Episode data unavailable', ha='center', va='center', transform=axes[1, 1].transAxes)
        
        plt.tight_layout()
        plt.savefig(plot_dir / 'reward_analysis.png', dpi=150, bbox_inches='tight')
        plt.close()
        print("✓ Reward analysis plot created")
        
    except Exception as e:
        print(f"Error in plotting: {e}")
        import traceback
        traceback.print_exc()
        plt.close('all')
    
    print(f"All plots saved to: {plot_dir}")
    
    # Print summary statistics
    print("\n=== SC-1 TRAINING SUMMARY ===")
    print(f"Total steps: {len(df)}")
    if 'episode' in df.columns and len(df) > 0:
        try:
            max_episode = df['episode'].max()
            if pd.notna(max_episode):
                print(f"Total episodes: {int(max_episode) + 1}")
                print(f"Average episode length: {df.groupby('episode').size().mean():.1f} steps")
        except (ValueError, TypeError) as e:
            print(f"Could not calculate episode statistics: {e}")
    if len(df) > 0:
        print(f"Final distance to target: {df['distance_to_target'].iloc[-1]:.4f} m")
        print(f"Average reward: {df['reward'].mean():.4f}")
        print(f"Average tendon usage: {(df['tendon_force_index'].mean() + df['tendon_force_middle'].mean() + df['tendon_force_ring'].mean() + df['tendon_force_thumb'].mean()) / 4:.4f}")
        total_contact_rate = (df['tactile_contact_finger1'].mean() + df['tactile_contact_finger2'].mean() + 
                             df['tactile_contact_finger3'].mean() + df['tactile_contact_finger4'].mean())
        print(f"Average accidental contacts per step: {total_contact_rate:.3f}")
        print(f"Success rate: {df['success'].mean()*100:.1f}%")


# --- Main Training and Analysis Script ---
if __name__ == "__main__":
    import sys
    
    run_number = sys.argv[1] if len(sys.argv) > 1 else "1"
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    base_training_dir = Path("./SC1_Training_Runs/")
    base_training_dir.mkdir(parents=True, exist_ok=True)
    
    log_dir = base_training_dir / f"Run_{timestamp}_SC1_WandB_Tactile"
    log_dir.mkdir(parents=True, exist_ok=True)
    
    print("=" * 80)
    print(f"SC-1 WANDB + BINARY TACTILE TRAINING - RUN #{run_number}")
    print("=" * 80)
    
    try:
        # Initialize WandB
        wandb.init(
            project="space-touch-sc1",
            name=f"SC1_TendonControl_BinaryTactile_run{run_number}",
            config={
                "algorithm": "PPO",
                "total_timesteps": 10000000,
                "learning_rate": 3e-4,
                "n_steps": 2048,
                "batch_size": 64,
                "n_epochs": 10,
                "gamma": 0.99,
                "target_position": [0.25, 0.15, 0.35],
                "max_episode_steps": 500,
                "action_space": "10D (6 base + 4 tendons)",
                "observation_space": "30D (with binary tactile)",
                "tactile_type": "binary_contact",
                "reward_distance_weight": 5.0,
                "reward_accel_penalty_weight": 0.1,
                "notes": "Binary tactile, anti-jerk penalty, 5x distance reward, target sphere enabled"
            },
            tags=["tendon-control", "binary-tactile", "anti-jerk", "ppo"]
        )
        
        data_logger = DataLogger(log_dir)
        
        print("\nCreating tendon-controlled reaching environment with binary tactile...")
        env = TendonAllegroReachingEnv(vis=False)
        
        callback = WandBCallback(data_logger, log_freq=100)
        
        print("Creating PPO model with tendon control...")
        
        if torch.cuda.is_available():
            device = "cuda"
            print(f"✓ GPU acceleration enabled: {torch.cuda.get_device_name(0)}")
            print(f"  GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
        else:
            device = "cpu"
            print("⚠ No GPU found, using CPU")
        
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
        print("SC-1 TRAINING CONFIGURATION")
        print("=" * 80)
        print(f"Log Directory: {log_dir}")
        print(f"WandB Project: space-touch-sc1")
        print(f"Target Position: {env.target_pos} (STATIC)")
        print(f"Max Steps per Episode: {env.max_steps}")
        print(f"Total Training Timesteps: 100,000")
        print("Action Space: 10D (6 base movement + 4 tendon forces)")
        print("Observation Space: 30D (base + target + fingers + BINARY_TACTILE + tendons)")
        print("\nNEW FEATURES:")
        print("  ✓ Binary tactile sensors (0/1 per fingertip)")
        print("  ✓ Target sphere spawned for contact testing")
        print("  ✓ Multi-spawn prevention check")
        print("  ✓ Acceleration/jerk penalty (anti-jerky control)")
        print("  ✓ 5x increased distance reward")
        print("  ✓ Smoothness penalty removed")
        print("\nReward Components:")
        print("  - Distance reward (5x weight, exponential)")
        print("  - Tendon efficiency reward") 
        print("  - Tactile contact PENALTY (avoid accidental touch)")
        print("  - Movement efficiency penalty")
        print("  - Acceleration penalty (NEW - anti-jerk)")
        print("  - Success bonus")
        print("=" * 80)
        
        print(f"\nView live training at: https://wandb.ai/pranavlakshman79-lossfunk/space-touch-sc1")
        print("\n" + "=" * 80 + "\n")
        
        print("Starting SC-1 training with WandB logging...")
        model.learn(
            total_timesteps=600000,
            callback=callback,
            log_interval=10,
            reset_num_timesteps=True,
            progress_bar=True
        )
        
        model_path = log_dir / f"sc1_wandb_model_run{run_number}_{timestamp}"
        model.save(str(model_path))
        print(f"\n✓ Training Complete! Model saved to {model_path}.zip")
        
        # Upload model to WandB
        wandb.save(str(model_path) + ".zip")
        
        # Test the trained model
        print(f"\nTesting trained model for 10 episodes...")
        
        test_data_logger = DataLogger(log_dir / "test_data")
        env_test = TendonAllegroReachingEnv(vis=False)
        env_test.set_data_logger(test_data_logger)
        
        obs = env_test.reset()
        test_results = []
        test_episode = 0
        test_steps = 0
        
        while test_episode < 10 and test_steps < 5000:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, info = env_test.step(action)
            test_steps += 1
            
            if done[0]:
                test_episode += 1
                test_data_logger.new_episode()
                
                distance = info[0].get('distance', float('inf'))
                success = info[0].get('success', False)
                episode_reward = info[0].get('episode', {}).get('r', 0)
                tendon_forces = info[0].get('tendon_forces', {})
                avg_tendon = np.mean(list(tendon_forces.values())) if tendon_forces else 0
                tactile_contacts = info[0].get('tactile_contacts', [0,0,0,0])
                total_contacts = sum(tactile_contacts)
                
                test_results.append({
                    'episode': test_episode, 
                    'distance': distance, 
                    'success': success,
                    'reward': episode_reward,
                    'avg_tendon': avg_tendon,
                    'total_contacts': total_contacts
                })
                status = "SUCCESS" if success else "FAILED"
                print(f"  Test Episode {test_episode:2d}: {status} | Dist={distance:.4f}m | Reward={episode_reward:.1f} | Contacts={total_contacts}")
                obs = env_test.reset()
        
        env_test.close()
        
        test_csv = test_data_logger.save_to_csv("sc1_test_data.csv")
        print(f"\n✓ Test data saved to: {test_csv}")
        
        # Log test results to WandB
        successes = sum(r['success'] for r in test_results)
        avg_distance = np.mean([r['distance'] for r in test_results]) if test_results else 0
        avg_reward = np.mean([r['reward'] for r in test_results]) if test_results else 0
        avg_tendon = np.mean([r['avg_tendon'] for r in test_results]) if test_results else 0
        avg_contacts = np.mean([r['total_contacts'] for r in test_results]) if test_results else 0
        
        wandb.log({
            "test/success_rate": successes / len(test_results) if test_results else 0,
            "test/avg_distance": avg_distance,
            "test/avg_reward": avg_reward,
            "test/avg_tendon_usage": avg_tendon,
            "test/avg_accidental_contacts": avg_contacts
        })
        
        print("\n" + "=" * 80)
        print("SC-1 TEST RESULTS SUMMARY")
        print("=" * 80)
        print(f"Success Rate: {successes}/{len(test_results)} ({successes/len(test_results)*100:.1f}%)")
        print(f"Average Final Distance: {avg_distance:.4f}m")
        print(f"Average Episode Reward: {avg_reward:.1f}")
        print(f"Average Tendon Usage: {avg_tendon:.3f}")
        print(f"Average Accidental Contacts: {avg_contacts:.1f}")
        print("=" * 80)
        
        print(f"\nGenerating analysis plots from test data...")
        create_plots(test_csv)
        
        # Upload plots to WandB
        plot_dir = log_dir / "test_data" / "plots"
        if plot_dir.exists():
            for plot_file in plot_dir.glob("*.png"):
                wandb.log({f"plots/{plot_file.stem}": wandb.Image(str(plot_file))})
        
        wandb.finish()
        
        print("\n" + "=" * 80)
        print("SC-1 WANDB TRAINING COMPLETE!")
        print("=" * 80)
        print(f"All results saved to: {log_dir}")
        print(f"Test data: {log_dir}/test_data/")
        print(f"Analysis plots: {log_dir}/test_data/plots/")
        print(f"WandB Dashboard: https://wandb.ai/pranavlakshman79-lossfunk/space-touch-sc1")
        print("=" * 80)
        
    except KeyboardInterrupt:
        print("\nTraining interrupted by user")
        wandb.finish()
        
    except Exception as e:
        print(f"\nError occurred: {e}")
        import traceback
        traceback.print_exc()
        wandb.finish()
        
    finally:
        try:
            if 'env' in locals():
                env.close()
                print("Environment closed.")
        except:
            pass