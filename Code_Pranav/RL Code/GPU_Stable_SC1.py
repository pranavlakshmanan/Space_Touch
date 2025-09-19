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
from stable_baselines3.common.logger import configure
import os
import pandas as pd
from datetime import datetime
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend for headless operation
import matplotlib.pyplot as plt
from pathlib import Path
import torch.nn.functional as F
import tacto

# GPU STABILITY CONFIGURATION
import torch
if torch.cuda.is_available():
    # Limit GPU memory to prevent crashes with RTX 5050
    torch.cuda.set_per_process_memory_fraction(0.1)  # Use only 60% of GPU memory
    torch.cuda.empty_cache()  # Clear any cached memory
    print(f"GPU stability mode enabled - using 60% of {torch.cuda.get_device_name(0)}")
    print(f"Available GPU memory: {torch.cuda.get_device_properties(0).total_memory * 0.6 / 1e9:.1f} GB")
else:
    print("Running on CPU - no GPU acceleration")

# Fix for attrdict Python 3.13 compatibility
import collections.abc
import collections
collections.Mapping = collections.abc.Mapping
collections.MutableMapping = collections.abc.MutableMapping
collections.Sequence = collections.abc.Sequence
collections.MutableSequence = collections.abc.MutableSequence


class GPUMonitor:
    """Monitor GPU usage to prevent crashes"""
    
    def __init__(self, max_memory_gb=4.8):  # 60% of 8GB RTX 5050
        self.max_memory_gb = max_memory_gb
        self.warning_threshold = max_memory_gb * 0.8
        
    def check_gpu_status(self):
        if not torch.cuda.is_available():
            return True, 0.0
            
        current_memory = torch.cuda.memory_allocated() / 1e9
        
        if current_memory > self.max_memory_gb:
            print(f"CRITICAL: GPU memory exceeded limit! {current_memory:.1f}GB > {self.max_memory_gb}GB")
            torch.cuda.empty_cache()
            return False, current_memory
        elif current_memory > self.warning_threshold:
            print(f"WARNING: High GPU memory usage: {current_memory:.1f}GB")
            
        return True, current_memory


class DataLogger:
    """Handles data logging for analysis and plotting"""
    
    def __init__(self, log_dir):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize data storage
        self.data = {
            'timestamp': [],
            'step': [],
            'episode': [],
            
            # Base position and velocity
            'base_pos_x': [], 'base_pos_y': [], 'base_pos_z': [],
            'base_vel_x': [], 'base_vel_y': [], 'base_vel_z': [],
            'base_ang_vel_x': [], 'base_ang_vel_y': [], 'base_ang_vel_z': [],
            
            # End effector positions (for multiple fingertips)
            'ee1_pos_x': [], 'ee1_pos_y': [], 'ee1_pos_z': [],
            'ee2_pos_x': [], 'ee2_pos_y': [], 'ee2_pos_z': [],
            'ee3_pos_x': [], 'ee3_pos_y': [], 'ee3_pos_z': [],
            'ee4_pos_x': [], 'ee4_pos_y': [], 'ee4_pos_z': [],
            
            # Target and distances
            'target_x': [], 'target_y': [], 'target_z': [],
            'distance_to_target': [],
            'ee_target_distances': [],  # Average distance from EEs to target
            
            # Tendon forces for each finger
            'tendon_force_index': [],
            'tendon_force_middle': [],
            'tendon_force_ring': [],
            'tendon_force_thumb': [],
            
            # Control inputs (6 DOF base movement + 4 tendon forces)
            'control_linear_x': [], 'control_linear_y': [], 'control_linear_z': [],
            'control_angular_x': [], 'control_angular_y': [], 'control_angular_z': [],
            
            # Tactile sensor data
            'tactile_deformation_spans': [],
            'tactile_contact_detected': [],
            
            # Rewards and success metrics
            'reward': [],
            'distance_reward': [],
            'tendon_efficiency_reward': [],
            'success': [],
            
            # GPU monitoring
            'gpu_memory_used': [],
        }
        
        self.current_episode = 0
        self.global_step = 0
        
    def log_step(self, data_dict):
        """Log a single step of data"""
        self.global_step += 1
        
        for key, value in data_dict.items():
            if key in self.data:
                self.data[key].append(value)
        
        # Always append timestamp and step
        self.data['timestamp'].append(time.time())
        self.data['step'].append(self.global_step)
        self.data['episode'].append(self.current_episode)
        
        # Log GPU usage
        if torch.cuda.is_available():
            gpu_memory = torch.cuda.memory_allocated() / 1e9
            self.data['gpu_memory_used'].append(gpu_memory)
        else:
            self.data['gpu_memory_used'].append(0.0)
    
    def new_episode(self):
        """Increment episode counter"""
        self.current_episode += 1
    
    def save_to_csv(self, filename="sc1_training_data.csv"):
        """Save all logged data to CSV"""
        # Ensure all arrays have the same length
        max_len = max(len(arr) for arr in self.data.values() if arr)
        
        for key, arr in self.data.items():
            if len(arr) < max_len:
                # Pad with last value or NaN
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
        
        # Reduced tendon control parameters for stability
        self.TENDON_FORCE_GAIN = 10.0   # Reduced from 15.0
        self.TENDON_DAMPING = 1.2       # Increased for more stability
        self.MAX_TENDON_FORCE = 40.0    # Reduced from 60.0
        
        # Define finger chains (base to tip joint indices)
        self.FINGER_CHAINS = {
            "index": ["joint_8.0", "joint_9.0", "joint_10.0", "joint_11.0"],
            "middle": ["joint_4.0", "joint_5.0", "joint_6.0", "joint_7.0"],
            "ring": ["joint_0.0", "joint_1.0", "joint_2.0", "joint_3.0"],
            "thumb": ["joint_12.0", "joint_13.0", "joint_14.0", "joint_15.0"]
        }
        
        # Reference axis points for each finger (parallel to finger, within hand)
        self.FINGER_REFERENCE_AXES = {
            "index": {"start": np.array([0.0, -0.02, 0.0]), "end": np.array([0.0, -0.02, 0.08])},
            "middle": {"start": np.array([0.0, -0.01, 0.0]), "end": np.array([0.0, -0.01, 0.08])},
            "ring": {"start": np.array([0.0, 0.01, 0.0]), "end": np.array([0.0, 0.01, 0.08])},
            "thumb": {"start": np.array([-0.02, 0.0, 0.0]), "end": np.array([-0.02, 0.0, 0.06])}
        }
        
        # Create mapping from joint names to indices
        self.name_to_idx = {name: idx for name, idx in zip(joint_names, joint_indices)}
        
        # Organize joints by finger
        self.finger_joints = {}
        for finger, chain in self.FINGER_CHAINS.items():
            self.finger_joints[finger] = []
            for joint_name in chain:
                if joint_name in self.name_to_idx:
                    self.finger_joints[finger].append(self.name_to_idx[joint_name])
        
        # Pre-compute reference axis data and moment arms for each joint
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
                # Get joint position in hand frame (approximate)
                joint_info = p.getJointInfo(self.hand_id, joint_idx)
                joint_pos = np.array(joint_info[14])  # joint frame position relative to parent
                
                # For simplicity, we'll use a fixed moment arm based on joint position
                moment_arm = self._compute_moment_arm_to_axis(joint_pos, axis_start, axis_direction)
                self.joint_moment_arms[finger].append(moment_arm)
    
    def _compute_moment_arm_to_axis(self, point, axis_start, axis_direction):
        """Compute perpendicular distance from point to line (moment arm)"""
        # Vector from axis start to point
        point_vec = point - axis_start
        
        # Project point onto axis
        projection_length = np.dot(point_vec, axis_direction)
        projection_point = axis_start + projection_length * axis_direction
        
        # Perpendicular distance (moment arm)
        moment_arm_vec = point - projection_point
        moment_arm = np.linalg.norm(moment_arm_vec)
        
        # Use a minimum moment arm to avoid singularities
        return max(moment_arm, 0.005)  # 5mm minimum
    
    def compute_tendon_torques(self, tendon_forces):
        """
        Compute torque commands for each joint based on tendon forces.
        All joints in a finger experience the same tendon force, creating consistent torques.
        
        Args:
            tendon_forces: dict of finger -> normalized force (0 to 1)
        
        Returns:
            torques: array of torque commands for all joints
        """
        torques = np.zeros(len(self.joint_indices))
        
        for finger, normalized_force in tendon_forces.items():
            if finger not in self.finger_joints or finger not in self.joint_moment_arms:
                continue
            
            # Convert normalized force (0-1) to actual tendon force
            actual_force = normalized_force * self.MAX_TENDON_FORCE
            
            joints = self.finger_joints[finger]
            moment_arms = self.joint_moment_arms[finger]
            
            for i, joint_idx in enumerate(joints):
                if i >= len(moment_arms):
                    continue
                
                # Same force for all joints in finger, different moment arms
                moment_arm = moment_arms[i]
                
                # Torque = Force × Moment Arm
                torque = actual_force * moment_arm * self.TENDON_FORCE_GAIN
                
                # Get current joint velocity for damping
                joint_state = p.getJointState(self.hand_id, joint_idx)
                current_velocity = joint_state[1]
                
                # Add damping to prevent oscillations
                damping_torque = -self.TENDON_DAMPING * current_velocity
                
                # Final torque with damping
                final_torque = torque + damping_torque
                
                # Find index in joint list
                idx_in_list = self.joint_indices.index(joint_idx)
                torques[idx_in_list] = final_torque
        
        return torques


class EnhancedTensorboardCallback(BaseCallback):
    """Enhanced callback for comprehensive TensorBoard logging with GPU monitoring"""
    
    def __init__(self, data_logger, gpu_monitor, log_freq=200, verbose=0):  # Increased log_freq
        super(EnhancedTensorboardCallback, self).__init__(verbose)
        self.data_logger = data_logger
        self.gpu_monitor = gpu_monitor
        self.log_freq = log_freq
        self.episode_rewards = []
        self.episode_lengths = []
        self.episode_successes = []
        
        # Buffers for averaging metrics
        self.recent_distances = []
        self.recent_tendon_forces = []
        self.recent_tactile_spans = []
        
    def _on_step(self) -> bool:
        # GPU monitoring - check every step
        gpu_ok, gpu_memory = self.gpu_monitor.check_gpu_status()
        if not gpu_ok:
            print("STOPPING TRAINING - GPU memory limit exceeded!")
            return False  # Stop training
            
        # Log custom metrics from info
        if self.locals.get('infos'):
            for info in self.locals['infos']:
                # Episode completion logging
                if 'episode' in info:
                    ep_reward = info['episode']['r']
                    ep_length = info['episode']['l']
                    
                    self.episode_rewards.append(ep_reward)
                    self.episode_lengths.append(ep_length)
                    self.data_logger.new_episode()
                    
                    # Log episode metrics to TensorBoard
                    self.logger.record('episode/reward', ep_reward)
                    self.logger.record('episode/length', ep_length)
                    self.logger.record('gpu/memory_gb', gpu_memory)
                    
                    if len(self.episode_rewards) >= 100:
                        self.logger.record('episode/reward_mean_100', np.mean(self.episode_rewards[-100:]))
                        self.logger.record('episode/length_mean_100', np.mean(self.episode_lengths[-100:]))
                
                # Step-wise metrics logging
                if 'distance' in info:
                    self.recent_distances.append(info['distance'])
                
                if 'success' in info:
                    if 'episode' in info:  # Only track at episode end
                        self.episode_successes.append(float(info['success']))
                
                # Tendon force logging
                if 'tendon_forces' in info:
                    avg_tendon_force = np.mean(list(info['tendon_forces'].values()))
                    self.recent_tendon_forces.append(avg_tendon_force)
                
                # Tactile sensor logging
                if 'tactile_spans' in info:
                    avg_tactile_span = np.mean(info['tactile_spans'])
                    self.recent_tactile_spans.append(avg_tactile_span)
                
                # Log step data to custom logger
                if hasattr(info, 'keys'):
                    step_data = {k: v for k, v in info.items() 
                               if k in self.data_logger.data.keys()}
                    if step_data:
                        self.data_logger.log_step(step_data)
        
        # Log aggregated metrics to TensorBoard at regular intervals
        if self.num_timesteps % self.log_freq == 0:
            # Distance metrics
            if self.recent_distances:
                self.logger.record('metrics/distance_to_target_mean', np.mean(self.recent_distances))
                self.logger.record('metrics/distance_to_target_min', np.min(self.recent_distances))
                self.recent_distances = []
            
            # Tendon force metrics
            if self.recent_tendon_forces:
                self.logger.record('tendon/average_force', np.mean(self.recent_tendon_forces))
                self.logger.record('tendon/max_force', np.max(self.recent_tendon_forces))
                self.recent_tendon_forces = []
            
            # Tactile sensor metrics
            if self.recent_tactile_spans:
                self.logger.record('tactile/average_deformation', np.mean(self.recent_tactile_spans))
                self.logger.record('tactile/max_deformation', np.max(self.recent_tactile_spans))
                self.recent_tactile_spans = []
            
            # Success rate
            if self.episode_successes:
                success_rate = np.mean(self.episode_successes[-50:]) if len(self.episode_successes) >= 50 else np.mean(self.episode_successes)
                self.logger.record('performance/success_rate_50', success_rate)
                self.logger.record('performance/total_successes', sum(self.episode_successes))
            
            # Training progress and GPU monitoring
            self.logger.record('training/timesteps', self.num_timesteps)
            self.logger.record('training/episodes', len(self.episode_rewards))
            self.logger.record('gpu/memory_current_gb', gpu_memory)
            
            # Clear GPU cache periodically
            if self.num_timesteps % (self.log_freq * 5) == 0:  # Every 1000 steps
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    
        return True


class TendonAllegroReachingEnv(VecEnv):
    """
    Enhanced environment with tendon-based control for finger movements and tactile feedback.
    Combines base movement for reaching with tendon control for grasping.
    GPU-optimized version with reduced computational load.
    """
    
    def __init__(self, 
                 num_envs=1,
                 vis=False,
                 max_steps=300,  # Reduced from 500
                 urdf_hand="/home/pralak/Space_touch/examples/allegro_hand_description/allegro_hand_description_left_digit_fixed.urdf",
                 target_range=0.3):
        
        self.num_envs = num_envs
        self.vis = vis
        self.max_steps = max_steps
        self.sim_freq = 120  # Reduced from 240 for GPU stability
        self.urdf_hand = urdf_hand
        self.target_range = target_range
        
        # Static target position for consistent training
        self.target_pos = np.array([0.25, 0.15, 0.35])  # Fixed static target
        
        # Initialize PyBullet in DIRECT mode for headless operation
        self._init_pybullet()
        
        self.hand = None
        self.target_sphere = None
        self.tendon_controller = None
        self.tactile_sensor = None
        
        # Action space: 6 DOF base movement + 4 tendon forces (index, middle, ring, thumb)
        action_dim = 10
        self.action_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(action_dim,), dtype=np.float32)
        
        # Enhanced observation: base pose + target pose + finger positions + tactile feedback + tendon forces
        obs_dim = 3 + 3 + 3 + 1 + 12 + 4 + 4  # base_pos + target_pos + base_vel + distance + finger_positions + tactile + tendon_forces
        self.observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32)
        
        super().__init__(num_envs, self.observation_space, self.action_space)
        
        self.step_counts = np.zeros(num_envs, dtype=np.int32)
        self.episode_rewards = np.zeros(num_envs)
        self.episode_lengths = np.zeros(num_envs, dtype=np.int32)
        
        # For tracking previous actions (smoothness penalty)
        self.prev_actions = np.zeros((num_envs, action_dim), dtype=np.float32)
        
        # Data logging
        self.data_logger = None
        
        # Initialize environment
        self.reset()

    def _init_pybullet(self):
        """Initialize PyBullet connection - always DIRECT for headless"""
        self.client_id = p.connect(p.DIRECT)  # Always use DIRECT mode
        
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.setGravity(0, 0, -9.81)  # Normal gravity
        p.setRealTimeSimulation(0)
        p.setTimeStep(1.0/self.sim_freq)
        
        # Load ground plane
        p.loadURDF("plane.urdf", [0, 0, 0], useFixedBase=True)

    def set_data_logger(self, data_logger):
        """Set the data logger for this environment"""
        self.data_logger = data_logger

    def _setup_simulation(self):
        """Setup the simulation environment"""
        try:
            # Load hand
            if os.path.exists(self.urdf_hand):
                self.hand = p.loadURDF(
                    self.urdf_hand, 
                    basePosition=[0, 0, 0.2], 
                    baseOrientation=p.getQuaternionFromEuler([0, 0, 0]),
                    useFixedBase=False
                )
                if not hasattr(self, '_hand_loaded_printed'):
                    print(f"Hand loaded from: {self.urdf_hand}")
                    self._hand_loaded_printed = True
            else:
                print(f"Hand URDF not found: {self.urdf_hand}")
                # Create simple box hand
                hand_collision = p.createCollisionShape(p.GEOM_BOX, halfExtents=[0.1, 0.05, 0.02])
                hand_visual = p.createVisualShape(p.GEOM_BOX, halfExtents=[0.1, 0.05, 0.02], rgbaColor=[0.8, 0.6, 0.4, 1])
                self.hand = p.createMultiBody(
                    baseMass=1.0, 
                    baseCollisionShapeIndex=hand_collision, 
                    baseVisualShapeIndex=hand_visual, 
                    basePosition=[0, 0, 0.2]
                )
            
            # Create target sphere
            target_collision = p.createCollisionShape(p.GEOM_SPHERE, radius=0.02)
            target_visual = p.createVisualShape(p.GEOM_SPHERE, radius=0.02, rgbaColor=[1, 0, 0, 1])
            self.target_sphere = p.createMultiBody(
                baseMass=0,  # Static target
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
            
            # Initialize tactile sensor with simplified setup for stability
            try:
                # Disable tactile sensor for GPU stability - can be re-enabled for advanced training
                self.tactile_sensor = None
                if not hasattr(self, '_tactile_disabled_printed'):
                    print("Tactile sensor disabled for GPU stability")
                    self._tactile_disabled_printed = True
                            
            except Exception as e:
                if not hasattr(self, '_tactile_error_printed'):
                    print(f"Tactile sensor disabled due to error: {e}")
                    self._tactile_error_printed = True
                self.tactile_sensor = None
            
            # Reduced simulation settle time
            for _ in range(20):  # Reduced from 50
                p.stepSimulation()
                
        except Exception as e:
            print(f"Error setting up simulation: {e}")
            raise

    def _get_finger_positions(self):
        """Get positions of all fingertips"""
        finger_positions = []
        
        if self.hand is None:
            return np.zeros(12)  # 4 fingers * 3 coordinates
            
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
                # Use base position as fallback
                base_pos, _ = p.getBasePositionAndOrientation(self.hand)
                finger_positions.extend(base_pos)
        
        return np.array(finger_positions)

    def _get_tactile_feedback(self):
        """Get tactile sensor feedback - simplified for stability"""
        # Return zeros since tactile is disabled for GPU stability
        return np.zeros(4)  # 4 sensors

    def _get_observation(self):
        """Get current observation"""
        try:
            if self.hand is None:
                # Return default observation if hand not loaded
                obs = np.zeros(self.observation_space.shape[0])
                return np.expand_dims(obs.astype(np.float32), axis=0)
            
            # Get base position and velocity
            base_pos, _ = p.getBasePositionAndOrientation(self.hand)
            base_vel, _ = p.getBaseVelocity(self.hand)
            
            base_pos = np.array(base_pos)
            base_vel = np.array(base_vel)
            
            # Calculate distance to target
            distance = np.linalg.norm(base_pos - self.target_pos)
            
            # Get finger positions
            finger_positions = self._get_finger_positions()
            
            # Get tactile feedback (simplified)
            tactile_feedback = self._get_tactile_feedback()
            
            # Get current tendon forces (from previous action)
            tendon_forces = getattr(self, 'current_tendon_forces', np.zeros(4))
            
            # Combine into observation
            obs = np.concatenate([
                base_pos,           # 3D
                self.target_pos,    # 3D  
                base_vel,           # 3D
                [distance],         # 1D
                finger_positions,   # 12D (4 fingers * 3 coordinates)
                tactile_feedback,   # 4D
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
                # Return default values if hand not loaded
                obs = self._get_observation()
                return obs, np.array([-1.0], dtype=np.float32), np.array([True]), [{"error": "Hand not loaded"}]
            
            # Split actions: first 6 for base movement, last 4 for tendon forces
            base_actions = actions[:6]
            tendon_actions = actions[6:10]
            
            # Apply base movement (further reduced for stability)
            linear_vel = base_actions[:3] * 0.2  # Reduced from 0.3
            angular_vel = base_actions[3:6] * 0.5  # Reduced from 0.8
            
            p.resetBaseVelocity(
                self.hand, 
                linearVelocity=linear_vel, 
                angularVelocity=angular_vel
            )
            
            # Apply tendon forces
            # Convert tendon actions from [-1, 1] to [0, 1] for tendon forces
            tendon_forces = (tendon_actions + 1.0) / 2.0
            self.current_tendon_forces = tendon_forces
            
            tendon_force_dict = {
                "index": tendon_forces[0],
                "middle": tendon_forces[1], 
                "ring": tendon_forces[2],
                "thumb": tendon_forces[3]
            }
            
            # Get torque commands from tendon controller
            torques = self.tendon_controller.compute_tendon_torques(tendon_force_dict)
            
            # Apply torque control to joints
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
            
            # Get finger positions and tactile feedback
            finger_positions = self._get_finger_positions()
            tactile_spans = self._get_tactile_feedback()
            
            # 1. Distance reward (main objective) - less aggressive
            distance_reward = np.exp(-5.0 * distance)  # Reduced from -8.0
            
            # 2. Tendon efficiency reward (encourage controlled use of tendons)
            tendon_efficiency = 1.0 - 0.3 * np.mean(tendon_forces)  # Reduced penalty
            tendon_efficiency_reward = 0.15 * tendon_efficiency  # Reduced from 0.2
            
            # 3. Tactile contact reward (simplified since tactile is disabled)
            tactile_contact_reward = 0.0  # Disabled for stability
            
            # 4. Movement efficiency penalty (reduced)
            linear_vel_magnitude = np.linalg.norm(linear_vel)
            angular_vel_magnitude = np.linalg.norm(angular_vel)
            movement_penalty = -0.005 * (linear_vel_magnitude + angular_vel_magnitude)  # Reduced
            
            # 5. Action smoothness penalty (reduced)
            action_diff = np.linalg.norm(actions - self.prev_actions[0])
            smoothness_penalty = -0.02 * action_diff  # Reduced from -0.05
            
            # 6. Success bonus
            success_bonus = 5.0 if distance < 0.1 else 0.0  # Reduced from 10.0
            
            # Combine all reward components
            total_reward = (distance_reward + tendon_efficiency_reward + tactile_contact_reward + 
                           movement_penalty + smoothness_penalty + success_bonus)
            
            # Update previous actions for next step
            self.prev_actions[0] = actions.copy()
            
            self.episode_rewards[0] += total_reward
            
            # Success condition: base close to target
            success = distance < 0.1
            
            # Termination conditions
            dones = np.array([
                self.step_counts[0] >= self.max_steps or 
                success or
                distance > 1.5 or  # Reduced from 2.0
                base_pos[2] < 0.05  # Fell down
            ])
            
            # Comprehensive info dictionary for TensorBoard logging
            infos = [{
                "success": success,
                "distance": distance,
                "distance_reward": distance_reward,
                "tendon_efficiency_reward": tendon_efficiency_reward,
                "tactile_contact_reward": tactile_contact_reward,
                "movement_penalty": movement_penalty,
                "smoothness_penalty": smoothness_penalty,
                "tendon_forces": tendon_force_dict,
                "tactile_spans": tactile_spans.tolist(),
                
                # For data logging
                "base_pos_x": base_pos[0], "base_pos_y": base_pos[1], "base_pos_z": base_pos[2],
                "base_vel_x": base_vel[0], "base_vel_y": base_vel[1], "base_vel_z": base_vel[2],
                "base_ang_vel_x": base_ang_vel[0], "base_ang_vel_y": base_ang_vel[1], "base_ang_vel_z": base_ang_vel[2],
                "target_x": self.target_pos[0], "target_y": self.target_pos[1], "target_z": self.target_pos[2],
                "distance_to_target": distance,
                "tendon_force_index": tendon_forces[0], "tendon_force_middle": tendon_forces[1],
                "tendon_force_ring": tendon_forces[2], "tendon_force_thumb": tendon_forces[3],
                "control_linear_x": base_actions[0], "control_linear_y": base_actions[1], "control_linear_z": base_actions[2],
                "control_angular_x": base_actions[3], "control_angular_y": base_actions[4], "control_angular_z": base_actions[5],
                "tactile_deformation_spans": np.mean(tactile_spans),
                "tactile_contact_detected": 0,  # Disabled for stability
                "reward": total_reward,
                "tendon_efficiency_reward": tendon_efficiency_reward,
            }]
            
            # Add episode info when done
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
            # Clean up
            if self.hand is not None: 
                p.removeBody(self.hand)
                self.hand = None
            if self.target_sphere is not None:
                p.removeBody(self.target_sphere)
                self.target_sphere = None
            
            # Setup new simulation
            self._setup_simulation()
            
            # Reset counters and previous actions
            self.step_counts[dones] = 0
            self.episode_rewards[dones] = 0
            self.episode_lengths[dones] = 0
            self.prev_actions[dones] = 0
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
    """Create comprehensive plots from the logged data with GPU monitoring"""
    print(f"Creating plots from {csv_file}...")
    
    try:
        # Load data
        df = pd.read_csv(csv_file)
        print(f"Loaded data with {len(df)} rows and {len(df.columns)} columns")
        
        if len(df) == 0:
            print("No data to plot")
            return
        
        # Clean episode column
        if 'episode' in df.columns:
            df['episode'] = pd.to_numeric(df['episode'], errors='coerce')
            df['episode'] = df['episode'].ffill().fillna(0).astype(int)
        
        # SAMPLE DATA FOR PLOTTING (aggressive sampling for large datasets)
        max_plot_points = 5000  # Reduced from 10000
        if len(df) > max_plot_points:
            print(f"Sampling {max_plot_points} points from {len(df)} for plotting performance")
            step_size = len(df) // max_plot_points
            df_plot = df.iloc[::step_size].copy()
        else:
            df_plot = df.copy()
        
        # Create output directory for plots
        plot_dir = csv_file.parent / "plots"
        plot_dir.mkdir(exist_ok=True)
        
        plt.style.use('default')
        plt.rcParams['agg.path.chunksize'] = 5000  # Reduced for memory
        plt.rcParams['figure.max_open_warning'] = 0
        
        step_vals = df_plot['step'].values
        
        print(f"Creating plots with {len(df_plot)} sampled data points...")
        
        # 1. Base Position and Movement Analysis
        fig, axes = plt.subplots(2, 2, figsize=(12, 8))  # Smaller figure
        fig.suptitle('Base Position and Movement Analysis', fontsize=14)
        
        # Position plot
        axes[0, 0].plot(step_vals, df_plot['base_pos_x'].values, label='X', alpha=0.7, linewidth=0.5)
        axes[0, 0].plot(step_vals, df_plot['base_pos_y'].values, label='Y', alpha=0.7, linewidth=0.5)
        axes[0, 0].plot(step_vals, df_plot['base_pos_z'].values, label='Z', alpha=0.7, linewidth=0.5)
        axes[0, 0].axhline(y=df_plot['target_x'].iloc[0], color='r', linestyle='--', label='Target X', alpha=0.6)
        axes[0, 0].axhline(y=df_plot['target_y'].iloc[0], color='g', linestyle='--', label='Target Y', alpha=0.6)
        axes[0, 0].axhline(y=df_plot['target_z'].iloc[0], color='b', linestyle='--', label='Target Z', alpha=0.6)
        axes[0, 0].set_title('Base Position vs Time')
        axes[0, 0].set_xlabel('Step')
        axes[0, 0].set_ylabel('Position (m)')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # Distance plot
        axes[0, 1].plot(step_vals, df_plot['distance_to_target'].values, label='Distance to Target', linewidth=1.0)
        axes[0, 1].set_title('Distance to Target vs Time')
        axes[0, 1].set_xlabel('Step')
        axes[0, 1].set_ylabel('Distance (m)')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # GPU memory usage
        if 'gpu_memory_used' in df_plot.columns:
            axes[1, 0].plot(step_vals, df_plot['gpu_memory_used'].values, 'purple', linewidth=1.0)
            axes[1, 0].axhline(y=4.8, color='red', linestyle='--', label='GPU Limit (60%)', alpha=0.8)
            axes[1, 0].set_title('GPU Memory Usage')
            axes[1, 0].set_xlabel('Step')
            axes[1, 0].set_ylabel('Memory (GB)')
            axes[1, 0].legend()
            axes[1, 0].grid(True, alpha=0.3)
        
        # Reward over time
        axes[1, 1].plot(step_vals, df_plot['reward'].values, alpha=0.6, linewidth=0.5)
        window_size = min(200, len(df_plot) // 20)
        if window_size > 1:
            rolling_reward = df_plot['reward'].rolling(window=window_size, center=True).mean()
            axes[1, 1].plot(step_vals, rolling_reward.values, 'red', linewidth=2, label=f'Rolling Mean')
            axes[1, 1].legend()
        axes[1, 1].set_title('Reward vs Time')
        axes[1, 1].set_xlabel('Step')
        axes[1, 1].set_ylabel('Reward')
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(plot_dir / 'training_overview.png', dpi=120, bbox_inches='tight')
        plt.close()
        print("Training overview plot created")
        
        # 2. Tendon Force Analysis
        fig, axes = plt.subplots(2, 2, figsize=(12, 8))
        fig.suptitle('Tendon Force Analysis', fontsize=14)
        
        # Individual tendon forces
        axes[0, 0].plot(step_vals, df_plot['tendon_force_index'].values, label='Index', alpha=0.7, linewidth=0.8)
        axes[0, 0].plot(step_vals, df_plot['tendon_force_middle'].values, label='Middle', alpha=0.7, linewidth=0.8)
        axes[0, 0].plot(step_vals, df_plot['tendon_force_ring'].values, label='Ring', alpha=0.7, linewidth=0.8)
        axes[0, 0].plot(step_vals, df_plot['tendon_force_thumb'].values, label='Thumb', alpha=0.7, linewidth=0.8)
        axes[0, 0].set_title('Individual Tendon Forces')
        axes[0, 0].set_xlabel('Step')
        axes[0, 0].set_ylabel('Normalized Force')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # Average tendon force
        avg_tendon_force = (df_plot['tendon_force_index'] + df_plot['tendon_force_middle'] + 
                           df_plot['tendon_force_ring'] + df_plot['tendon_force_thumb']) / 4
        axes[0, 1].plot(step_vals, avg_tendon_force.values, linewidth=1.2)
        axes[0, 1].set_title('Average Tendon Force')
        axes[0, 1].set_xlabel('Step')
        axes[0, 1].set_ylabel('Normalized Force')
        axes[0, 1].grid(True, alpha=0.3)
        
        # Tendon force distribution
        axes[1, 0].hist([df_plot['tendon_force_index'].values, df_plot['tendon_force_middle'].values,
                        df_plot['tendon_force_ring'].values, df_plot['tendon_force_thumb'].values],
                       bins=30, alpha=0.7, label=['Index', 'Middle', 'Ring', 'Thumb'])  # Reduced bins
        axes[1, 0].set_title('Tendon Force Distribution')
        axes[1, 0].set_xlabel('Normalized Force')
        axes[1, 0].set_ylabel('Frequency')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        
        # Success rate over episodes
        if 'episode' in df_plot.columns and 'success' in df_plot.columns:
            try:
                episode_success = df.groupby('episode')['success'].max()
                episode_indices = np.array(episode_success.index)
                success_values = np.array(episode_success.values)
                
                if len(episode_indices) > 500:  # Sample episodes for plotting
                    step_ep = len(episode_indices) // 500
                    episode_indices = episode_indices[::step_ep]
                    success_values = success_values[::step_ep]
                
                axes[1, 1].plot(episode_indices, success_values, 'o-', alpha=0.7, markersize=1)
                if len(episode_success) > 10:
                    window = min(25, len(episode_success) // 10)  # Smaller window
                    rolling_success = episode_success.rolling(window=window, center=True).mean()
                    if len(episode_indices) > 500:
                        rolling_vals = np.array(rolling_success.values)[::step_ep]
                    else:
                        rolling_vals = np.array(rolling_success.values)
                    axes[1, 1].plot(episode_indices, rolling_vals, 'red', linewidth=2, 
                                   label=f'Rolling Success Rate')
                    axes[1, 1].legend()
                axes[1, 1].set_title('Success Rate per Episode')
                axes[1, 1].set_xlabel('Episode')
                axes[1, 1].set_ylabel('Success (0/1)')
                axes[1, 1].grid(True, alpha=0.3)
            except Exception as e:
                print(f"Skipping episode success plot due to error: {e}")
                axes[1, 1].text(0.5, 0.5, 'Episode data unavailable', ha='center', va='center', transform=axes[1, 1].transAxes)
        
        plt.tight_layout()
        plt.savefig(plot_dir / 'tendon_analysis.png', dpi=120, bbox_inches='tight')
        plt.close()
        print("Tendon analysis plot created")
        
    except Exception as e:
        print(f"Error in plotting: {e}")
        import traceback
        traceback.print_exc()
        plt.close('all')
    
    print(f"Plots saved to: {plot_dir}")
    
    # Print summary statistics
    print("\n=== SC-1 GPU-STABLE TRAINING SUMMARY ===")
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
        print(f"Success rate: {df['success'].mean()*100:.1f}%")
        if 'gpu_memory_used' in df.columns:
            max_gpu_mem = df['gpu_memory_used'].max()
            avg_gpu_mem = df['gpu_memory_used'].mean()
            print(f"Max GPU memory used: {max_gpu_mem:.2f} GB")
            print(f"Average GPU memory: {avg_gpu_mem:.2f} GB")


# --- Main Training and Analysis Script ---
if __name__ == "__main__":
    # Create descriptive log directory name
    import sys
    
    # Get run number from command line or use 1 as default
    run_number = sys.argv[1] if len(sys.argv) > 1 else "1"
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Create organized training directory structure
    base_training_dir = Path("./SC1_Training_Runs_Stable/")
    base_training_dir.mkdir(parents=True, exist_ok=True)
    
    log_dir = base_training_dir / f"Run_{timestamp}_SC1_GPU_Stable"
    log_dir.mkdir(parents=True, exist_ok=True)
    
    print("=" * 80)
    print(f"SC-1 GPU-STABLE TENDON-BASED TRAINING - RUN #{run_number}")
    print("=" * 80)
    
    # Create subdirectory for TensorBoard logs
    tb_log_dir = log_dir / "tensorboard"
    tb_log_dir.mkdir(parents=True, exist_ok=True)
    
    try:
        # Initialize GPU monitor
        gpu_monitor = GPUMonitor(max_memory_gb=4.8)  # 60% of 8GB RTX 5050
        
        # Create data logger
        data_logger = DataLogger(log_dir)
        
        # Create environment with reduced computational load
        print("\nCreating GPU-stable tendon-controlled environment...")
        env = TendonAllegroReachingEnv(vis=False)
        env.set_data_logger(data_logger)
        
        # Create enhanced callback for TensorBoard logging with GPU monitoring
        callback = EnhancedTensorboardCallback(data_logger, gpu_monitor, log_freq=200)
        
        # Configure custom logger for TensorBoard
        custom_logger = configure(str(tb_log_dir), ["stdout", "tensorboard"])
        
        # Create PPO model optimized for GPU stability
        print("Creating GPU-stable PPO model...")
        
        # Check for GPU availability
        if torch.cuda.is_available():
            device = "cuda"
            print(f"GPU acceleration enabled: {torch.cuda.get_device_name(0)}")
            print(f"GPU Memory limit: {torch.cuda.get_device_properties(0).total_memory * 0.6 / 1e9:.1f} GB (60% of total)")
        else:
            device = "cpu"
            print("No GPU found, using CPU")
        
        model = PPO(
            "MlpPolicy",
            env, 
            verbose=1,
            tensorboard_log=str(tb_log_dir),
            n_steps=1024,      # Reduced from 2048
            learning_rate=3e-4,
            n_epochs=5,        # Reduced from 10
            batch_size=32,     # Reduced from 64  
            gamma=0.99,
            gae_lambda=0.95,
            clip_range=0.2,
            ent_coef=0.01,
            vf_coef=0.5,
            max_grad_norm=0.5,
            device=device
        )
        
        # Set the custom logger
        model.set_logger(custom_logger)

        print("\n" + "=" * 80)
        print("SC-1 GPU-STABLE TRAINING CONFIGURATION")
        print("=" * 80)
        print(f"Log Directory: {log_dir}")
        print(f"TensorBoard Directory: {tb_log_dir}")
        print(f"Target Position: {env.target_pos} (STATIC)")
        print(f"Max Steps per Episode: {env.max_steps} (reduced for stability)")
        print(f"Total Training Timesteps: 100,000 (reduced for testing)")
        print(f"Simulation Frequency: {env.sim_freq} Hz (reduced for GPU stability)")
        print("GPU Stability Features:")
        print("  - 60% GPU memory limit")
        print("  - Reduced batch sizes and simulation frequency")
        print("  - Tactile sensors disabled") 
        print("  - GPU memory monitoring and cleanup")
        print("  - Automatic training stop on GPU overload")
        print("Action Space: 10D (6 base movement + 4 tendon forces)")
        print("Observation Space: 30D (base + target + fingers + tendons)")
        print("=" * 80)
        
        print(f"\nTo monitor training in TensorBoard, run:")
        print(f"   tensorboard --logdir {tb_log_dir}")
        print("   Then open: http://localhost:6006")
        print("\n" + "=" * 80 + "\n")
        
        # Train the model with reduced timesteps for testing
        print("Starting SC-1 GPU-stable training (reduced timesteps for testing)...")
        model.learn(
            total_timesteps=80000,  # Reduced from 500000 for initial testing
            callback=callback,
            log_interval=20,  # Increased interval
            tb_log_name=f"SC1_GPUStable_run{run_number}",
            reset_num_timesteps=True,
            progress_bar=True
        )
        
        # Save the model
        model_path = log_dir / f"sc1_gpu_stable_model_run{run_number}_{timestamp}"
        model.save(str(model_path))
        print(f"\nTraining Complete! Model saved to {model_path}.zip")
        
        # Save training data to CSV
        csv_file = data_logger.save_to_csv("sc1_gpu_stable_training_data.csv")
        
        # Test the trained model
        print(f"\nTesting trained GPU-stable model for 5 episodes...")
        print("=" * 60)
        
        env_test = TendonAllegroReachingEnv(vis=False)
        obs = env_test.reset()
        test_results = []
        test_episode = 0
        test_steps = 0
        
        while test_episode < 5 and test_steps < 2000:  # Reduced test episodes
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, info = env_test.step(action)
            test_steps += 1
            
            if done[0]:
                test_episode += 1
                distance = info[0].get('distance', float('inf'))
                success = info[0].get('success', False)
                episode_reward = info[0].get('episode', {}).get('r', 0)
                tendon_forces = info[0].get('tendon_forces', {})
                avg_tendon = np.mean(list(tendon_forces.values())) if tendon_forces else 0
                
                test_results.append({
                    'episode': test_episode, 
                    'distance': distance, 
                    'success': success,
                    'reward': episode_reward,
                    'avg_tendon': avg_tendon
                })
                status = "SUCCESS" if success else "FAILED"
                print(f"  Test Episode {test_episode}: {status} | Distance={distance:.4f}m | Reward={episode_reward:.1f}")
                obs = env_test.reset()
        
        env_test.close()
        
        # Calculate test statistics
        successes = sum(r['success'] for r in test_results)
        avg_distance = np.mean([r['distance'] for r in test_results]) if test_results else 0
        avg_reward = np.mean([r['reward'] for r in test_results]) if test_results else 0
        
        print("\n" + "=" * 80)
        print("SC-1 GPU-STABLE TEST RESULTS")
        print("=" * 80)
        print(f"Success Rate: {successes}/{len(test_results)} ({successes/len(test_results)*100:.1f}%)")
        print(f"Average Final Distance: {avg_distance:.4f}m")
        print(f"Average Episode Reward: {avg_reward:.1f}")
        print("=" * 80)
        
        # Generate analysis plots
        print(f"\nGenerating analysis plots...")
        create_plots(csv_file)
        
        print("\n" + "=" * 80)
        print("SC-1 GPU-STABLE TRAINING COMPLETE!")
        print("=" * 80)
        print(f"All results saved to: {log_dir}")
        print(f"TensorBoard logs: {tb_log_dir}")
        print(f"Analysis plots: {log_dir}/plots/")
        print("=" * 80)
        
        # Final GPU cleanup
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            print("GPU cache cleared")
        
    except KeyboardInterrupt:
        print("\nTraining interrupted by user")
        if 'data_logger' in locals():
            csv_file = data_logger.save_to_csv("sc1_training_data_interrupted.csv")
            create_plots(csv_file)
        print("Partial results saved.")
        
    except Exception as e:
        print(f"\nError occurred: {e}")
        import traceback
        traceback.print_exc()
        
    finally:
        try:
            if 'env' in locals():
                env.close()
            # Final GPU cleanup
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            print("Environment closed and GPU cache cleared.")
        except:
            pass