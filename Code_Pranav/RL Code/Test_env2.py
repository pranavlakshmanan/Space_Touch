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
            
            # Control inputs (6 DOF base movement)
            'control_linear_x': [], 'control_linear_y': [], 'control_linear_z': [],
            'control_angular_x': [], 'control_angular_y': [], 'control_angular_z': [],
            
            # Rewards and success metrics
            'reward': [],
            'distance_reward': [],
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
        
        # Always append timestamp and step
        self.data['timestamp'].append(time.time())
        self.data['step'].append(self.global_step)
        self.data['episode'].append(self.current_episode)
    
    def new_episode(self):
        """Increment episode counter"""
        self.current_episode += 1
    
    def save_to_csv(self, filename="training_data.csv"):
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


class EnhancedTensorboardCallback(BaseCallback):
    """Enhanced callback for comprehensive TensorBoard logging"""
    
    def __init__(self, data_logger, log_freq=100, verbose=0):
        super(EnhancedTensorboardCallback, self).__init__(verbose)
        self.data_logger = data_logger
        self.log_freq = log_freq
        self.episode_rewards = []
        self.episode_lengths = []
        self.episode_successes = []
        
        # Buffers for averaging metrics
        self.recent_distances = []
        self.recent_velocities = []
        self.recent_angular_velocities = []
        self.recent_control_inputs = []
        self.recent_ee_distances = []
        
    def _on_step(self) -> bool:
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
                    if len(self.episode_rewards) >= 100:
                        self.logger.record('episode/reward_mean_100', np.mean(self.episode_rewards[-100:]))
                        self.logger.record('episode/length_mean_100', np.mean(self.episode_lengths[-100:]))
                
                # Step-wise metrics logging
                if 'distance' in info:
                    self.recent_distances.append(info['distance'])
                
                if 'success' in info:
                    if 'episode' in info:  # Only track at episode end
                        self.episode_successes.append(float(info['success']))
                
                if 'avg_ee_distance' in info:
                    self.recent_ee_distances.append(info['avg_ee_distance'])
                
                # Extract velocity and control data
                if 'base_vel_x' in info:
                    vel_magnitude = np.sqrt(info.get('base_vel_x', 0)**2 + 
                                           info.get('base_vel_y', 0)**2 + 
                                           info.get('base_vel_z', 0)**2)
                    self.recent_velocities.append(vel_magnitude)
                
                if 'base_ang_vel_x' in info:
                    ang_vel_magnitude = np.sqrt(info.get('base_ang_vel_x', 0)**2 + 
                                               info.get('base_ang_vel_y', 0)**2 + 
                                               info.get('base_ang_vel_z', 0)**2)
                    self.recent_angular_velocities.append(ang_vel_magnitude)
                
                # Control input magnitudes
                if 'control_linear_x' in info:
                    control_magnitude = np.sqrt(info.get('control_linear_x', 0)**2 + 
                                              info.get('control_linear_y', 0)**2 + 
                                              info.get('control_linear_z', 0)**2)
                    self.recent_control_inputs.append(control_magnitude)
                
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
            
            # End effector distances
            if self.recent_ee_distances:
                self.logger.record('metrics/ee_distance_mean', np.mean(self.recent_ee_distances))
                self.recent_ee_distances = []
            
            # Velocity metrics
            if self.recent_velocities:
                self.logger.record('dynamics/linear_velocity_mean', np.mean(self.recent_velocities))
                self.logger.record('dynamics/linear_velocity_max', np.max(self.recent_velocities))
                self.recent_velocities = []
            
            # Angular velocity metrics
            if self.recent_angular_velocities:
                self.logger.record('dynamics/angular_velocity_mean', np.mean(self.recent_angular_velocities))
                self.logger.record('dynamics/angular_velocity_max', np.max(self.recent_angular_velocities))
                self.recent_angular_velocities = []
            
            # Control input metrics
            if self.recent_control_inputs:
                self.logger.record('control/input_magnitude_mean', np.mean(self.recent_control_inputs))
                self.logger.record('control/input_magnitude_std', np.std(self.recent_control_inputs))
                
                # Detect chattering (high frequency changes)
                if len(self.recent_control_inputs) > 1:
                    control_diff = np.abs(np.diff(self.recent_control_inputs))
                    chattering_metric = np.mean(control_diff)
                    self.logger.record('control/chattering_metric', chattering_metric)
                
                self.recent_control_inputs = []
            
            # Success rate
            if self.episode_successes:
                success_rate = np.mean(self.episode_successes[-100:]) if len(self.episode_successes) >= 100 else np.mean(self.episode_successes)
                self.logger.record('performance/success_rate_100', success_rate)
                self.logger.record('performance/total_successes', sum(self.episode_successes))
            
            # Training progress
            self.logger.record('training/timesteps', self.num_timesteps)
            self.logger.record('training/episodes', len(self.episode_rewards))
            
            # Learning rate (if available)
            if hasattr(self.model, 'learning_rate'):
                if callable(self.model.learning_rate):
                    lr = self.model.learning_rate(self.model._current_progress_remaining)
                else:
                    lr = self.model.learning_rate
                self.logger.record('training/learning_rate', lr)
            
        return True
    
    def _on_training_end(self) -> None:
        """Called at the end of training"""
        if self.episode_successes:
            final_success_rate = np.mean(self.episode_successes[-100:]) if len(self.episode_successes) >= 100 else np.mean(self.episode_successes)
            print(f"\nFinal Success Rate (last 100 episodes): {final_success_rate*100:.1f}%")
            self.logger.record('final/success_rate', final_success_rate)
            self.logger.record('final/total_episodes', len(self.episode_rewards))
            self.logger.record('final/mean_reward', np.mean(self.episode_rewards[-100:]) if len(self.episode_rewards) >= 100 else np.mean(self.episode_rewards))


class SimplifiedAllegroReachingEnv(VecEnv):
    """
    Simplified environment focusing only on arm base movement to reach targets.
    Fingers are kept static to focus on basic reaching behavior.
    """
    
    def __init__(self, 
                 num_envs=1,
                 vis=False,
                 max_steps=500,
                 urdf_hand="/home/pralak/Space_touch/examples/allegro_hand_description/allegro_hand_description_left_digit_fixed.urdf",
                 target_range=0.3):
        
        self.num_envs = num_envs
        self.vis = vis
        self.max_steps = max_steps
        self.sim_freq = 240
        self.urdf_hand = urdf_hand
        self.target_range = target_range
        
        # Static target position for consistent training
        self.target_pos = np.array([0.25, 0.15, 0.35])  # Fixed static target
        
        # Initialize PyBullet in DIRECT mode for headless operation
        self._init_pybullet()
        
        self.hand = None
        self.target_sphere = None
        
        # Only base movement (6 DOF: 3 linear + 3 angular)
        action_dim = 6
        self.action_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(action_dim,), dtype=np.float32)
        
        # Simple observation: base pose + target pose + distances
        obs_dim = 3 + 3 + 3 + 1  # base_pos + target_pos + base_vel + distance
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
                print(f"Loaded hand from: {self.urdf_hand}")
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
            
            # Keep all joints static (hardcoded) - FIXED: Proper joint control
            num_joints = p.getNumJoints(self.hand)
            for i in range(num_joints):
                joint_info = p.getJointInfo(self.hand, i)
                joint_type = joint_info[2]
                if joint_type in (p.JOINT_REVOLUTE, p.JOINT_PRISMATIC):
                    # Set to a comfortable pose and hold
                    p.resetJointState(self.hand, i, 0.5)  # Slightly bent fingers
                    p.setJointMotorControl2(
                        self.hand, i, 
                        controlMode=p.POSITION_CONTROL, 
                        targetPosition=0.5,
                        force=100
                    )
            
            # Let simulation settle
            for _ in range(50):
                p.stepSimulation()
                
        except Exception as e:
            print(f"Error setting up simulation: {e}")
            raise

    def _get_end_effector_positions(self):
        """Get positions of all fingertips"""
        ee_positions = []
        
        if self.hand is None:
            return [np.array([0, 0, 0.2])] * 4
            
        num_joints = p.getNumJoints(self.hand)
        
        for i in range(num_joints):
            joint_info = p.getJointInfo(self.hand, i)
            link_name = joint_info[12].decode('utf-8')  # Link name
            
            if 'tip' in link_name.lower():
                try:
                    link_state = p.getLinkState(self.hand, i)
                    ee_positions.append(np.array(link_state[0]))
                except:
                    continue
        
        # If no tips found, use the last few links
        if not ee_positions:
            for i in range(max(0, num_joints-4), num_joints):
                try:
                    link_state = p.getLinkState(self.hand, i)
                    ee_positions.append(np.array(link_state[0]))
                except:
                    continue
        
        # Ensure we have at least 4 positions (pad with base position if needed)
        if self.hand is not None:
            base_pos, _ = p.getBasePositionAndOrientation(self.hand)
        else:
            base_pos = [0, 0, 0.2]
            
        while len(ee_positions) < 4:
            ee_positions.append(np.array(base_pos))
            
        return ee_positions[:4]  # Return only first 4

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
            
            # Combine into observation
            obs = np.concatenate([
                base_pos,           # 3D
                self.target_pos,    # 3D  
                base_vel,           # 3D
                [distance]          # 1D
            ])
            
            return np.expand_dims(obs.astype(np.float32), axis=0)
            
        except Exception as e:
            print(f"Error getting observation: {e}")
            return np.zeros((1, self.observation_space.shape[0]), dtype=np.float32)

    def _log_step_data(self, actions, reward_info):
        """Log data for current step"""
        if self.data_logger is None or self.hand is None:
            return
            
        try:
            # Get current state
            base_pos, base_orn = p.getBasePositionAndOrientation(self.hand)
            base_vel, base_ang_vel = p.getBaseVelocity(self.hand)
            ee_positions = self._get_end_effector_positions()
            
            # Calculate distances
            distance_to_target = np.linalg.norm(np.array(base_pos) - self.target_pos)
            ee_distances = [np.linalg.norm(ee_pos - self.target_pos) for ee_pos in ee_positions]
            avg_ee_distance = np.mean(ee_distances)
            
            # Prepare data
            step_data = {
                # Base position and velocity
                'base_pos_x': base_pos[0], 'base_pos_y': base_pos[1], 'base_pos_z': base_pos[2],
                'base_vel_x': base_vel[0], 'base_vel_y': base_vel[1], 'base_vel_z': base_vel[2],
                'base_ang_vel_x': base_ang_vel[0], 'base_ang_vel_y': base_ang_vel[1], 'base_ang_vel_z': base_ang_vel[2],
                
                # End effector positions
                'ee1_pos_x': ee_positions[0][0], 'ee1_pos_y': ee_positions[0][1], 'ee1_pos_z': ee_positions[0][2],
                'ee2_pos_x': ee_positions[1][0], 'ee2_pos_y': ee_positions[1][1], 'ee2_pos_z': ee_positions[1][2],
                'ee3_pos_x': ee_positions[2][0], 'ee3_pos_y': ee_positions[2][1], 'ee3_pos_z': ee_positions[2][2],
                'ee4_pos_x': ee_positions[3][0], 'ee4_pos_y': ee_positions[3][1], 'ee4_pos_z': ee_positions[3][2],
                
                # Target and distances
                'target_x': self.target_pos[0], 'target_y': self.target_pos[1], 'target_z': self.target_pos[2],
                'distance_to_target': distance_to_target,
                'ee_target_distances': avg_ee_distance,
                
                # Control inputs
                'control_linear_x': actions[0], 'control_linear_y': actions[1], 'control_linear_z': actions[2],
                'control_angular_x': actions[3], 'control_angular_y': actions[4], 'control_angular_z': actions[5],
                
                # Rewards
                'reward': reward_info['total_reward'],
                'distance_reward': reward_info['distance_reward'],
                'success': reward_info['success'],
            }
            
            self.data_logger.log_step(step_data)
            
        except Exception as e:
            print(f"Error logging step data: {e}")

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
            
            # Apply base movement only
            linear_vel = actions[:3] * 0.3  # Reduced from 0.5 for smoother movement
            angular_vel = actions[3:6] * 0.8  # Reduced from 1.0 for smoother movement
            
            p.resetBaseVelocity(
                self.hand, 
                linearVelocity=linear_vel, 
                angularVelocity=angular_vel
            )
            
            p.stepSimulation()
            
            # Calculate reward components
            base_pos, base_orn = p.getBasePositionAndOrientation(self.hand)
            base_vel, base_ang_vel = p.getBaseVelocity(self.hand)
            base_pos = np.array(base_pos)
            distance = np.linalg.norm(base_pos - self.target_pos)
            
            # Get end effector positions for surrounding target check
            ee_positions = self._get_end_effector_positions()
            ee_distances = [np.linalg.norm(ee_pos - self.target_pos) for ee_pos in ee_positions]
            avg_ee_distance = np.mean(ee_distances)
            
            # 1. Distance reward (main objective)
            distance_reward = np.exp(-8.0 * distance)  # Reduced from -10.0 for gentler gradient
            
            # 2. End effector positioning reward
            surrounding_reward = np.exp(-4.0 * avg_ee_distance)  # Reduced from -5.0
            
            # 3. Movement efficiency penalty (discourage unnecessary movement)
            linear_vel_magnitude = np.linalg.norm(linear_vel)
            angular_vel_magnitude = np.linalg.norm(angular_vel)
            movement_penalty = -0.01 * (linear_vel_magnitude + angular_vel_magnitude)
            
            # 4. Action smoothness penalty (discourage jerky movements)
            action_diff = np.linalg.norm(actions - self.prev_actions[0])
            smoothness_penalty = -0.05 * action_diff
            
            # 5. Small stability reward for being stationary when close
            stability_reward = 0.1 if distance < 0.15 and linear_vel_magnitude < 0.1 else 0.0
            
            # 6. Bonus for reaching target
            success_bonus = 10.0 if distance < 0.1 else 0.0
            
            # Combine all reward components
            total_reward = (distance_reward + surrounding_reward + movement_penalty + 
                           smoothness_penalty + stability_reward + success_bonus)
            
            # Update previous actions for next step
            self.prev_actions[0] = actions.copy()
            
            self.episode_rewards[0] += total_reward
            
            # Success condition: base close to target
            success = distance < 0.1
            
            # Termination conditions
            dones = np.array([
                self.step_counts[0] >= self.max_steps or 
                success or
                distance > 2.0 or  # Too far away
                base_pos[2] < 0.05  # Fell down
            ])
            
            # Prepare reward info for logging
            reward_info = {
                'total_reward': total_reward,
                'distance_reward': distance_reward,
                'surrounding_reward': surrounding_reward,
                'movement_penalty': movement_penalty,
                'smoothness_penalty': smoothness_penalty,
                'success': success
            }
            
            # Log step data
            self._log_step_data(actions, reward_info)
            
            # Comprehensive info dictionary for TensorBoard logging
            infos = [{
                "success": success,
                "distance": distance,
                "avg_ee_distance": avg_ee_distance,
                "distance_reward": distance_reward,
                "surrounding_reward": surrounding_reward,
                "movement_penalty": movement_penalty,
                "smoothness_penalty": smoothness_penalty,
                "action_smoothness": action_diff,
                
                # Velocities for TensorBoard
                "base_vel_x": base_vel[0],
                "base_vel_y": base_vel[1], 
                "base_vel_z": base_vel[2],
                "base_ang_vel_x": base_ang_vel[0],
                "base_ang_vel_y": base_ang_vel[1],
                "base_ang_vel_z": base_ang_vel[2],
                
                # Control inputs for TensorBoard
                "control_linear_x": actions[0],
                "control_linear_y": actions[1],
                "control_linear_z": actions[2],
                "control_angular_x": actions[3],
                "control_angular_y": actions[4],
                "control_angular_z": actions[5],
                
                **reward_info
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
            self.prev_actions[dones] = 0  # Reset previous actions
        
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
        # Load data
        df = pd.read_csv(csv_file)
        print(f"Loaded data with {len(df)} rows and {len(df.columns)} columns")
        
        if len(df) == 0:
            print("No data to plot")
            return
        
        # Clean episode column - remove any dictionary strings
        if 'episode' in df.columns:
            # Convert episode column to numeric, coercing errors to NaN
            df['episode'] = pd.to_numeric(df['episode'], errors='coerce')
            # Forward fill NaN values (using new method)
            df['episode'] = df['episode'].ffill()
            # Fill any remaining NaN with 0
            df['episode'] = df['episode'].fillna(0)
            # Convert to int
            df['episode'] = df['episode'].astype(int)
        
        # SAMPLE DATA FOR PLOTTING - Handle large datasets
        max_plot_points = 10000  # Limit points for matplotlib performance
        if len(df) > max_plot_points:
            print(f"Sampling {max_plot_points} points from {len(df)} for plotting performance")
            # Use every nth point to sample evenly across the dataset
            step_size = len(df) // max_plot_points
            df_plot = df.iloc[::step_size].copy()
        else:
            df_plot = df.copy()
        
        # Create output directory for plots
        plot_dir = csv_file.parent / "plots"
        plot_dir.mkdir(exist_ok=True)
        
        # Set matplotlib styles and configure for large datasets
        plt.style.use('default')
        plt.rcParams['agg.path.chunksize'] = 10000  # Increase chunk size
        plt.rcParams['figure.max_open_warning'] = 0  # Disable warning
        
        # Convert to numpy arrays to avoid pandas indexing issues
        step_vals = df_plot['step'].values
        
        print(f"Creating plots with {len(df_plot)} sampled data points...")
        
        # 1. Base Position vs Time
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('Base Position and Velocity Analysis', fontsize=16)
        
        # Position plot
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
        
        # Velocity plot
        axes[0, 1].plot(step_vals, df_plot['base_vel_x'].values, label='Vel X', alpha=0.8, linewidth=0.8)
        axes[0, 1].plot(step_vals, df_plot['base_vel_y'].values, label='Vel Y', alpha=0.8, linewidth=0.8)
        axes[0, 1].plot(step_vals, df_plot['base_vel_z'].values, label='Vel Z', alpha=0.8, linewidth=0.8)
        axes[0, 1].set_title('Base Velocity vs Time')
        axes[0, 1].set_xlabel('Step')
        axes[0, 1].set_ylabel('Velocity (m/s)')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # Angular velocity plot
        axes[1, 0].plot(step_vals, df_plot['base_ang_vel_x'].values, label='AngVel X', alpha=0.8, linewidth=0.8)
        axes[1, 0].plot(step_vals, df_plot['base_ang_vel_y'].values, label='AngVel Y', alpha=0.8, linewidth=0.8)
        axes[1, 0].plot(step_vals, df_plot['base_ang_vel_z'].values, label='AngVel Z', alpha=0.8, linewidth=0.8)
        axes[1, 0].set_title('Base Angular Velocity vs Time')
        axes[1, 0].set_xlabel('Step')
        axes[1, 0].set_ylabel('Angular Velocity (rad/s)')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        
        # Distance to target
        axes[1, 1].plot(step_vals, df_plot['distance_to_target'].values, label='Base to Target', linewidth=1.2)
        axes[1, 1].plot(step_vals, df_plot['ee_target_distances'].values, label='Avg EE to Target', linewidth=1.2)
        axes[1, 1].set_title('Distance to Target vs Time')
        axes[1, 1].set_xlabel('Step')
        axes[1, 1].set_ylabel('Distance (m)')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(plot_dir / 'base_analysis.png', dpi=150, bbox_inches='tight')  # Reduced DPI
        plt.close()
        print("✓ Base analysis plot created")
        
        # 2. Reward Analysis
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('Reward and Success Analysis', fontsize=16)
        
        # Reward over time
        axes[0, 0].plot(step_vals, df_plot['reward'].values, alpha=0.7, linewidth=0.8)
        # Add rolling average on sampled data
        window_size = min(500, len(df_plot) // 20)
        if window_size > 1:
            rolling_reward = df_plot['reward'].rolling(window=window_size, center=True).mean()
            axes[0, 0].plot(step_vals, rolling_reward.values, 'red', linewidth=2, label=f'Rolling Mean ({window_size})')
            axes[0, 0].legend()
        axes[0, 0].set_title('Reward vs Time')
        axes[0, 0].set_xlabel('Step')
        axes[0, 0].set_ylabel('Reward')
        axes[0, 0].grid(True, alpha=0.3)
        
        # Distance reward component
        axes[0, 1].plot(step_vals, df_plot['distance_reward'].values, alpha=0.8, linewidth=0.8)
        axes[0, 1].set_title('Distance Reward Component')
        axes[0, 1].set_xlabel('Step')
        axes[0, 1].set_ylabel('Distance Reward')
        axes[0, 1].grid(True, alpha=0.3)
        
        # Success rate over episodes - FIXED INDEXING ERROR
        if 'episode' in df_plot.columns:
            try:
                episode_success = df.groupby('episode')['success'].max()  # Use full dataset for episode analysis
                # Convert index to numpy array to avoid pandas indexing issues
                episode_indices = np.array(episode_success.index)
                success_values = np.array(episode_success.values)
                
                # Sample episodes if too many
                if len(episode_indices) > 1000:
                    step_ep = len(episode_indices) // 1000
                    episode_indices = episode_indices[::step_ep]
                    success_values = success_values[::step_ep]
                
                axes[1, 0].plot(episode_indices, success_values, 'o-', alpha=0.7, markersize=2)
                # Rolling success rate
                if len(episode_success) > 10:
                    window = min(50, len(episode_success) // 10)
                    rolling_success = episode_success.rolling(window=window, center=True).mean()
                    if len(episode_indices) > 1000:
                        rolling_vals = np.array(rolling_success.values)[::step_ep]
                    else:
                        rolling_vals = np.array(rolling_success.values)
                    axes[1, 0].plot(episode_indices, rolling_vals, 'red', linewidth=2, 
                                   label=f'Rolling Success Rate ({window})')
                    axes[1, 0].legend()
                axes[1, 0].set_title('Success Rate per Episode')
                axes[1, 0].set_xlabel('Episode')
                axes[1, 0].set_ylabel('Success (0/1)')
                axes[1, 0].grid(True, alpha=0.3)
            except Exception as e:
                print(f"Skipping episode success plot due to error: {e}")
                axes[1, 0].text(0.5, 0.5, 'Episode data unavailable', ha='center', va='center', transform=axes[1, 0].transAxes)
        
        # Control input analysis
        control_magnitude = np.sqrt(df_plot['control_linear_x']**2 + df_plot['control_linear_y']**2 + df_plot['control_linear_z']**2)
        axes[1, 1].plot(step_vals, control_magnitude.values, alpha=0.7, linewidth=0.8)
        axes[1, 1].set_title('Control Input Magnitude')
        axes[1, 1].set_xlabel('Step')
        axes[1, 1].set_ylabel('Control Magnitude')
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(plot_dir / 'reward_analysis.png', dpi=150, bbox_inches='tight')
        plt.close()
        print("✓ Reward analysis plot created")
        
        # 3. Episode-based analysis (use full dataset for accuracy)
        if 'episode' in df.columns:
            fig, axes = plt.subplots(2, 2, figsize=(15, 10))
            fig.suptitle('Episode-based Analysis', fontsize=16)
            
            # Episode lengths
            episode_lengths = df.groupby('episode').size()
            episode_indices = np.array(episode_lengths.index)
            length_values = np.array(episode_lengths.values)
            
            # Sample if too many episodes
            if len(episode_indices) > 1000:
                step_ep = len(episode_indices) // 1000
                episode_indices = episode_indices[::step_ep]
                length_values = length_values[::step_ep]
            
            axes[0, 0].plot(episode_indices, length_values, 'o-', alpha=0.7, markersize=2)
            axes[0, 0].set_title('Episode Lengths')
            axes[0, 0].set_xlabel('Episode')
            axes[0, 0].set_ylabel('Steps')
            axes[0, 0].grid(True, alpha=0.3)
            
            # Episode rewards
            episode_rewards = df.groupby('episode')['reward'].sum()
            reward_indices = np.array(episode_rewards.index)
            reward_values = np.array(episode_rewards.values)
            
            if len(reward_indices) > 1000:
                step_ep = len(reward_indices) // 1000
                reward_indices = reward_indices[::step_ep]
                reward_values = reward_values[::step_ep]
            
            axes[0, 1].plot(reward_indices, reward_values, 'o-', alpha=0.7, markersize=2)
            if len(episode_rewards) > 10:
                window = min(50, len(episode_rewards) // 10)
                rolling_rewards = episode_rewards.rolling(window=window, center=True).mean()
                if len(reward_indices) > 1000:
                    rolling_vals = np.array(rolling_rewards.values)[::step_ep]
                else:
                    rolling_vals = np.array(rolling_rewards.values)
                axes[0, 1].plot(reward_indices, rolling_vals, 'red', linewidth=2,
                               label=f'Rolling Mean ({window})')
                axes[0, 1].legend()
            axes[0, 1].set_title('Episode Total Rewards')
            axes[0, 1].set_xlabel('Episode')
            axes[0, 1].set_ylabel('Total Reward')
            axes[0, 1].grid(True, alpha=0.3)
            
            # Final distance per episode
            episode_final_distance = df.groupby('episode')['distance_to_target'].last()
            distance_indices = np.array(episode_final_distance.index)
            distance_values = np.array(episode_final_distance.values)
            
            if len(distance_indices) > 1000:
                step_ep = len(distance_indices) // 1000
                distance_indices = distance_indices[::step_ep]
                distance_values = distance_values[::step_ep]
            
            axes[1, 0].plot(distance_indices, distance_values, 'o-', alpha=0.7, markersize=2)
            axes[1, 0].set_title('Final Distance per Episode')
            axes[1, 0].set_xlabel('Episode')
            axes[1, 0].set_ylabel('Final Distance (m)')
            axes[1, 0].grid(True, alpha=0.3)
            
            # Learning progress (distance improvement)
            if len(episode_final_distance) > 10:
                window = min(50, len(episode_final_distance) // 10)
                rolling_distance = episode_final_distance.rolling(window=window, center=True).mean()
                axes[1, 1].plot(distance_indices, distance_values, alpha=0.3, label='Raw', markersize=1)
                if len(distance_indices) > 1000:
                    rolling_vals = np.array(rolling_distance.values)[::step_ep]
                else:
                    rolling_vals = np.array(rolling_distance.values)
                axes[1, 1].plot(distance_indices, rolling_vals, 'red', linewidth=2,
                               label=f'Rolling Mean ({window})')
                axes[1, 1].set_title('Learning Progress (Distance)')
                axes[1, 1].set_xlabel('Episode')
                axes[1, 1].set_ylabel('Final Distance (m)')
                axes[1, 1].legend()
                axes[1, 1].grid(True, alpha=0.3)
            
            plt.tight_layout()
            plt.savefig(plot_dir / 'episode_analysis.png', dpi=150, bbox_inches='tight')
            plt.close()
            print("✓ Episode analysis plot created")
        
    except Exception as e:
        print(f"Error in plotting: {e}")
        import traceback
        traceback.print_exc()
        plt.close('all')
    
    print(f"All plots saved to: {plot_dir}")
    
    # Print summary statistics
    print("\n=== TRAINING SUMMARY ===")
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
        print(f"Success rate: {df['success'].mean()*100:.1f}%")


# --- Main Training and Analysis Script ---
if __name__ == "__main__":
    # Create descriptive log directory name
    import sys
    
    # Get run number from command line or use 1 as default
    run_number = sys.argv[1] if len(sys.argv) > 1 else "1"
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir = Path(f"./log{run_number}_allegro_reaching_{timestamp}/")
    log_dir.mkdir(parents=True, exist_ok=True)
    
    print("=" * 80)
    print(f"ALLEGRO HAND REACHING TRAINING - RUN #{run_number}")
    print("=" * 80)
    
    # Create subdirectory for TensorBoard logs
    tb_log_dir = log_dir / "tensorboard"
    tb_log_dir.mkdir(parents=True, exist_ok=True)
    
    try:
        # Create data logger
        data_logger = DataLogger(log_dir)
        
        # Create environment (always headless)
        print("\nCreating headless reaching environment...")
        env = SimplifiedAllegroReachingEnv(vis=False)  # Always headless
        env.set_data_logger(data_logger)
        
        # Create enhanced callback for TensorBoard logging
        callback = EnhancedTensorboardCallback(data_logger, log_freq=100)
        
        # Configure custom logger for TensorBoard
        custom_logger = configure(str(tb_log_dir), ["stdout", "tensorboard"])
        
        # Create PPO model with parameters suitable for reaching task
        print("Creating PPO model with enhanced logging...")
        model = PPO(
            "MlpPolicy",  # Simple MLP for position-based observations
            env, 
            verbose=1,
            tensorboard_log=str(tb_log_dir),
            n_steps=2048,  # Increased for better learning
            learning_rate=3e-4,
            n_epochs=10,
            batch_size=64,
            gamma=0.99,
            gae_lambda=0.95,
            clip_range=0.2,
            ent_coef=0.01,
            vf_coef=0.5,
            max_grad_norm=0.5,
            device="auto"
        )
        
        # Set the custom logger
        model.set_logger(custom_logger)

        print("\n" + "=" * 80)
        print("TRAINING CONFIGURATION")
        print("=" * 80)
        print(f"Log Directory: {log_dir}")
        print(f"TensorBoard Directory: {tb_log_dir}")
        print(f"Target Position: {env.target_pos} (STATIC)")
        print(f"Max Steps per Episode: {env.max_steps}")
        print(f"Total Training Timesteps: 100,000")
        print("Reward Components:")
        print("  - Distance reward (exponential)")
        print("  - End effector positioning") 
        print("  - Movement efficiency penalty")
        print("  - Action smoothness penalty")
        print("  - Success bonus")
        print("=" * 80)
        
        print(f"\nTo monitor training in TensorBoard, run:")
        print(f"   tensorboard --logdir {tb_log_dir}")
        print("   Then open: http://localhost:6006")
        print("\n" + "=" * 80 + "\n")
        
        # Train the model
        print("Starting training...")
        model.learn(
            total_timesteps=2000000,  # Increased for better convergence
            callback=callback,
            log_interval=10,
            tb_log_name=f"PPO_reaching_run{run_number}",
            reset_num_timesteps=True,
            progress_bar=True
        )
        
        # Save the model
        model_path = log_dir / f"ppo_reaching_model_run{run_number}_{timestamp}"
        model.save(str(model_path))
        print(f"\nTraining Complete! Model saved to {model_path}.zip")
        
        # Save training data to CSV
        csv_file = data_logger.save_to_csv("reaching_training_data.csv")
        
        # Test the trained model
        print(f"\nTesting trained model for 10 episodes...")
        print("=" * 60)
        print("TESTING PHASE EXPLANATION:")
        print("We create a fresh environment and test the trained model")
        print("to evaluate its performance on unseen episodes.")
        print("This helps us understand:")
        print("  1. How well the policy generalizes")
        print("  2. Consistency of performance")
        print("  3. Success rate without exploration noise")
        print("  4. Final distance achieved")
        print("=" * 60)
        
        env_test = SimplifiedAllegroReachingEnv(vis=False)  # Create fresh env for testing
        obs = env_test.reset()
        test_results = []
        test_episode = 0
        test_steps = 0
        
        while test_episode < 10 and test_steps < 5000:
            # Use deterministic=True to disable exploration noise
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, info = env_test.step(action)
            test_steps += 1
            
            if done[0]:
                test_episode += 1
                distance = info[0].get('distance', float('inf'))
                success = info[0].get('success', False)
                episode_reward = info[0].get('episode', {}).get('r', 0)
                test_results.append({
                    'episode': test_episode, 
                    'distance': distance, 
                    'success': success,
                    'reward': episode_reward
                })
                status = "SUCCESS" if success else "FAILED"
                print(f"  Test Episode {test_episode:2d}: {status} | Distance={distance:.4f}m | Reward={episode_reward:.1f}")
                obs = env_test.reset()
        
        env_test.close()
        
        # Calculate test statistics
        successes = sum(r['success'] for r in test_results)
        avg_distance = np.mean([r['distance'] for r in test_results]) if test_results else 0
        avg_reward = np.mean([r['reward'] for r in test_results]) if test_results else 0
        
        print("\n" + "=" * 80)
        print("TEST RESULTS SUMMARY")
        print("=" * 80)
        print(f"Success Rate: {successes}/{len(test_results)} ({successes/len(test_results)*100:.1f}%)")
        print(f"Average Final Distance: {avg_distance:.4f}m")
        print(f"Average Episode Reward: {avg_reward:.1f}")
        print("=" * 80)
        
        # Save final data with test results
        final_csv = data_logger.save_to_csv("reaching_training_data_final.csv")
        
        # Generate analysis plots
        print(f"\nGenerating analysis plots...")
        create_plots(final_csv)
        
        print("\n" + "=" * 80)
        print("TRAINING COMPLETE!")
        print("=" * 80)
        print(f"All results saved to: {log_dir}")
        print(f"TensorBoard logs: {tb_log_dir}")
        print(f"Analysis plots: {log_dir}/plots/")
        print(f"\nView TensorBoard results:")
        print(f"   tensorboard --logdir {tb_log_dir}")
        print("=" * 80)
        
    except KeyboardInterrupt:
        print("\nTraining interrupted by user")
        if 'data_logger' in locals():
            csv_file = data_logger.save_to_csv("reaching_training_data_interrupted.csv")
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
                print("Environment closed.")
        except:
            pass