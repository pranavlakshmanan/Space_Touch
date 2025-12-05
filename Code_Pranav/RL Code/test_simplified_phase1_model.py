#!/usr/bin/env python3
"""
Test Script for SC-1 Simplified Phase 1 Models
Finds latest model and generates comprehensive plots for test cases
Matches the simplified phase 1 training approach
"""

import time
import numpy as np
import pybullet as p
import pybullet_data
import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import VecEnv
import os
import pandas as pd
from datetime import datetime
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
import torch
import sys

# Fix compatibility
import collections.abc
import collections
collections.Mapping = collections.abc.Mapping
collections.MutableMapping = collections.abc.MutableMapping
collections.Sequence = collections.abc.Sequence
collections.MutableSequence = collections.abc.MutableSequence


class DataLogger:
    """SIMPLIFIED data logger matching sc1_simplified_phase1_training.py"""

    def __init__(self, log_dir):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)

        # Simplified data storage - matching the simplified training
        self.data = {
            'timestamp': [], 'step': [], 'episode': [],
            'distance_to_target': [],
            'reward': [],
            'distance_reward': [],
            'success_bonus': [],
            'tactile_reward': [],
            'success': [],
            'num_active_fingers': [],
            'target_x': [], 'target_y': [], 'target_z': [],
            'base_pos_x': [], 'base_pos_y': [], 'base_pos_z': [],
            'tendon_force_index': [], 'tendon_force_middle': [],
            'tendon_force_ring': [], 'tendon_force_thumb': [],
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

    def save_to_csv(self, filename="test_data.csv"):
        """Save all logged data to CSV"""
        max_len = max(len(arr) for arr in self.data.values() if arr)

        for key, arr in self.data.items():
            if len(arr) < max_len:
                last_val = arr[-1] if arr else 0
                arr.extend([last_val] * (max_len - len(arr)))

        df = pd.DataFrame(self.data)
        filepath = self.log_dir / filename
        df.to_csv(filepath, index=False)
        print(f"Test data saved to: {filepath}")
        return filepath


class TendonController:
    """Simplified tendon controller matching training script"""

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

        self.name_to_idx = {name: idx for name, idx in zip(joint_names, joint_indices)}

        self.finger_joints = {}
        for finger, chain in self.FINGER_CHAINS.items():
            self.finger_joints[finger] = []
            for joint_name in chain:
                if joint_name in self.name_to_idx:
                    self.finger_joints[finger].append(self.name_to_idx[joint_name])

    def compute_tendon_torques(self, tendon_forces):
        """Compute torque commands for each joint based on tendon forces"""
        torques = np.zeros(len(self.joint_indices))

        for finger, normalized_force in tendon_forces.items():
            if finger not in self.finger_joints:
                continue

            actual_force = normalized_force * self.MAX_TENDON_FORCE
            joints = self.finger_joints[finger]

            for joint_idx in joints:
                # Simplified: same moment arm for all joints
                moment_arm = 0.02  # 2cm fixed moment arm
                torque = actual_force * moment_arm * self.TENDON_FORCE_GAIN

                joint_state = p.getJointState(self.hand_id, joint_idx)
                current_velocity = joint_state[1]
                damping_torque = -self.TENDON_DAMPING * current_velocity
                final_torque = torque + damping_torque

                idx_in_list = self.joint_indices.index(joint_idx)
                torques[idx_in_list] = final_torque

        return torques


class SimplifiedTestEnv(VecEnv):
    """Test environment matching sc1_simplified_phase1_training.py exactly"""

    def __init__(self, num_envs=1, vis=True, max_steps=1000,
                 urdf_hand="/home/pralak/Space_Touch/examples/allegro_hand_description/allegro_hand_description_left_digit_fixed.urdf",
                 test_scenario=None):

        self.num_envs = num_envs
        self.vis = vis
        self.max_steps = max_steps
        self.urdf_hand = urdf_hand
        self.test_scenario = test_scenario
        self.is_testing = True
        self.training_timesteps = 0  # For curriculum compatibility

        self.sim_freq = 240.0
        self._init_pybullet()

        self.hand = None
        self.target_sphere = None
        self.tendon_controller = None
        self.hand_spawned = False

        # Action space: 6 DOF base movement + 4 tendon forces
        action_dim = 10
        self.action_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(action_dim,), dtype=np.float32)

        # SIMPLIFIED: 25D observation space (matching simplified training)
        obs_dim = 25  # base_pos(3) + target_pos(3) + base_vel(3) + finger_positions(12) + binary_tactile(4)
        self.observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32)

        super().__init__(num_envs, self.observation_space, self.action_space)

        self.step_counts = np.zeros(num_envs, dtype=np.int32)
        self.episode_rewards = np.zeros(num_envs)
        self.episode_lengths = np.zeros(num_envs, dtype=np.int32)
        self.prev_actions = np.zeros((num_envs, action_dim), dtype=np.float32)
        self.prev_distance = np.zeros(num_envs, dtype=np.float32)
        self.data_logger = None

        # Test scenarios matching different phases
        self.test_scenarios = {
            "static_close": {"distance": 0.2, "velocity": [0,0,0], "description": "Static close (Phase 1)"},
            "static_medium": {"distance": 0.3, "velocity": [0,0,0], "description": "Static medium (Phase 1)"},
            "static_far": {"distance": 0.4, "velocity": [0,0,0], "description": "Static far (Phase 1)"},
            "moving_slow": {"distance": 0.25, "velocity": [0.01,0,0], "description": "Slow moving (Phase 2)"},
            "moving_medium": {"distance": 0.3, "velocity": [0.02,0,0], "description": "Medium moving (Phase 3)"},
        }

        self.reset()

    def _init_pybullet(self):
        """Initialize PyBullet connection"""
        if self.vis:
            try:
                self.client_id = p.connect(p.GUI)
                p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0)  # Hide GUI panels
                print("✅ Visualization enabled")
            except:
                self.client_id = p.connect(p.DIRECT)
                print("⚠️  GUI failed, using headless mode")
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

    def _setup_simulation(self):
        """Setup the simulation environment matching simplified training"""
        try:
            # Always remove and recreate PyBullet bodies during testing
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
                if not hasattr(self, '_hand_loaded_printed'):
                    print(f"✓ Hand loaded from: {self.urdf_hand}")
                    self._hand_loaded_printed = True
            else:
                # Fallback simple hand
                hand_collision = p.createCollisionShape(p.GEOM_BOX, halfExtents=[0.1, 0.05, 0.02])
                hand_visual = p.createVisualShape(p.GEOM_BOX, halfExtents=[0.1, 0.05, 0.02],
                                                rgbaColor=[0.8, 0.6, 0.4, 1])
                self.hand = p.createMultiBody(baseMass=1.0, baseCollisionShapeIndex=hand_collision,
                                            baseVisualShapeIndex=hand_visual, basePosition=[0, 0, 0.2])
                self.hand_spawned = True

            # Setup target based on test scenario
            target_collision = p.createCollisionShape(p.GEOM_SPHERE, radius=0.02)
            target_visual = p.createVisualShape(p.GEOM_SPHERE, radius=0.02, rgbaColor=[1, 0, 0, 1])

            # Default target position
            target_pos = np.array([0.25, 0.15, 0.35])
            initial_velocity = [0, 0, 0]
            initial_angular_velocity = [0, 0, 0]

            # Apply test scenario if specified
            if self.test_scenario and self.test_scenario in self.test_scenarios:
                scenario = self.test_scenarios[self.test_scenario]
                # Adjust target distance from hand
                direction = target_pos / np.linalg.norm(target_pos)
                target_pos = direction * scenario["distance"]
                initial_velocity = scenario["velocity"]
                print(f"Test scenario: {scenario['description']}")

            self.target_sphere = p.createMultiBody(baseMass=0.5, baseCollisionShapeIndex=target_collision,
                                                 baseVisualShapeIndex=target_visual, basePosition=target_pos)

            p.resetBaseVelocity(self.target_sphere, linearVelocity=initial_velocity,
                              angularVelocity=initial_angular_velocity)

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

            # Initialize fingertip links for tactile sensing
            self.fingertip_links = []
            tip_labels = ["joint_15.0_tip", "joint_11.0_tip", "joint_7.0_tip", "joint_3.0_tip"]

            for i in range(num_joints):
                joint_info = p.getJointInfo(self.hand, i)
                joint_name = joint_info[1].decode()
                if joint_name in tip_labels:
                    self.fingertip_links.append(i)

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
        """Get current observation - SIMPLIFIED (25D)"""
        try:
            if self.hand is None:
                obs = np.zeros(self.observation_space.shape[0])
                return np.expand_dims(obs.astype(np.float32), axis=0)

            base_pos, _ = p.getBasePositionAndOrientation(self.hand)
            base_vel, _ = p.getBaseVelocity(self.hand)

            # Get current target position (dynamic for moving target)
            target_pos, _ = p.getBasePositionAndOrientation(self.target_sphere)

            base_pos = np.array(base_pos)
            base_vel = np.array(base_vel)
            target_pos = np.array(target_pos)

            finger_positions = self._get_finger_positions()
            binary_tactile = self._get_binary_tactile_feedback()

            # 25D observation (NO convex hull flag)
            obs = np.concatenate([
                base_pos,           # 3D
                target_pos,         # 3D
                base_vel,           # 3D
                finger_positions,   # 12D
                binary_tactile,     # 4D
            ])

            return np.expand_dims(obs.astype(np.float32), axis=0)

        except Exception as e:
            print(f"Error getting observation: {e}")
            return np.zeros((1, self.observation_space.shape[0]), dtype=np.float32)

    def step_wait(self):
        """Execute one step - SIMPLIFIED REWARD FUNCTION matching training"""
        self.step_counts += 1
        self.episode_lengths += 1
        actions = self.actions[0]

        try:
            if self.hand is None:
                obs = self._get_observation()
                return obs, np.array([-1.0], dtype=np.float32), np.array([True]), [{"error": "Hand not loaded"}]

            base_actions = actions[:6]
            tendon_actions = actions[6:10]

            # Apply base movement (no filtering to reduce complexity)
            linear_vel = base_actions[:3] * 0.3
            angular_vel = base_actions[3:6] * 0.8

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

            # ============================================================================
            # SIMPLIFIED REWARD FUNCTION - MATCHING TRAINING EXACTLY
            # ============================================================================

            # Get current distance
            base_pos, base_orn = p.getBasePositionAndOrientation(self.hand)
            target_pos, target_orn = p.getBasePositionAndOrientation(self.target_sphere)
            base_pos = np.array(base_pos)
            target_pos = np.array(target_pos)
            distance = np.linalg.norm(base_pos - target_pos)

            # Get tactile feedback
            binary_tactile = self._get_binary_tactile_feedback()
            num_active_fingers = np.sum(binary_tactile)

            # Calculate progress ratio (for curriculum)
            progress = min(1.0, self.training_timesteps / 500000.0) if hasattr(self, 'training_timesteps') else 0.0

            # ============ REWARD COMPONENT 1: Distance Progress ============
            distance_delta = self.prev_distance[0] - distance
            distance_reward = 20.0 * distance_delta  # Increased weight

            # Update previous distance
            self.prev_distance[0] = distance

            # ============ REWARD COMPONENT 2: Staged Success Bonus ============
            success_bonus = 0.0
            success = False

            # Stage 1: Just get close (easier threshold)
            if distance < 0.15:
                success_bonus += 10.0

            # Stage 2: Get very close with at least 1 finger contact
            if distance < 0.1 and num_active_fingers >= 1:
                success_bonus += 20.0

            # Stage 3: Perfect grasp - close distance with 2+ fingers
            if distance < 0.1 and num_active_fingers >= 2:
                success_bonus += 50.0
                success = True

            # ============ REWARD COMPONENT 3: Tactile Engagement (Curriculum) ============
            # Only apply after 100K timesteps when basic approach is learned
            tactile_reward = 0.0
            if progress > 0.2:  # After 20% of training
                if num_active_fingers >= 2:
                    tactile_reward = 5.0 * progress  # Scales up over training
                elif num_active_fingers >= 1:
                    tactile_reward = 2.0 * progress

            # ============ TOTAL REWARD ============
            total_reward = distance_reward + success_bonus + tactile_reward

            # Update action history
            self.prev_actions[0] = actions.copy()
            self.episode_rewards[0] += total_reward

            # ============ TERMINATION CONDITIONS ============
            # More lenient - give agent time to learn
            dones = np.array([
                self.step_counts[0] >= self.max_steps or
                success or
                distance > 3.0 or  # Increased from 2.0
                base_pos[2] < 0.01  # Only fail if completely fallen
            ])

            # ============ LOGGING (Simplified) ============
            self.step_data = {
                "distance_to_target": distance,
                "reward": total_reward,
                "distance_reward": distance_reward,
                "success_bonus": success_bonus,
                "tactile_reward": tactile_reward,
                "success": success,
                "num_active_fingers": num_active_fingers,
                "target_x": target_pos[0], "target_y": target_pos[1], "target_z": target_pos[2],
                "base_pos_x": base_pos[0], "base_pos_y": base_pos[1], "base_pos_z": base_pos[2],
                "tendon_force_index": tendon_forces[0], "tendon_force_middle": tendon_forces[1],
                "tendon_force_ring": tendon_forces[2], "tendon_force_thumb": tendon_forces[3],
            }

            if self.data_logger is not None:
                self.data_logger.log_step(self.step_data)

            # ============ INFO FOR CALLBACKS ============
            infos = [{
                "success": success,
                "distance": distance,
                "num_active_fingers": num_active_fingers,
                "reward_breakdown": {
                    "distance": distance_reward,
                    "success": success_bonus,
                    "tactile": tactile_reward
                }
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
            # Always remove and recreate PyBullet bodies during reset
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

            # Initialize previous distance
            if self.hand is not None:
                base_pos, _ = p.getBasePositionAndOrientation(self.hand)
                target_pos, _ = p.getBasePositionAndOrientation(self.target_sphere)
                initial_distance = np.linalg.norm(np.array(base_pos) - np.array(target_pos))
                self.prev_distance[0] = initial_distance

            # Reset counters and clear action history
            self.step_counts[dones] = 0
            self.episode_rewards[dones] = 0
            self.episode_lengths[dones] = 0
            self.prev_actions[dones] = 0
            self.current_tendon_forces = np.zeros(4)

        return self._get_observation()

    def reset(self):
        """Full environment reset"""
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


def find_latest_model():
    """Find the latest trained model from any SC1 training run"""
    base_dir = Path("./SC1_Training_Runs/")
    if not base_dir.exists():
        print("❌ No training runs found!")
        return None

    # Look for model files in all subdirectories
    model_files = []

    # Search patterns for different model types
    patterns = [
        "**/sc1_simplified_phase1_final_model*.zip",  # Simplified phase 1 final models
        "**/checkpoint_*.zip",  # Checkpoints
        "**/sc1_*final*.zip",   # Other final models
        "**/*model*.zip",       # Any model files
    ]

    for pattern in patterns:
        model_files.extend(list(base_dir.glob(pattern)))

    if not model_files:
        print("❌ No model files found!")
        return None

    # Sort by modification time, newest first
    model_files.sort(key=lambda x: x.stat().st_mtime, reverse=True)
    latest_model = model_files[0]

    print(f"✅ Found latest model: {latest_model.name}")
    print(f"   Path: {latest_model}")
    print(f"   Modified: {datetime.fromtimestamp(latest_model.stat().st_mtime)}")

    return latest_model


def create_phase1_plots(csv_file):
    """Create comprehensive plots matching simplified phase 1 approach"""
    try:
        df = pd.read_csv(csv_file)
        print(f"Creating Phase 1 plots from {len(df)} data points...")

        if len(df) == 0:
            print("No data to plot")
            return

        plot_dir = csv_file.parent / "plots"
        plot_dir.mkdir(exist_ok=True)

        plt.style.use('default')
        plt.rcParams['figure.figsize'] = (15, 10)

        # Sample data if too large
        if len(df) > 5000:
            step_size = len(df) // 5000
            df_plot = df.iloc[::step_size].copy()
        else:
            df_plot = df.copy()

        step_vals = df_plot['step'].values

        # 1. Phase 1 Performance Overview
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('SC-1 Simplified Phase 1 Test Results', fontsize=16, fontweight='bold')

        # Success over episodes
        if 'episode' in df_plot.columns:
            episode_success = df.groupby('episode')['success'].max()
            axes[0,0].plot(episode_success.index, episode_success.values, 'o-', alpha=0.7, markersize=4)
            success_rate = episode_success.mean()
            axes[0,0].axhline(y=success_rate, color='red', linestyle='--',
                             label=f'Avg Success Rate: {success_rate:.1%}')
            axes[0,0].axhline(y=0.2, color='green', linestyle='--', alpha=0.5,
                             label='Phase 1 Target (20%)')
            axes[0,0].set_title('Success Rate by Episode')
            axes[0,0].set_ylabel('Success (0/1)')
            axes[0,0].set_xlabel('Episode')
            axes[0,0].legend()
            axes[0,0].grid(True, alpha=0.3)

        # Distance to target over time
        axes[0,1].plot(step_vals, df_plot['distance_to_target'], alpha=0.7, linewidth=0.8, color='blue')
        axes[0,1].axhline(y=0.15, color='orange', linestyle='--', label='Stage 1 (0.15m)')
        axes[0,1].axhline(y=0.1, color='red', linestyle='--', label='Stage 2/3 (0.1m)')
        axes[0,1].set_title('Distance to Target Over Time')
        axes[0,1].set_ylabel('Distance (m)')
        axes[0,1].set_xlabel('Step')
        axes[0,1].legend()
        axes[0,1].grid(True, alpha=0.3)

        # Tactile engagement (key Phase 1 metric)
        axes[0,2].plot(step_vals, df_plot['num_active_fingers'], alpha=0.7, linewidth=0.8, color='green')
        axes[0,2].axhline(y=1, color='orange', linestyle='--', label='Stage 2 (1+ fingers)')
        axes[0,2].axhline(y=2, color='red', linestyle='--', label='Stage 3 (2+ fingers)')
        axes[0,2].set_title('Tactile Engagement (Active Fingers)')
        axes[0,2].set_ylabel('Number of Active Fingers')
        axes[0,2].set_xlabel('Step')
        axes[0,2].legend()
        axes[0,2].grid(True, alpha=0.3)

        # Reward components breakdown
        axes[1,0].plot(step_vals, df_plot['reward'], alpha=0.7, linewidth=0.8, label='Total', color='black')
        axes[1,0].plot(step_vals, df_plot['distance_reward'], alpha=0.7, linewidth=0.8, label='Distance', color='blue')
        axes[1,0].plot(step_vals, df_plot['success_bonus'], alpha=0.7, linewidth=0.8, label='Success', color='green')
        if 'tactile_reward' in df_plot.columns:
            axes[1,0].plot(step_vals, df_plot['tactile_reward'], alpha=0.7, linewidth=0.8, label='Tactile', color='purple')
        axes[1,0].set_title('Reward Components (Phase 1 Simplified)')
        axes[1,0].set_xlabel('Step')
        axes[1,0].set_ylabel('Reward')
        axes[1,0].legend()
        axes[1,0].grid(True, alpha=0.3)

        # Tendon usage analysis
        if all(col in df_plot.columns for col in ['tendon_force_index', 'tendon_force_middle', 'tendon_force_ring', 'tendon_force_thumb']):
            axes[1,1].plot(step_vals, df_plot['tendon_force_index'], label='Index', alpha=0.8, linewidth=0.8)
            axes[1,1].plot(step_vals, df_plot['tendon_force_middle'], label='Middle', alpha=0.8, linewidth=0.8)
            axes[1,1].plot(step_vals, df_plot['tendon_force_ring'], label='Ring', alpha=0.8, linewidth=0.8)
            axes[1,1].plot(step_vals, df_plot['tendon_force_thumb'], label='Thumb', alpha=0.8, linewidth=0.8)
            axes[1,1].set_title('Tendon Force Usage')
            axes[1,1].set_xlabel('Step')
            axes[1,1].set_ylabel('Normalized Force')
            axes[1,1].legend()
            axes[1,1].grid(True, alpha=0.3)

        # Phase 1 learning progression
        if 'episode' in df_plot.columns:
            # Calculate rolling metrics per episode
            episode_stats = df.groupby('episode').agg({
                'distance_to_target': 'min',
                'num_active_fingers': 'max',
                'success': 'max'
            }).reset_index()

            if len(episode_stats) > 1:
                window = min(5, len(episode_stats) // 3)
                if window > 1:
                    episode_stats['distance_rolling'] = episode_stats['distance_to_target'].rolling(window).mean()
                    episode_stats['fingers_rolling'] = episode_stats['num_active_fingers'].rolling(window).mean()

                    ax_twin = axes[1,2].twinx()
                    line1 = axes[1,2].plot(episode_stats['episode'], episode_stats['distance_rolling'],
                                          'b-', label='Min Distance', linewidth=2)
                    line2 = ax_twin.plot(episode_stats['episode'], episode_stats['fingers_rolling'],
                                        'g-', label='Max Fingers', linewidth=2)

                    axes[1,2].set_xlabel('Episode')
                    axes[1,2].set_ylabel('Distance (m)', color='b')
                    ax_twin.set_ylabel('Active Fingers', color='g')
                    axes[1,2].set_title('Learning Progression (Phase 1)')

                    # Combine legends
                    lines = line1 + line2
                    labels = [l.get_label() for l in lines]
                    axes[1,2].legend(lines, labels, loc='center right')
                    axes[1,2].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(plot_dir / 'phase1_performance_overview.png', dpi=150, bbox_inches='tight')
        plt.close()

        # 2. Trajectory Analysis
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        fig.suptitle('Hand Movement Trajectory Analysis', fontsize=16)

        if all(col in df_plot.columns for col in ['base_pos_x', 'base_pos_y', 'base_pos_z']):
            # 3D trajectory
            try:
                ax = fig.add_subplot(1, 3, 1, projection='3d')
                ax.plot(df_plot['base_pos_x'], df_plot['base_pos_y'], df_plot['base_pos_z'],
                       alpha=0.7, linewidth=1, color='blue')
                ax.scatter(df_plot['target_x'].iloc[0], df_plot['target_y'].iloc[0], df_plot['target_z'].iloc[0],
                          color='red', s=100, label='Target')
                ax.set_title('3D Hand Trajectory')
                ax.set_xlabel('X (m)')
                ax.set_ylabel('Y (m)')
                ax.set_zlabel('Z (m)')
                ax.legend()
            except:
                axes[0].text(0.5, 0.5, '3D plot unavailable', ha='center', va='center', transform=axes[0].transAxes)

            # XY trajectory
            axes[1].plot(df_plot['base_pos_x'], df_plot['base_pos_y'], alpha=0.7, linewidth=1, color='blue')
            axes[1].scatter(df_plot['target_x'].iloc[0], df_plot['target_y'].iloc[0],
                           color='red', s=100, label='Target')
            axes[1].set_title('XY Trajectory (Top View)')
            axes[1].set_xlabel('X (m)')
            axes[1].set_ylabel('Y (m)')
            axes[1].legend()
            axes[1].grid(True, alpha=0.3)
            axes[1].axis('equal')

            # Distance evolution with success coloring
            colors = ['green' if success else 'red' for success in df_plot['success']]
            axes[2].scatter(step_vals, df_plot['distance_to_target'], c=colors, alpha=0.6, s=1)
            axes[2].axhline(y=0.15, color='orange', linestyle='--', label='Stage 1')
            axes[2].axhline(y=0.1, color='green', linestyle='--', label='Stage 2/3')
            axes[2].set_title('Distance Evolution (Green=Success, Red=Fail)')
            axes[2].set_xlabel('Step')
            axes[2].set_ylabel('Distance (m)')
            axes[2].legend()
            axes[2].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(plot_dir / 'trajectory_analysis.png', dpi=150, bbox_inches='tight')
        plt.close()

        print(f"✅ Phase 1 test plots saved to: {plot_dir}")

    except Exception as e:
        print(f"❌ Error creating plots: {e}")
        import traceback
        traceback.print_exc()


def run_phase1_tests(model_path, visualize=True, episodes_per_scenario=3):
    """Run comprehensive Phase 1 tests matching the simplified training approach"""

    print("=" * 80)
    print("🧪 SC-1 SIMPLIFIED PHASE 1 MODEL TESTING")
    print("=" * 80)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    test_dir = Path(f"./SC1_Model_Tests/Phase1_Test_{timestamp}")
    test_dir.mkdir(parents=True, exist_ok=True)

    # Load model
    try:
        print(f"📂 Loading model: {model_path.name}")
        # Create dummy environment for model loading
        dummy_env = SimplifiedTestEnv(vis=False)
        model = PPO.load(str(model_path), env=dummy_env)
        dummy_env.close()
        print("✅ Model loaded successfully!")

        if torch.cuda.is_available():
            device = torch.cuda.get_device_name(0)
            print(f"🚀 Using GPU: {device}")

    except Exception as e:
        print(f"❌ Error loading model: {e}")
        return

    all_results = []

    # Test scenarios
    dummy_env = SimplifiedTestEnv(vis=False)
    scenarios = dummy_env.test_scenarios
    dummy_env.close()

    for scenario_name, scenario_config in scenarios.items():
        print(f"\n🎯 Testing scenario: {scenario_name}")
        print(f"   {scenario_config['description']}")

        # Create test environment for this scenario
        env_test = SimplifiedTestEnv(vis=visualize, test_scenario=scenario_name)
        test_data_logger = DataLogger(test_dir / f"scenario_{scenario_name}")
        env_test.set_data_logger(test_data_logger)

        scenario_results = []

        for episode in range(episodes_per_scenario):
            print(f"  Episode {episode+1}/{episodes_per_scenario}...", end=" ")

            obs = env_test.reset()
            episode_reward = 0
            episode_steps = 0
            max_fingers = 0
            min_distance = float('inf')
            achieved_stage1 = False
            achieved_stage2 = False
            achieved_stage3 = False

            while episode_steps < 1000:
                action, _ = model.predict(obs, deterministic=True)
                obs, reward, done, info = env_test.step(action)
                episode_reward += reward[0]
                episode_steps += 1

                # Track Phase 1 specific metrics
                fingers = info[0].get('num_active_fingers', 0)
                distance = info[0].get('distance', float('inf'))

                if fingers > max_fingers:
                    max_fingers = fingers
                if distance < min_distance:
                    min_distance = distance

                # Track stage achievements
                if distance < 0.15:
                    achieved_stage1 = True
                if distance < 0.1 and fingers >= 1:
                    achieved_stage2 = True
                if distance < 0.1 and fingers >= 2:
                    achieved_stage3 = True

                if done[0]:
                    break

                # Add visualization delay
                if visualize:
                    time.sleep(0.02)

            test_data_logger.new_episode()

            # Extract final results
            final_distance = info[0].get('distance', float('inf'))
            success = info[0].get('success', False)
            final_fingers = info[0].get('num_active_fingers', 0)

            result = {
                'scenario': scenario_name,
                'episode': episode + 1,
                'success': success,
                'final_distance': final_distance,
                'min_distance': min_distance,
                'final_fingers': final_fingers,
                'max_fingers': max_fingers,
                'achieved_stage1': achieved_stage1,
                'achieved_stage2': achieved_stage2,
                'achieved_stage3': achieved_stage3,
                'reward': episode_reward,
                'steps': episode_steps
            }

            scenario_results.append(result)
            all_results.append(result)

            status = "✅ SUCCESS" if success else "❌ FAILED"
            stages = []
            if achieved_stage1: stages.append("S1")
            if achieved_stage2: stages.append("S2")
            if achieved_stage3: stages.append("S3")
            stage_str = "+".join(stages) if stages else "None"

            print(f"{status} | Dist: {final_distance:.3f}m (min: {min_distance:.3f}) | Fingers: {final_fingers} (max: {max_fingers}) | Stages: {stage_str}")

        # Save scenario data and create plots
        test_csv = test_data_logger.save_to_csv(f"{scenario_name}_test_data.csv")
        create_phase1_plots(test_csv)

        # Scenario summary
        successes = sum(r['success'] for r in scenario_results)
        success_rate = successes / len(scenario_results)
        avg_final_distance = np.mean([r['final_distance'] for r in scenario_results])
        avg_min_distance = np.mean([r['min_distance'] for r in scenario_results])
        avg_max_fingers = np.mean([r['max_fingers'] for r in scenario_results])
        stage1_rate = np.mean([r['achieved_stage1'] for r in scenario_results])
        stage2_rate = np.mean([r['achieved_stage2'] for r in scenario_results])
        stage3_rate = np.mean([r['achieved_stage3'] for r in scenario_results])

        print(f"  📊 {scenario_name} Results:")
        print(f"     Success: {success_rate:.1%} | Avg Final Dist: {avg_final_distance:.3f}m | Avg Min Dist: {avg_min_distance:.3f}m")
        print(f"     Max Fingers: {avg_max_fingers:.1f} | Stage Rates: S1={stage1_rate:.1%}, S2={stage2_rate:.1%}, S3={stage3_rate:.1%}")

        env_test.close()

    # Overall Phase 1 analysis
    print(f"\n{'='*80}")
    print("📈 PHASE 1 COMPREHENSIVE TEST RESULTS")
    print(f"{'='*80}")

    total_episodes = len(all_results)
    total_successes = sum(r['success'] for r in all_results)
    overall_success_rate = total_successes / total_episodes

    overall_avg_final_distance = np.mean([r['final_distance'] for r in all_results])
    overall_avg_min_distance = np.mean([r['min_distance'] for r in all_results])
    overall_avg_max_fingers = np.mean([r['max_fingers'] for r in all_results])
    overall_avg_reward = np.mean([r['reward'] for r in all_results])
    overall_stage1_rate = np.mean([r['achieved_stage1'] for r in all_results])
    overall_stage2_rate = np.mean([r['achieved_stage2'] for r in all_results])
    overall_stage3_rate = np.mean([r['achieved_stage3'] for r in all_results])

    print(f"🎯 Overall Success Rate: {total_successes}/{total_episodes} ({overall_success_rate:.1%})")
    print(f"📏 Average Final Distance: {overall_avg_final_distance:.3f}m")
    print(f"🎯 Average Best Approach: {overall_avg_min_distance:.3f}m")
    print(f"👆 Average Max Fingers: {overall_avg_max_fingers:.1f}")
    print(f"🏆 Average Episode Reward: {overall_avg_reward:.1f}")
    print(f"\n🎪 PHASE 1 STAGE ACHIEVEMENTS:")
    print(f"   Stage 1 (≤0.15m): {overall_stage1_rate:.1%}")
    print(f"   Stage 2 (≤0.1m + 1 finger): {overall_stage2_rate:.1%}")
    print(f"   Stage 3 (≤0.1m + 2 fingers): {overall_stage3_rate:.1%}")

    # Scenario breakdown
    print(f"\n📊 SCENARIO BREAKDOWN:")
    for scenario_name in scenarios.keys():
        scenario_data = [r for r in all_results if r['scenario'] == scenario_name]
        if scenario_data:
            s_successes = sum(r['success'] for r in scenario_data)
            s_rate = s_successes / len(scenario_data)
            s_avg_dist = np.mean([r['final_distance'] for r in scenario_data])
            print(f"   {scenario_name:15} | Success: {s_rate:.1%} | Avg Dist: {s_avg_dist:.3f}m")

    # Phase 1 assessment
    print(f"\n🎯 PHASE 1 ASSESSMENT:")
    if overall_success_rate >= 0.2:
        print("✅ SUCCESS RATE: Exceeds Phase 1 target (≥20%)")
    elif overall_success_rate >= 0.1:
        print("🟨 SUCCESS RATE: Moderate progress toward Phase 1 (≥10%)")
    else:
        print("❌ SUCCESS RATE: Below Phase 1 expectations (<10%)")

    if overall_avg_min_distance <= 0.3:
        print("✅ APPROACH: Good distance learning")
    elif overall_avg_min_distance <= 0.5:
        print("🟨 APPROACH: Moderate distance learning")
    else:
        print("❌ APPROACH: Poor distance learning")

    if overall_avg_max_fingers >= 1.0:
        print("✅ TACTILE: Learning finger engagement")
    elif overall_avg_max_fingers >= 0.5:
        print("🟨 TACTILE: Some finger activation")
    else:
        print("❌ TACTILE: Poor finger activation")

    if overall_stage1_rate >= 0.5:
        print("✅ STAGE 1: Good basic approach (≥50%)")
    elif overall_stage1_rate >= 0.3:
        print("🟨 STAGE 1: Moderate approach (≥30%)")
    else:
        print("❌ STAGE 1: Poor basic approach (<30%)")

    print(f"\n📁 All test results saved to: {test_dir}")
    print("="*80)

    return all_results


def main():
    """Main testing function"""

    # Parse command line arguments
    visualize = True
    episodes_per_scenario = 3
    specific_model = None

    if len(sys.argv) > 1:
        for arg in sys.argv[1:]:
            if arg == "--no-vis":
                visualize = False
            elif arg.startswith("--episodes="):
                episodes_per_scenario = int(arg.split("=")[1])
            elif arg.startswith("--model="):
                specific_model = Path(arg.split("=")[1])
            elif arg == "--help":
                print("Usage: python test_simplified_phase1_model.py [OPTIONS]")
                print("Options:")
                print("  --no-vis          : Disable visualization")
                print("  --episodes=N      : Episodes per scenario (default: 3)")
                print("  --model=PATH      : Specific model path to test")
                print("  --help           : Show this help")
                return

    print("🚀 SC-1 Simplified Phase 1 Model Tester")
    print(f"   Visualization: {'ON' if visualize else 'OFF'}")
    print(f"   Episodes per scenario: {episodes_per_scenario}")
    print()

    # Find model
    if specific_model and specific_model.exists():
        model_path = specific_model
        print(f"✅ Using specified model: {model_path}")
    else:
        model_path = find_latest_model()
        if model_path is None:
            return

    # Run Phase 1 tests
    results = run_phase1_tests(
        model_path=model_path,
        visualize=visualize,
        episodes_per_scenario=episodes_per_scenario
    )

    print("\n🎉 Phase 1 testing completed!")


if __name__ == "__main__":
    main()