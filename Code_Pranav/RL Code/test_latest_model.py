#!/usr/bin/env python3
import time
import numpy as np
import cv2
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
from scipy.spatial import ConvexHull, Delaunay

# Fix for attrdict Python 3.13 compatibility
import collections.abc
import collections
collections.Mapping = collections.abc.Mapping
collections.MutableMapping = collections.abc.MutableMapping
collections.Sequence = collections.abc.Sequence
collections.MutableSequence = collections.abc.MutableSequence

# Import the simplified environment from the training script
class DataLogger:
    """Test data logger"""
    def __init__(self, log_dir):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)

        self.data = {
            'timestamp': [], 'step': [], 'episode': [],
            'base_pos_x': [], 'base_pos_y': [], 'base_pos_z': [],
            'target_x': [], 'target_y': [], 'target_z': [],
            'distance_to_target': [], 'reward': [],
            'distance_reward': [], 'success_bonus': [], 'tactile_reward': [],
            'success': [], 'num_active_fingers': [],
        }
        self.current_episode = 0
        self.global_step = 0

    def log_step(self, data_dict):
        self.global_step += 1
        for key, value in data_dict.items():
            if key in self.data:
                self.data[key].append(value)
        self.data['timestamp'].append(time.time())
        self.data['step'].append(self.global_step)
        self.data['episode'].append(self.current_episode)

    def new_episode(self):
        self.current_episode += 1

    def save_to_csv(self, filename="test_data.csv"):
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
    """Tendon controller for testing"""
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
        torques = np.zeros(len(self.joint_indices))
        for finger, normalized_force in tendon_forces.items():
            if finger not in self.finger_joints:
                continue
            actual_force = normalized_force * self.MAX_TENDON_FORCE
            joints = self.finger_joints[finger]
            for joint_idx in joints:
                moment_arm = 0.02
                torque = actual_force * moment_arm * self.TENDON_FORCE_GAIN
                joint_state = p.getJointState(self.hand_id, joint_idx)
                current_velocity = joint_state[1]
                damping_torque = -self.TENDON_DAMPING * current_velocity
                final_torque = torque + damping_torque
                idx_in_list = self.joint_indices.index(joint_idx)
                torques[idx_in_list] = final_torque
        return torques


class TendonAllegroTestEnv(VecEnv):
    """Test environment with visualization"""

    def __init__(self, num_envs=1, vis=True, max_steps=1000,
                 urdf_hand="/home/pralak/Space_Touch/examples/allegro_hand_description/allegro_hand_description_left_digit_fixed.urdf",
                 test_scenario=None):

        self.num_envs = num_envs
        self.vis = vis
        self.max_steps = max_steps
        self.urdf_hand = urdf_hand
        self.test_scenario = test_scenario
        self.is_testing = True
        self.training_timesteps = 0

        self.sim_freq = 240.0
        self._init_pybullet()

        self.hand = None
        self.target_sphere = None
        self.tendon_controller = None

        action_dim = 10
        self.action_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(action_dim,), dtype=np.float32)
        obs_dim = 25  # Simplified observation space
        self.observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32)

        super().__init__(num_envs, self.observation_space, self.action_space)

        self.step_counts = np.zeros(num_envs, dtype=np.int32)
        self.episode_rewards = np.zeros(num_envs)
        self.episode_lengths = np.zeros(num_envs, dtype=np.int32)
        self.prev_actions = np.zeros((num_envs, action_dim), dtype=np.float32)
        self.prev_distance = np.zeros(num_envs, dtype=np.float32)
        self.data_logger = None

        # Test scenarios
        self.test_scenarios = {
            "static_close": {"distance": 0.2, "target_vel": [0,0,0], "description": "Static target close"},
            "static_medium": {"distance": 0.3, "target_vel": [0,0,0], "description": "Static target medium"},
            "static_far": {"distance": 0.4, "target_vel": [0,0,0], "description": "Static target far"},
            "moving_slow": {"distance": 0.25, "target_vel": [0.02,0,0], "description": "Slow moving target"},
            "moving_medium": {"distance": 0.3, "target_vel": [0.05,0,0], "description": "Medium moving target"},
        }

        self.reset()

    def _init_pybullet(self):
        if self.vis:
            self.client_id = p.connect(p.GUI)
            p.configureDebugVisualizer(p.COV_ENABLE_GUI, 1)
        else:
            self.client_id = p.connect(p.DIRECT)

        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.setGravity(0, 0, -9.81)
        p.setRealTimeSimulation(0)
        p.setTimeStep(1.0/self.sim_freq)
        p.loadURDF("plane.urdf", [0, 0, 0], useFixedBase=True)

    def set_data_logger(self, data_logger):
        self.data_logger = data_logger

    def _setup_simulation(self):
        try:
            if self.hand is not None:
                try:
                    p.removeBody(self.hand)
                except:
                    pass
            if self.target_sphere is not None:
                try:
                    p.removeBody(self.target_sphere)
                except:
                    pass

            # Load hand
            if os.path.exists(self.urdf_hand):
                self.hand = p.loadURDF(self.urdf_hand, basePosition=[0, 0, 0.2],
                                     baseOrientation=p.getQuaternionFromEuler([0, 0, 0]),
                                     useFixedBase=False)
            else:
                hand_collision = p.createCollisionShape(p.GEOM_BOX, halfExtents=[0.1, 0.05, 0.02])
                hand_visual = p.createVisualShape(p.GEOM_BOX, halfExtents=[0.1, 0.05, 0.02],
                                                rgbaColor=[0.8, 0.6, 0.4, 1])
                self.hand = p.createMultiBody(baseMass=1.0, baseCollisionShapeIndex=hand_collision,
                                            baseVisualShapeIndex=hand_visual, basePosition=[0, 0, 0.2])

            # Setup target based on test scenario
            target_collision = p.createCollisionShape(p.GEOM_SPHERE, radius=0.02)
            target_visual = p.createVisualShape(p.GEOM_SPHERE, radius=0.02, rgbaColor=[1, 0, 0, 1])

            # Default target position
            target_pos = np.array([0.25, 0.15, 0.35])
            initial_velocity = [0, 0, 0]

            # Apply test scenario if specified
            if self.test_scenario and self.test_scenario in self.test_scenarios:
                scenario = self.test_scenarios[self.test_scenario]
                direction = target_pos / np.linalg.norm(target_pos)
                target_pos = direction * scenario["distance"]
                initial_velocity = scenario["target_vel"]
                print(f"Test scenario: {scenario['description']}")

            self.target_sphere = p.createMultiBody(baseMass=0.5, baseCollisionShapeIndex=target_collision,
                                                 baseVisualShapeIndex=target_visual, basePosition=target_pos)

            p.resetBaseVelocity(self.target_sphere, linearVelocity=initial_velocity, angularVelocity=[0,0,0])

            # Initialize tendon controller
            joint_inds, joint_names = [], []
            num_joints = p.getNumJoints(self.hand)
            for i in range(num_joints):
                joint_info = p.getJointInfo(self.hand, i)
                joint_type = joint_info[2]
                if joint_type in (p.JOINT_REVOLUTE, p.JOINT_PRISMATIC):
                    joint_inds.append(i)
                    joint_names.append(joint_info[1].decode())

            self.tendon_controller = TendonController(self.hand, joint_names, joint_inds)

            # Initialize fingertip links
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
        try:
            if self.hand is None:
                obs = np.zeros(self.observation_space.shape[0])
                return np.expand_dims(obs.astype(np.float32), axis=0)

            base_pos, _ = p.getBasePositionAndOrientation(self.hand)
            base_vel, _ = p.getBaseVelocity(self.hand)
            target_pos, _ = p.getBasePositionAndOrientation(self.target_sphere)

            base_pos = np.array(base_pos)
            base_vel = np.array(base_vel)
            target_pos = np.array(target_pos)

            finger_positions = self._get_finger_positions()
            binary_tactile = self._get_binary_tactile_feedback()

            obs = np.concatenate([base_pos, target_pos, base_vel, finger_positions, binary_tactile])
            return np.expand_dims(obs.astype(np.float32), axis=0)

        except Exception as e:
            print(f"Error getting observation: {e}")
            return np.zeros((1, self.observation_space.shape[0]), dtype=np.float32)

    def step_wait(self):
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
            p.resetBaseVelocity(self.hand, linearVelocity=linear_vel, angularVelocity=angular_vel)

            # Apply tendon forces
            tendon_forces = (tendon_actions + 1.0) / 2.0
            tendon_force_dict = {
                "index": tendon_forces[0], "middle": tendon_forces[1],
                "ring": tendon_forces[2], "thumb": tendon_forces[3]
            }

            torques = self.tendon_controller.compute_tendon_torques(tendon_force_dict)
            if len(torques) > 0:
                p.setJointMotorControlArray(bodyUniqueId=self.hand,
                                          jointIndices=self.tendon_controller.joint_indices,
                                          controlMode=p.TORQUE_CONTROL, forces=torques.tolist())

            p.stepSimulation()

            # Calculate rewards (same as training)
            base_pos, _ = p.getBasePositionAndOrientation(self.hand)
            target_pos, _ = p.getBasePositionAndOrientation(self.target_sphere)
            base_pos = np.array(base_pos)
            target_pos = np.array(target_pos)
            distance = np.linalg.norm(base_pos - target_pos)

            binary_tactile = self._get_binary_tactile_feedback()
            num_active_fingers = np.sum(binary_tactile)

            # Simplified reward calculation
            distance_delta = self.prev_distance[0] - distance
            distance_reward = 20.0 * distance_delta
            self.prev_distance[0] = distance

            success_bonus = 0.0
            success = False
            if distance < 0.15:
                success_bonus += 10.0
            if distance < 0.1 and num_active_fingers >= 1:
                success_bonus += 20.0
            if distance < 0.1 and num_active_fingers >= 2:
                success_bonus += 50.0
                success = True

            tactile_reward = 0.0
            if num_active_fingers >= 2:
                tactile_reward = 5.0
            elif num_active_fingers >= 1:
                tactile_reward = 2.0

            total_reward = distance_reward + success_bonus + tactile_reward
            self.episode_rewards[0] += total_reward

            # Termination conditions
            dones = np.array([
                self.step_counts[0] >= self.max_steps or success or
                distance > 3.0 or base_pos[2] < 0.01
            ])

            # Logging
            self.step_data = {
                "base_pos_x": base_pos[0], "base_pos_y": base_pos[1], "base_pos_z": base_pos[2],
                "target_x": target_pos[0], "target_y": target_pos[1], "target_z": target_pos[2],
                "distance_to_target": distance, "reward": total_reward,
                "distance_reward": distance_reward, "success_bonus": success_bonus,
                "tactile_reward": tactile_reward, "success": success,
                "num_active_fingers": num_active_fingers,
            }

            if self.data_logger is not None:
                self.data_logger.log_step(self.step_data)

            infos = [{
                "success": success, "distance": distance,
                "num_active_fingers": num_active_fingers,
                "reward_breakdown": {"distance": distance_reward, "success": success_bonus, "tactile": tactile_reward}
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
        if np.any(dones):
            if self.hand is not None:
                try:
                    p.removeBody(self.hand)
                except:
                    pass
                self.hand = None
            if self.target_sphere is not None:
                try:
                    p.removeBody(self.target_sphere)
                except:
                    pass
                self.target_sphere = None

            self._setup_simulation()

            if self.hand is not None:
                base_pos, _ = p.getBasePositionAndOrientation(self.hand)
                target_pos, _ = p.getBasePositionAndOrientation(self.target_sphere)
                initial_distance = np.linalg.norm(np.array(base_pos) - np.array(target_pos))
                self.prev_distance[0] = initial_distance

            self.step_counts[dones] = 0
            self.episode_rewards[dones] = 0
            self.episode_lengths[dones] = 0
            self.prev_actions[dones] = 0

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

    # VecEnv methods
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
    """Find the latest trained model"""
    base_dir = Path("./SC1_Training_Runs/")
    if not base_dir.exists():
        print("❌ No training runs found!")
        return None

    # Look for model files in all subdirectories
    model_files = []
    for run_dir in base_dir.glob("Run_*"):
        # Look for final models
        for model_file in run_dir.glob("*final*.zip"):
            model_files.append(model_file)
        # Look for checkpoints
        for model_file in run_dir.glob("checkpoint_*.zip"):
            model_files.append(model_file)

    if not model_files:
        print("❌ No model files found!")
        return None

    # Sort by modification time, newest first
    model_files.sort(key=lambda x: x.stat().st_mtime, reverse=True)
    latest_model = model_files[0]

    print(f"✅ Found latest model: {latest_model}")
    print(f"   Modified: {datetime.fromtimestamp(latest_model.stat().st_mtime)}")

    return latest_model


def create_test_plots(csv_file):
    """Create comprehensive test plots"""
    try:
        df = pd.read_csv(csv_file)
        print(f"Creating plots from {len(df)} data points...")

        plot_dir = csv_file.parent / "plots"
        plot_dir.mkdir(exist_ok=True)

        plt.style.use('default')

        # 1. Performance Overview
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('SC-1 Test Performance Overview', fontsize=16)

        # Success over episodes
        if 'episode' in df.columns:
            episode_success = df.groupby('episode')['success'].max()
            axes[0,0].plot(episode_success.index, episode_success.values, 'o-', alpha=0.7)
            axes[0,0].set_title('Success Rate by Episode')
            axes[0,0].set_ylabel('Success (0/1)')
            axes[0,0].grid(True, alpha=0.3)

        # Distance to target
        axes[0,1].plot(df['step'], df['distance_to_target'], alpha=0.7, linewidth=0.8)
        axes[0,1].axhline(y=0.1, color='red', linestyle='--', label='Success Threshold')
        axes[0,1].set_title('Distance to Target Over Time')
        axes[0,1].set_ylabel('Distance (m)')
        axes[0,1].legend()
        axes[0,1].grid(True, alpha=0.3)

        # Tactile engagement
        axes[1,0].plot(df['step'], df['num_active_fingers'], alpha=0.7, linewidth=0.8, color='green')
        axes[1,0].axhline(y=2, color='red', linestyle='--', label='Success Threshold')
        axes[1,0].set_title('Active Finger Count')
        axes[1,0].set_xlabel('Step')
        axes[1,0].set_ylabel('Number of Fingers')
        axes[1,0].legend()
        axes[1,0].grid(True, alpha=0.3)

        # Reward components
        axes[1,1].plot(df['step'], df['reward'], alpha=0.7, linewidth=0.8, label='Total')
        axes[1,1].plot(df['step'], df['distance_reward'], alpha=0.7, linewidth=0.8, label='Distance')
        axes[1,1].plot(df['step'], df['success_bonus'], alpha=0.7, linewidth=0.8, label='Success')
        axes[1,1].plot(df['step'], df['tactile_reward'], alpha=0.7, linewidth=0.8, label='Tactile')
        axes[1,1].set_title('Reward Components')
        axes[1,1].set_xlabel('Step')
        axes[1,1].set_ylabel('Reward')
        axes[1,1].legend()
        axes[1,1].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(plot_dir / 'test_performance_overview.png', dpi=150, bbox_inches='tight')
        plt.close()

        # 2. Trajectory Analysis
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        fig.suptitle('Hand Movement Trajectory Analysis', fontsize=16)

        # 3D trajectory
        ax = fig.add_subplot(1, 3, 1, projection='3d')
        ax.plot(df['base_pos_x'], df['base_pos_y'], df['base_pos_z'], alpha=0.7, linewidth=1)
        ax.scatter(df['target_x'].iloc[0], df['target_y'].iloc[0], df['target_z'].iloc[0],
                  color='red', s=100, label='Target')
        ax.set_title('3D Hand Trajectory')
        ax.set_xlabel('X (m)')
        ax.set_ylabel('Y (m)')
        ax.set_zlabel('Z (m)')
        ax.legend()

        # XY trajectory
        axes[1].plot(df['base_pos_x'], df['base_pos_y'], alpha=0.7, linewidth=1)
        axes[1].scatter(df['target_x'].iloc[0], df['target_y'].iloc[0], color='red', s=100, label='Target')
        axes[1].set_title('XY Trajectory (Top View)')
        axes[1].set_xlabel('X (m)')
        axes[1].set_ylabel('Y (m)')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        axes[1].axis('equal')

        # Distance over time with color coding
        colors = ['red' if not success else 'green' for success in df['success']]
        axes[2].scatter(df['step'], df['distance_to_target'], c=colors, alpha=0.6, s=1)
        axes[2].axhline(y=0.1, color='blue', linestyle='--', label='Success Threshold')
        axes[2].set_title('Distance Evolution (Red=Fail, Green=Success)')
        axes[2].set_xlabel('Step')
        axes[2].set_ylabel('Distance (m)')
        axes[2].legend()
        axes[2].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(plot_dir / 'trajectory_analysis.png', dpi=150, bbox_inches='tight')
        plt.close()

        print(f"✅ Test plots saved to: {plot_dir}")

    except Exception as e:
        print(f"❌ Error creating plots: {e}")
        import traceback
        traceback.print_exc()


def run_comprehensive_tests(model_path, visualize=True, episodes_per_scenario=5):
    """Run comprehensive tests on the model"""

    print("=" * 80)
    print("🧪 SC-1 MODEL COMPREHENSIVE TESTING")
    print("=" * 80)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    test_dir = Path(f"./SC1_Model_Tests/Test_{timestamp}")
    test_dir.mkdir(parents=True, exist_ok=True)

    # Load model
    try:
        print(f"📂 Loading model: {model_path}")
        # Create a dummy environment for model loading
        dummy_env = TendonAllegroTestEnv(vis=False)
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
    env_test = TendonAllegroTestEnv(vis=False)  # Create base environment
    scenarios = env_test.test_scenarios
    env_test.close()

    for scenario_name, scenario_config in scenarios.items():
        print(f"\n🎯 Testing scenario: {scenario_name}")
        print(f"   {scenario_config['description']}")

        # Create test environment for this scenario
        env_test = TendonAllegroTestEnv(vis=visualize, test_scenario=scenario_name)
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

            while episode_steps < 1000:
                action, _ = model.predict(obs, deterministic=True)
                obs, reward, done, info = env_test.step(action)
                episode_reward += reward[0]
                episode_steps += 1

                # Track metrics
                if info[0].get('num_active_fingers', 0) > max_fingers:
                    max_fingers = info[0].get('num_active_fingers', 0)
                if info[0].get('distance', float('inf')) < min_distance:
                    min_distance = info[0].get('distance', float('inf'))

                if done[0]:
                    break

                # Add small delay for visualization
                if visualize:
                    time.sleep(0.01)

            test_data_logger.new_episode()

            # Extract results
            distance = info[0].get('distance', float('inf'))
            success = info[0].get('success', False)
            num_fingers = info[0].get('num_active_fingers', 0)

            result = {
                'scenario': scenario_name,
                'episode': episode + 1,
                'final_distance': distance,
                'min_distance': min_distance,
                'success': success,
                'reward': episode_reward,
                'steps': episode_steps,
                'final_fingers': num_fingers,
                'max_fingers': max_fingers,
            }

            scenario_results.append(result)
            all_results.append(result)

            status = "✅ SUCCESS" if success else "❌ FAILED"
            print(f"{status} | Dist: {distance:.3f}m | Fingers: {num_fingers} | Steps: {episode_steps}")

        # Save scenario data and create plots
        test_csv = test_data_logger.save_to_csv(f"{scenario_name}_test_data.csv")
        create_test_plots(test_csv)

        # Scenario summary
        successes = sum(r['success'] for r in scenario_results)
        avg_distance = np.mean([r['final_distance'] for r in scenario_results])
        avg_fingers = np.mean([r['final_fingers'] for r in scenario_results])

        print(f"  📊 Scenario Results: {successes}/{len(scenario_results)} success ({successes/len(scenario_results):.1%})")
        print(f"     Avg distance: {avg_distance:.3f}m | Avg fingers: {avg_fingers:.1f}")

        env_test.close()

    # Overall analysis
    print(f"\n{'='*80}")
    print("📈 COMPREHENSIVE TEST RESULTS")
    print(f"{'='*80}")

    total_episodes = len(all_results)
    total_successes = sum(r['success'] for r in all_results)
    overall_success_rate = total_successes / total_episodes

    overall_avg_distance = np.mean([r['final_distance'] for r in all_results])
    overall_avg_fingers = np.mean([r['final_fingers'] for r in all_results])
    overall_avg_reward = np.mean([r['reward'] for r in all_results])

    print(f"🎯 Overall Success Rate: {total_successes}/{total_episodes} ({overall_success_rate:.1%})")
    print(f"📏 Average Final Distance: {overall_avg_distance:.3f}m")
    print(f"👆 Average Active Fingers: {overall_avg_fingers:.1f}")
    print(f"🏆 Average Episode Reward: {overall_avg_reward:.1f}")

    # Scenario breakdown
    print(f"\n📊 SCENARIO BREAKDOWN:")
    for scenario_name in scenarios.keys():
        scenario_data = [r for r in all_results if r['scenario'] == scenario_name]
        if scenario_data:
            s_successes = sum(r['success'] for r in scenario_data)
            s_rate = s_successes / len(scenario_data)
            s_avg_dist = np.mean([r['final_distance'] for r in scenario_data])
            print(f"   {scenario_name:15} | Success: {s_rate:.1%} | Avg Dist: {s_avg_dist:.3f}m")

    # Performance assessment
    print(f"\n🎯 PERFORMANCE ASSESSMENT:")
    if overall_success_rate >= 0.2:
        print("✅ SUCCESS RATE: Meets Phase 1 target (≥20%)")
    else:
        print("❌ SUCCESS RATE: Below Phase 1 target (<20%)")

    if overall_avg_distance <= 0.3:
        print("✅ DISTANCE: Meets target (≤0.3m)")
    else:
        print("❌ DISTANCE: Above target (>0.3m)")

    if overall_avg_fingers >= 0.5:
        print("✅ TACTILE: Meets target (≥0.5 avg fingers)")
    else:
        print("❌ TACTILE: Below target (<0.5 avg fingers)")

    print(f"\n📁 All test results saved to: {test_dir}")
    print("="*80)

    return all_results


def main():
    """Main testing function"""

    # Parse command line arguments
    visualize = True
    episodes_per_scenario = 3  # Default episodes per scenario

    if len(sys.argv) > 1:
        if "--no-vis" in sys.argv:
            visualize = False
        if "--episodes" in sys.argv:
            idx = sys.argv.index("--episodes")
            if idx + 1 < len(sys.argv):
                episodes_per_scenario = int(sys.argv[idx + 1])

    print("🚀 SC-1 Model Tester")
    print(f"   Visualization: {'ON' if visualize else 'OFF'}")
    print(f"   Episodes per scenario: {episodes_per_scenario}")
    print()

    # Find latest model
    model_path = find_latest_model()
    if model_path is None:
        return

    # Run tests
    results = run_comprehensive_tests(
        model_path=model_path,
        visualize=visualize,
        episodes_per_scenario=episodes_per_scenario
    )

    print("\n🎉 Testing completed!")


if __name__ == "__main__":
    main()