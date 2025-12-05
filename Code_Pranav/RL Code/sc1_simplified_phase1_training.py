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

# Fix for attrdict Python 3.13 compatibility
import collections.abc
import collections
collections.Mapping = collections.abc.Mapping
collections.MutableMapping = collections.abc.MutableMapping
collections.Sequence = collections.abc.Sequence
collections.MutableSequence = collections.abc.MutableSequence


class DataLogger:
    """SIMPLIFIED data logger - only essential fields"""

    def __init__(self, log_dir):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)

        # SIMPLIFIED: Only track essential metrics
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
    """SIMPLIFIED tendon controller"""

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


class WandBCallback(BaseCallback):
    """SIMPLIFIED WandB callback"""

    def __init__(self, data_logger, log_freq=100, verbose=0):
        super(WandBCallback, self).__init__(verbose)
        self.data_logger = data_logger
        self.log_freq = log_freq

        # Simplified tracking
        self.episode_rewards = []
        self.episode_lengths = []
        self.episode_successes = []
        self.recent_distances = []
        self.recent_active_fingers = []

    def _on_step(self) -> bool:
        if self.locals.get('infos'):
            for info in self.locals['infos']:
                if 'episode' in info:
                    ep_reward = info['episode']['r']
                    ep_length = info['episode']['l']
                    self.episode_rewards.append(ep_reward)
                    self.episode_lengths.append(ep_length)
                    self.data_logger.new_episode()

                    wandb.log({
                        'episode/reward': ep_reward,
                        'episode/length': ep_length,
                    }, step=self.num_timesteps)

                    if len(self.episode_rewards) >= 100:
                        wandb.log({
                            'episode/reward_mean_100': np.mean(self.episode_rewards[-100:]),
                        }, step=self.num_timesteps)

                if 'distance' in info:
                    self.recent_distances.append(info['distance'])

                if 'success' in info and 'episode' in info:
                    self.episode_successes.append(float(info['success']))

                if 'num_active_fingers' in info:
                    self.recent_active_fingers.append(info['num_active_fingers'])

        # Log at intervals
        if self.num_timesteps % self.log_freq == 0:
            log_dict = {}

            if self.recent_distances:
                log_dict['metrics/distance_mean'] = np.mean(self.recent_distances)
                log_dict['metrics/distance_min'] = np.min(self.recent_distances)
                self.recent_distances = []

            if self.recent_active_fingers:
                log_dict['tactile/avg_active_fingers'] = np.mean(self.recent_active_fingers)
                self.recent_active_fingers = []

            if self.episode_successes:
                recent = self.episode_successes[-100:] if len(self.episode_successes) >= 100 else self.episode_successes
                log_dict['performance/success_rate'] = np.mean(recent)

            if log_dict:
                wandb.log(log_dict, step=self.num_timesteps)

        return True


class TendonAllegroReachingEnv(VecEnv):
    """SIMPLIFIED environment focused on basic approach learning"""

    def __init__(self,
                 num_envs=1,
                 vis=False,
                 max_steps=1000,  # INCREASED from 500 - give agent more time
                 urdf_hand="/home/pralak/Space_Touch/examples/allegro_hand_description/allegro_hand_description_left_digit_fixed.urdf",
                 target_range=0.3,
                 control_smoothing=False,  # DISABLED initially to reduce complexity
                 filter_cutoff=15.0):

        self.num_envs = num_envs
        self.vis = vis
        self.max_steps = max_steps
        self.urdf_hand = urdf_hand
        self.target_range = target_range
        self.control_smoothing = control_smoothing

        # ADD: Test flag for proper reset behavior
        self.is_testing = False

        # ADD: Track total training timesteps for curriculum learning
        self.training_timesteps = 0

        self.target_pos = np.array([0.25, 0.15, 0.35])

        self.sim_freq = 240.0
        self._init_pybullet()

        self.hand = None
        self.target_sphere = None
        self.tendon_controller = None
        self.hand_spawned = False

        # Action space: 6 DOF base movement + 4 tendon forces
        action_dim = 10
        self.action_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(action_dim,), dtype=np.float32)

        # SIMPLIFIED: Observation: base_pos(3) + target_pos(3) + base_vel(3) + finger_positions(12) + binary_tactile(4)
        # REMOVED: inside_hull flag (add back later)
        obs_dim = 3 + 3 + 3 + 12 + 4  # = 25 (was 26)
        self.observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32)

        super().__init__(num_envs, self.observation_space, self.action_space)

        self.step_counts = np.zeros(num_envs, dtype=np.int32)
        self.episode_rewards = np.zeros(num_envs)
        self.episode_lengths = np.zeros(num_envs, dtype=np.int32)

        # Track previous actions
        self.prev_actions = np.zeros((num_envs, action_dim), dtype=np.float32)

        # Track previous distance for delta reward calculation
        self.prev_distance = np.zeros(num_envs, dtype=np.float32)

        self.data_logger = None

        self.reset()

    def _init_pybullet(self):
        """Initialize PyBullet connection"""
        self.client_id = p.connect(p.GUI if self.vis else p.DIRECT)

        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.setGravity(0, 0, -9.81)
        p.setRealTimeSimulation(0)
        p.setTimeStep(1.0/self.sim_freq)

        p.loadURDF("plane.urdf", [0, 0, 0], useFixedBase=True)

    def set_data_logger(self, data_logger):
        """Set the data logger for this environment"""
        self.data_logger = data_logger

    def set_test_mode(self, is_testing=True):
        """Set testing mode flag"""
        self.is_testing = is_testing

    def _setup_simulation(self):
        """Setup the simulation environment"""
        try:
            # Always remove and recreate PyBullet bodies during testing
            if self.is_testing or self.hand_spawned:
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

            # CURRICULUM: Static target first, then add motion
            target_collision = p.createCollisionShape(p.GEOM_SPHERE, radius=0.02)
            target_visual = p.createVisualShape(p.GEOM_SPHERE, radius=0.02, rgbaColor=[1, 0, 0, 1])

            target_position = self.target_pos.copy()

            if hasattr(self, 'training_timesteps'):
                progress = self.training_timesteps / 500000.0
            else:
                progress = 0.0

            # Phase 1 (0-50K steps): Completely static
            if progress < 0.1:
                initial_velocity = [0, 0, 0]
                initial_angular_velocity = [0, 0, 0]
                if not hasattr(self, '_phase1_printed'):
                    print("📍 PHASE 1: Static target training")
                    self._phase1_printed = True

            # Phase 2 (50K-200K steps): Small random velocity
            elif progress < 0.4:
                initial_velocity = np.random.uniform(-0.01, 0.01, 3).tolist()
                initial_angular_velocity = [0, 0, 0]
                if not hasattr(self, '_phase2_printed'):
                    print("📍 PHASE 2: Slow moving target training")
                    self._phase2_printed = True

            # Phase 3 (200K+ steps): Full dynamics
            else:
                initial_velocity = np.random.uniform(-0.03, 0.03, 3).tolist()
                initial_angular_velocity = np.random.uniform(-0.1, 0.1, 3).tolist()
                if not hasattr(self, '_phase3_printed'):
                    print("📍 PHASE 3: Dynamic target training")
                    self._phase3_printed = True

            self.target_sphere = p.createMultiBody(
                baseMass=0.5,
                baseCollisionShapeIndex=target_collision,
                baseVisualShapeIndex=target_visual,
                basePosition=target_position
            )

            p.resetBaseVelocity(
                self.target_sphere,
                linearVelocity=initial_velocity,
                angularVelocity=initial_angular_velocity
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
        """Get current observation - SIMPLIFIED without convex hull"""
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

            # REMOVED: convex hull computation from observation
            obs = np.concatenate([
                base_pos,           # 3D
                target_pos,         # 3D
                base_vel,           # 3D
                finger_positions,   # 12D
                binary_tactile,     # 4D
                # REMOVED: inside_hull_flag
            ])

            return np.expand_dims(obs.astype(np.float32), axis=0)

        except Exception as e:
            print(f"Error getting observation: {e}")
            return np.zeros((1, self.observation_space.shape[0]), dtype=np.float32)

    def step_wait(self):
        """Execute one step - SIMPLIFIED REWARD FUNCTION"""
        # Increment global training counter
        if not self.is_testing:
            self.training_timesteps += 1

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

            # ============================================================================
            # SIMPLIFIED REWARD FUNCTION - PHASE 1: Learn Basic Approach
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

            # ============ TERMINATION CONDITIONS ============
            # More lenient - give agent time to learn
            dones = np.array([
                self.step_counts[0] >= self.max_steps or
                success or
                distance > 3.0 or  # Increased from 2.0
                base_pos[2] < 0.01  # Only fail if completely fallen
            ])

            # Update previous actions for next step
            self.prev_actions[0] = actions.copy()
            self.episode_rewards[0] += total_reward

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


def main(argv):
    """Main training function - SIMPLIFIED"""

    run_number = argv[1] if len(argv) > 1 else "1"
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    base_training_dir = Path("./SC1_Training_Runs/")
    base_training_dir.mkdir(parents=True, exist_ok=True)

    log_dir = base_training_dir / f"Run_{timestamp}_SC1_Simplified_Phase1"
    log_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print(f"SC-1 SIMPLIFIED PHASE 1 TRAINING - RUN #{run_number}")
    print("=" * 80)

    try:
        # Initialize WandB
        wandb.init(
            project="space-touch-sc1-simplified",
            name=f"SC1_Simplified_Phase1_run{run_number}",
            config={
                "algorithm": "PPO",
                "phase": "1_basic_approach",
                "total_timesteps": 500000,
                "checkpoint_frequency": 100000,

                # SIMPLIFIED HYPERPARAMETERS
                "learning_rate": 3e-5,
                "n_steps": 8192,
                "batch_size": 128,
                "max_grad_norm": 0.5,
                "n_epochs": 10,
                "gamma": 0.99,

                # ENVIRONMENT CONFIG
                "max_episode_steps": 1000,
                "success_threshold_easy": 0.15,
                "success_threshold_hard": 0.1,
                "target_physics": "curriculum_static_to_moving",
                "control_smoothing": False,

                # REWARD CONFIG
                "reward_components": "distance_progress + staged_success + curriculum_tactile",
                "distance_reward_weight": 20.0,
                "success_bonus_stage1": 10.0,
                "success_bonus_stage2": 20.0,
                "success_bonus_stage3": 50.0,

                # OBSERVATION
                "observation_space": "25D (removed convex hull)",
                "action_space": "10D (6 base + 4 tendons)",

                "notes": "SIMPLIFIED: Focus on basic approach first, curriculum learning, staged success"
            },
            tags=["simplified", "phase1", "curriculum", "stable-learning"]
        )

        data_logger = DataLogger(log_dir)

        print("\nCreating simplified environment...")

        # Base environment configuration
        base_env_kwargs = {
            "vis": False,
            "control_smoothing": False,  # Disabled for Phase 1 training
            "filter_cutoff": 15.0
        }

        env = TendonAllegroReachingEnv(**base_env_kwargs)
        env.set_data_logger(data_logger)

        # DISABLE VecNormalize for Phase 1 - adds complexity
        # env = VecNormalize(env, norm_obs=True, norm_reward=True, clip_obs=10.0)

        # Setup callbacks
        wandb_callback = WandBCallback(data_logger, log_freq=100)

        print("Setting up simplified PPO model...")

        if torch.cuda.is_available():
            device = "cuda"
            print(f"✓ GPU acceleration enabled: {torch.cuda.get_device_name(0)}")
        else:
            device = "cpu"
            print("⚠ No GPU found, using CPU")

        model = PPO(
            "MlpPolicy",
            env,
            verbose=1,
            # SIMPLIFIED HYPERPARAMETERS FOR STABLE LEARNING
            n_steps=8192,              # DOUBLED for better credit assignment
            learning_rate=3e-5,        # REDUCED by 3x for stability
            n_epochs=10,
            batch_size=128,
            gamma=0.99,
            gae_lambda=0.95,
            clip_range=0.2,
            ent_coef=0.01,
            vf_coef=0.5,
            max_grad_norm=0.5,         # INCREASED back to 0.5 (was too restrictive)
            device=device
        )

        print("\n" + "=" * 80)
        print("SIMPLIFIED SC-1 PHASE 1 TRAINING CONFIGURATION")
        print("=" * 80)
        print(f"Log Directory: {log_dir}")
        print(f"WandB Project: space-touch-sc1-simplified")
        print("Phase 1: Focus on basic approach learning")
        print("Total Training Timesteps: 500,000")
        print("Checkpoint Frequency: Every 100,000 timesteps")
        print("Action Space: 10D (6 base movement + 4 tendon forces)")
        print("Observation Space: 25D (removed convex hull complexity)")
        print("\nSIMPLIFIED FEATURES:")
        print("  ✓ Curriculum learning (static -> slow -> dynamic target)")
        print("  ✓ Staged success rewards (0.15m -> 0.1m -> 0.1m+2fingers)")
        print("  ✓ Distance delta rewards (20x weight)")
        print("  ✓ Progressive tactile engagement (after 20% training)")
        print("  ✓ Increased episode length (1000 steps)")
        print("  ✓ Control smoothing disabled (reduce complexity)")
        print("  ✓ Simplified observation space (no convex hull)")
        print("=" * 80)

        # Training loop
        TIMESTEPS_PER_ITERATION = 100000  # Save every 100K
        TOTAL_TIMESTEPS = 500000          # Train for 500K total

        current_timestep = 0
        iteration = 0

        print(f"\nStarting Phase 1 simplified training")
        print(f"Target: 20-30% success rate by 500K steps")

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
                callback=wandb_callback,
                log_interval=10,
                reset_num_timesteps=False,
                progress_bar=True
            )

            current_timestep = next_checkpoint

            # Save iteration checkpoint
            iteration_checkpoint = log_dir / f"checkpoint_{current_timestep}"
            model.save(str(iteration_checkpoint))
            print(f"✓ Iteration {iteration} complete! Checkpoint saved: {iteration_checkpoint}.zip")

            if current_timestep >= TOTAL_TIMESTEPS:
                break

        # Save final model
        final_model_path = log_dir / f"sc1_simplified_phase1_final_model_run{run_number}_{timestamp}"
        model.save(str(final_model_path))
        print(f"\n✓ Training Complete! Final model saved to {final_model_path}.zip")

        # SIMPLIFIED TESTING - Just verify basic functionality
        print(f"\nRunning simplified verification tests...")

        test_data_logger = DataLogger(log_dir / "test_data")
        env_test = TendonAllegroReachingEnv(**base_env_kwargs)
        env_test.set_test_mode(True)
        env_test.set_data_logger(test_data_logger)

        test_results = []

        for test_episode in range(20):
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

            distance = info[0].get('distance', float('inf'))
            success = info[0].get('success', False)
            num_fingers = info[0].get('num_active_fingers', 0)

            test_results.append({
                'distance': distance,
                'success': success,
                'reward': episode_reward,
                'steps': episode_steps,
                'num_fingers': num_fingers
            })

            status = "✓ SUCCESS" if success else "✗ FAILED"
            print(f"  Test {test_episode+1:2d}: {status} | Dist={distance:.3f}m | Fingers={num_fingers} | Reward={episode_reward:.1f}")

        env_test.close()

        # Calculate statistics
        successes = sum(r['success'] for r in test_results)
        avg_distance = np.mean([r['distance'] for r in test_results])
        avg_fingers = np.mean([r['num_fingers'] for r in test_results])

        print(f"\n{'='*80}")
        print(f"SIMPLIFIED TEST RESULTS")
        print(f"{'='*80}")
        print(f"Success Rate: {successes}/20 ({successes/20*100:.1f}%)")
        print(f"Average Distance: {avg_distance:.3f}m")
        print(f"Average Active Fingers: {avg_fingers:.1f}")
        print(f"{'='*80}")

        # Log to WandB
        wandb.log({
            "test/success_rate": successes/20,
            "test/avg_distance": avg_distance,
            "test/avg_active_fingers": avg_fingers,
        })

        # Check if ready for next phase
        print(f"\n{'='*80}")
        print("PHASE 1 COMPLETION CHECK")
        print(f"{'='*80}")

        if successes >= 4:  # 20% success rate
            print("✓ Phase 1 Complete!")
            print("  Next steps:")
            print("  1. Re-enable control smoothing")
            print("  2. Add back convex hull observations")
            print("  3. Add control smoothness penalties")
            print("  4. Continue training for 500K more steps")
        else:
            print("⚠ Phase 1 Incomplete")
            print("  Recommendations:")
            print("  1. Reduce learning rate to 1e-5")
            print("  2. Increase n_steps to 16384")
            print("  3. Train for another 500K steps")
            print("  4. Check if target is too far (try 0.2m initial distance)")

        print(f"{'='*80}")

        # Save test data
        test_csv = test_data_logger.save_to_csv("sc1_simplified_test_data.csv")

        wandb.finish()

        print("\n" + "=" * 80)
        print("SC-1 SIMPLIFIED PHASE 1 TRAINING COMPLETE!")
        print("=" * 80)
        print(f"All results saved to: {log_dir}")
        print(f"Test data: {log_dir}/test_data/")
        print(f"\nTarget Metrics Achieved:")
        success_check = "✓" if successes >= 4 else "✗"
        print(f"  {success_check} Success Rate: {successes/20*100:.1f}% (target: >20%)")
        distance_check = "✓" if avg_distance < 0.3 else "✗"
        print(f"  {distance_check} Avg Distance: {avg_distance:.3f}m (target: <0.3m)")
        fingers_check = "✓" if avg_fingers > 0.5 else "✗"
        print(f"  {fingers_check} Avg Active Fingers: {avg_fingers:.1f} (target: >0.5)")
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


if __name__ == "__main__":
    main(sys.argv)