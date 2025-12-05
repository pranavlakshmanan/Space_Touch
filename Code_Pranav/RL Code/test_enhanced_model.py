#!/usr/bin/env python3
"""Test the Enhanced V2 model with correct observation space"""

import sys
import numpy as np
import pybullet as p
import pybullet_data
import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import VecEnv
import os
from pathlib import Path
from datetime import datetime
import time

# Fix compatibility
import collections.abc
import collections
collections.Mapping = collections.abc.Mapping
collections.MutableMapping = collections.abc.MutableMapping
collections.Sequence = collections.abc.Sequence
collections.MutableSequence = collections.abc.MutableSequence


class TendonController:
    """Simplified tendon controller for testing"""
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


class EnhancedTestEnv(VecEnv):
    """Test environment that matches Enhanced V2 observation space (26D)"""

    def __init__(self, num_envs=1, vis=True, max_steps=1000,
                 urdf_hand="/home/pralak/Space_Touch/examples/allegro_hand_description/allegro_hand_description_left_digit_fixed.urdf"):

        self.num_envs = num_envs
        self.vis = vis
        self.max_steps = max_steps
        self.urdf_hand = urdf_hand
        self.is_testing = True

        self.sim_freq = 240.0
        self._init_pybullet()

        self.hand = None
        self.target_sphere = None
        self.tendon_controller = None

        action_dim = 10
        self.action_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(action_dim,), dtype=np.float32)

        # 26D observation space to match Enhanced V2: base_pos(3) + target_pos(3) + base_vel(3) + finger_positions(12) + binary_tactile(4) + inside_hull(1)
        obs_dim = 26
        self.observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32)

        super().__init__(num_envs, self.observation_space, self.action_space)

        self.step_counts = np.zeros(num_envs, dtype=np.int32)
        self.episode_rewards = np.zeros(num_envs)
        self.episode_lengths = np.zeros(num_envs, dtype=np.int32)
        self.prev_actions = np.zeros((num_envs, action_dim), dtype=np.float32)

        self.reset()

    def _init_pybullet(self):
        if self.vis:
            try:
                self.client_id = p.connect(p.GUI)
                print("✅ GUI mode enabled")
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

    def _setup_simulation(self):
        try:
            # Clean up existing bodies
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
                # Fallback simple hand
                hand_collision = p.createCollisionShape(p.GEOM_BOX, halfExtents=[0.1, 0.05, 0.02])
                hand_visual = p.createVisualShape(p.GEOM_BOX, halfExtents=[0.1, 0.05, 0.02],
                                                rgbaColor=[0.8, 0.6, 0.4, 1])
                self.hand = p.createMultiBody(baseMass=1.0, baseCollisionShapeIndex=hand_collision,
                                            baseVisualShapeIndex=hand_visual, basePosition=[0, 0, 0.2])

            # Create target
            target_collision = p.createCollisionShape(p.GEOM_SPHERE, radius=0.02)
            target_visual = p.createVisualShape(p.GEOM_SPHERE, radius=0.02, rgbaColor=[1, 0, 0, 1])
            target_pos = np.array([0.25, 0.15, 0.35])

            self.target_sphere = p.createMultiBody(baseMass=0.5, baseCollisionShapeIndex=target_collision,
                                                 baseVisualShapeIndex=target_visual, basePosition=target_pos)

            # Setup tendon controller
            joint_inds, joint_names = [], []
            num_joints = p.getNumJoints(self.hand)
            for i in range(num_joints):
                joint_info = p.getJointInfo(self.hand, i)
                joint_type = joint_info[2]
                if joint_type in (p.JOINT_REVOLUTE, p.JOINT_PRISMATIC):
                    joint_inds.append(i)
                    joint_names.append(joint_info[1].decode())

            self.tendon_controller = TendonController(self.hand, joint_names, joint_inds)

            # Setup fingertip links for tactile sensing
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

    def _compute_grasp_convex_hull(self):
        """Simplified convex hull check"""
        try:
            finger_positions = self._get_finger_positions().reshape(4, 3)
            base_pos, _ = p.getBasePositionAndOrientation(self.hand)
            target_pos, _ = p.getBasePositionAndOrientation(self.target_sphere)

            # Simple distance-based approximation instead of full convex hull
            distances = [np.linalg.norm(finger_positions[i] - np.array(target_pos)) for i in range(4)]
            inside = min(distances) < 0.05  # If any finger is very close
            return inside
        except:
            return False

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

            # Add convex hull information (26th dimension)
            inside_hull = self._compute_grasp_convex_hull()
            inside_hull_flag = np.array([1.0 if inside_hull else 0.0])

            obs = np.concatenate([
                base_pos,           # 3D
                target_pos,         # 3D
                base_vel,           # 3D
                finger_positions,   # 12D
                binary_tactile,     # 4D
                inside_hull_flag,   # 1D - This makes it 26D total
            ])

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

            # Get state
            base_pos, _ = p.getBasePositionAndOrientation(self.hand)
            target_pos, _ = p.getBasePositionAndOrientation(self.target_sphere)
            base_pos = np.array(base_pos)
            target_pos = np.array(target_pos)
            distance = np.linalg.norm(base_pos - target_pos)

            binary_tactile = self._get_binary_tactile_feedback()
            num_active_fingers = np.sum(binary_tactile)
            inside_hull = self._compute_grasp_convex_hull()

            # Simple reward calculation
            distance_reward = -distance * 10.0  # Penalty for distance
            success = distance < 0.1 and num_active_fingers >= 2
            success_bonus = 100.0 if success else 0.0
            tactile_bonus = num_active_fingers * 5.0
            hull_bonus = 20.0 if inside_hull else 0.0

            total_reward = distance_reward + success_bonus + tactile_bonus + hull_bonus
            self.episode_rewards[0] += total_reward

            # Termination
            dones = np.array([
                self.step_counts[0] >= self.max_steps or success or
                distance > 2.0 or base_pos[2] < 0.05
            ])

            infos = [{
                "success": success,
                "distance": distance,
                "num_active_fingers": num_active_fingers,
                "inside_hull": inside_hull,
                "reward": total_reward
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
            self._setup_simulation()
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


def test_enhanced_model(visualize=True, episodes=5):
    """Test the Enhanced V2 model"""

    print("🚀 TESTING ENHANCED V2 MODEL")
    print("=" * 50)

    # Find the model
    model_path = Path('SC1_Training_Runs/Run_20251106_120937_SC1_Enhanced_V2/checkpoints/checkpoint_50000.zip')

    if not model_path.exists():
        print("❌ Model not found!")
        return

    print(f"📂 Loading: {model_path.name}")

    try:
        # Load model with compatible environment
        env_dummy = EnhancedTestEnv(vis=False)
        model = PPO.load(str(model_path), env=env_dummy)
        env_dummy.close()
        print("✅ Model loaded successfully!")
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        return

    # Run tests
    print(f"\n🧪 Running {episodes} test episodes...")

    env_test = EnhancedTestEnv(vis=visualize)
    results = []

    for episode in range(episodes):
        print(f"  Episode {episode+1}/{episodes}... ", end="")

        obs = env_test.reset()
        episode_reward = 0
        episode_steps = 0
        max_fingers = 0
        min_distance = float('inf')
        hull_achieved = False

        while episode_steps < 1000:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, info = env_test.step(action)
            episode_reward += reward[0]
            episode_steps += 1

            # Track metrics
            fingers = info[0].get('num_active_fingers', 0)
            distance = info[0].get('distance', float('inf'))
            inside_hull = info[0].get('inside_hull', False)

            if fingers > max_fingers:
                max_fingers = fingers
            if distance < min_distance:
                min_distance = distance
            if inside_hull:
                hull_achieved = True

            if done[0]:
                break

            # Add visualization delay
            if visualize:
                time.sleep(0.02)

        # Get final results
        final_distance = info[0].get('distance', float('inf'))
        success = info[0].get('success', False)
        final_fingers = info[0].get('num_active_fingers', 0)

        results.append({
            'episode': episode + 1,
            'success': success,
            'final_distance': final_distance,
            'min_distance': min_distance,
            'final_fingers': final_fingers,
            'max_fingers': max_fingers,
            'hull_achieved': hull_achieved,
            'reward': episode_reward,
            'steps': episode_steps
        })

        status = "✅ SUCCESS" if success else "❌ FAILED"
        hull_status = "🎯 HULL" if hull_achieved else ""
        print(f"{status} {hull_status} | Dist: {final_distance:.3f}m (min: {min_distance:.3f}) | Fingers: {final_fingers} (max: {max_fingers}) | Steps: {episode_steps}")

    env_test.close()

    # Analysis
    print(f"\n📊 TEST RESULTS SUMMARY:")
    successes = sum(r['success'] for r in results)
    success_rate = successes / episodes
    avg_final_distance = np.mean([r['final_distance'] for r in results])
    avg_min_distance = np.mean([r['min_distance'] for r in results])
    avg_final_fingers = np.mean([r['final_fingers'] for r in results])
    avg_max_fingers = np.mean([r['max_fingers'] for r in results])
    hull_rate = np.mean([r['hull_achieved'] for r in results])
    avg_reward = np.mean([r['reward'] for r in results])

    print(f"   Success Rate: {successes}/{episodes} ({success_rate:.1%})")
    print(f"   Final Distance: {avg_final_distance:.3f}m (avg)")
    print(f"   Best Distance: {avg_min_distance:.3f}m (avg)")
    print(f"   Final Fingers: {avg_final_fingers:.1f} (avg)")
    print(f"   Max Fingers: {avg_max_fingers:.1f} (avg)")
    print(f"   Hull Achievement: {hull_rate:.1%}")
    print(f"   Average Reward: {avg_reward:.1f}")

    print(f"\n🎯 PERFORMANCE ASSESSMENT:")
    if success_rate >= 0.2:
        print("✅ SUCCESS RATE: Good progress (≥20%)")
    elif success_rate >= 0.1:
        print("🟨 SUCCESS RATE: Moderate progress (≥10%)")
    else:
        print("❌ SUCCESS RATE: Needs improvement (<10%)")

    if avg_min_distance <= 0.15:
        print("✅ APPROACH: Learning to get close")
    else:
        print("❌ APPROACH: Needs better distance learning")

    if avg_max_fingers >= 1.0:
        print("✅ TACTILE: Learning finger engagement")
    else:
        print("❌ TACTILE: Poor finger activation")

    print("=" * 50)

    return results


if __name__ == "__main__":
    # Parse arguments
    visualize = "--no-vis" not in sys.argv
    episodes = 5
    if "--episodes" in sys.argv:
        idx = sys.argv.index("--episodes")
        if idx + 1 < len(sys.argv):
            episodes = int(sys.argv[idx + 1])

    test_enhanced_model(visualize=visualize, episodes=episodes)