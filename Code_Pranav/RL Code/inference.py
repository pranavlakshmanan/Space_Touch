#!/usr/bin/env python3
import time
import numpy as np
import cv2
import pybullet as p
import pybullet_data
import tacto
import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import VecEnv
from types import SimpleNamespace
import argparse
import os
import glob


class AllegroHandTactileEnv(VecEnv):
    """
    Gym environment for Allegro hand with tactile sensors to grasp objects in zero gravity.
    Uses standard PyBullet functions instead of wrappers for stability.
    """
    
    def __init__(self, 
                 num_envs=1,
                 vis=True,
                 max_steps=1000,
                 urdf_hand="/home/pranav/Space_touch/examples/allegro_hand_description/allegro_hand_description_left_digit.urdf",
                 urdf_sphere="/home/pranav/Space_touch/examples/objects/sphere_small.urdf",
                 sensor_res=(60, 80),
                 render_sensor=False):
        
        self.vis = vis
        self.max_steps = max_steps
        self.sim_freq = 240
        self.urdf_hand = urdf_hand
        self.urdf_sphere = urdf_sphere
        self.sensor_res = sensor_res
        self.render_sensor = render_sensor
        
        self.tip_labels = {
            "joint_15.0_tip": "little", "joint_11.0_tip": "index", 
            "joint_7.0_tip": "middle", "joint_3.0_tip": "ring",
        }
        
        # Connect to PyBullet directly
        if self.vis: 
            p.connect(p.GUI)
        else: 
            p.connect(p.DIRECT)
        
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.setGravity(0, 0, 0)
        p.setRealTimeSimulation(0)
        p.loadURDF("plane.urdf", [0, 0, 0], useFixedBase=True)
        if self.vis: 
            p.resetDebugVisualizerCamera(0.8, 60, -30, [0, 0, 0.25])
        
        bg = cv2.imread("examples/conf/bg_digit_240_320.jpg")
        self.sensor = tacto.Sensor(sensor_res[1], sensor_res[0],
                                  background=bg,
                                  config_path=tacto.get_digit_config_path())
        
        self.hand = None
        self.sphere = None
        self.joint_indices = []
        
        NUM_ACTUATED_JOINTS = 16
        NUM_SENSORS = 4
        
        action_dim = 6 + NUM_ACTUATED_JOINTS
        self.action_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(action_dim,), dtype=np.float32)
        
        obs_shape = (NUM_SENSORS, sensor_res[0], sensor_res[1])
        self.observation_space = gym.spaces.Box(low=0, high=255, shape=obs_shape, dtype=np.uint8)
        
        super().__init__(num_envs, self.observation_space, self.action_space)
        
        self.step_counts = np.zeros(num_envs, dtype=np.int32)
        self.hold_counters = np.zeros(num_envs, dtype=np.int32)
        
        # Episode tracking
        self.episode_rewards = np.zeros(num_envs)
        self.episode_lengths = np.zeros(num_envs, dtype=np.int32)

    def _setup_simulation(self):
        """Sets up the simulation using standard PyBullet functions."""
        self.hand = p.loadURDF(self.urdf_hand, basePosition=[0, 0, 0.3], useFixedBase=False)
        
        self.joint_indices = []
        for i in range(p.getNumJoints(self.hand)):
            info = p.getJointInfo(self.hand, i)
            if info[2] in (p.JOINT_REVOLUTE, p.JOINT_PRISMATIC):
                self.joint_indices.append(i)

        self.sensor.bodies = []
        self.sensor.link_ids = []
        self.cam_order = []
        
        tip_entries = []
        for i in range(p.getNumJoints(self.hand)):
            jn = p.getJointInfo(self.hand, i)[1].decode()
            if jn.endswith("_tip") and jn in self.tip_labels:
                self.sensor.add_camera(self.hand, link_ids=[i])
                pos = p.getLinkState(self.hand, i)[0]
                tip_entries.append({"joint": jn, "x": pos[0], "cam_idx": len(tip_entries)})
        
        tip_entries.sort(key=lambda e: e["x"])
        self.cam_order = [e["cam_idx"] for e in tip_entries]
        
        sphere_pos = [np.random.uniform(-0.05, 0.05), np.random.uniform(0.15, 0.2), np.random.uniform(0.25, 0.3)]
        sphere_scaling = 0.8
        
        self.sphere = p.loadURDF(self.urdf_sphere, basePosition=sphere_pos, globalScaling=sphere_scaling, useFixedBase=False)
        
        # Create a mock object for Tacto that has the expected attributes
        sphere_object_for_tacto = SimpleNamespace(
            id=self.sphere,
            urdf_path=self.urdf_sphere,
            global_scaling=sphere_scaling
        )
        self.sensor.add_body(sphere_object_for_tacto)
        
        for _ in range(10): p.stepSimulation()

    def reset_(self, dones):
        if self.hand is not None: p.removeBody(self.hand)
        if self.sphere is not None: p.removeBody(self.sphere)
        self._setup_simulation()
        for idx in self.joint_indices: p.resetJointState(self.hand, idx, 0.0, 0.0)
        self.step_counts[dones] = 0
        self.hold_counters[dones] = 0
        self.episode_rewards[dones] = 0
        self.episode_lengths[dones] = 0
        return self._get_observation()

    def _get_observation(self):
        colors, _ = self.sensor.render()
        grayscale_images = [cv2.cvtColor(colors[i], cv2.COLOR_RGB2GRAY) for i in self.cam_order if i < len(colors)]
        if not grayscale_images: return np.zeros((1, *self.observation_space.shape), dtype=np.uint8)
        obs = np.stack(grayscale_images, axis=0)
        
        # Optionally render sensor views
        if self.render_sensor and colors:
            # Create a grid view of all sensor images
            sensor_grid = np.hstack([colors[i] for i in self.cam_order if i < len(colors)])
            cv2.imshow("Tactile Sensors", sensor_grid)
            cv2.waitKey(1)
        
        return np.expand_dims(obs, axis=0)

    def step_wait(self):
        self.step_counts += 1
        self.episode_lengths += 1
        actions = self.actions[0]
        
        p.resetBaseVelocity(self.hand, linearVelocity=actions[:3] * 0.5, angularVelocity=actions[3:6] * 1.0)
        p.setJointMotorControlArray(
            bodyUniqueId=self.hand, jointIndices=self.joint_indices,
            controlMode=p.VELOCITY_CONTROL, targetVelocities=actions[6:] * 2.0
        )
        p.stepSimulation()
        if self.vis: time.sleep(1.0 / self.sim_freq)
        
        hand_pos, _ = p.getBasePositionAndOrientation(self.hand)
        sphere_pos, _ = p.getBasePositionAndOrientation(self.sphere)
        distance = np.linalg.norm(np.array(hand_pos) - np.array(sphere_pos))
        
        _, depths = self.sensor.render()
        contact_masks = [(d - d.min()) / (d.max() - d.min() + 1e-6) < 0.15 for d in [depths[i] for i in self.cam_order] if d.size > 0]
        num_contacts = sum(np.mean(m) for m in contact_masks)
        
        is_grasping = num_contacts >= 3
        if is_grasping: self.hold_counters[0] += 1
        else: self.hold_counters[0] = 0
        
        reward = np.exp(-5.0 * distance) + (num_contacts / 4.0) * 2.0 + (self.hold_counters[0] * 0.1)
        self.episode_rewards[0] += reward
        
        dones = np.array([self.step_counts[0] >= self.max_steps or self.hold_counters[0] >= 50 or distance > 1.5])
        
        infos = [{
            "success": self.hold_counters[0] >= 50,
            "distance": distance,
            "num_contacts": num_contacts,
            "is_grasping": is_grasping,
            "hold_counter": self.hold_counters[0]
        }]
        
        # Add episode info when done
        if dones[0]:
            infos[0]["episode"] = {
                "r": self.episode_rewards[0],
                "l": self.episode_lengths[0],
                "t": time.time()
            }
        
        obs = self.reset_(dones) if dones[0] else self._get_observation()
        return obs, np.array([reward]), dones, infos

    def reset(self): 
        return self.reset_(np.ones(self.num_envs, dtype=bool))
    
    def step_async(self, actions): 
        self.actions = actions
    
    def close(self): 
        p.disconnect()
        cv2.destroyAllWindows()
    
    def seed(self, seed=None): 
        np.random.seed(seed)
    
    def get_attr(self, attr_name, indices=None): 
        return [getattr(self, attr_name, None)]
    
    def set_attr(self, attr_name, value, indices=None): 
        pass
    
    def env_method(self, method_name, *method_args, indices=None, **method_kwargs): 
        pass
    
    def env_is_wrapped(self, wrapper_class, indices=None): 
        return [False] * self.num_envs


def find_latest_model(base_path="ppo_allegro_hand*.zip"):
    """Find the most recent model file."""
    model_files = glob.glob(base_path)
    if not model_files:
        return None
    return max(model_files, key=os.path.getmtime)


def run_trained_model(model_path, num_episodes=10, render_sensor=False, deterministic=True):
    """Run the trained model for evaluation."""
    
    # Create environment with visualization
    env = AllegroHandTactileEnv(vis=True, render_sensor=render_sensor)
    
    # Load the trained model
    print(f"Loading model from: {model_path}")
    model = PPO.load(model_path)
    
    # Run evaluation episodes
    episode_rewards = []
    episode_successes = []
    
    for episode in range(num_episodes):
        obs = env.reset()
        done = False
        episode_reward = 0
        episode_info = {}
        
        print(f"\n--- Episode {episode + 1}/{num_episodes} ---")
        
        while not done:
            # Get action from the model
            action, _states = model.predict(obs, deterministic=deterministic)
            
            # Step the environment
            env.step_async(action)
            obs, reward, done, info = env.step_wait()
            
            episode_reward += reward[0]
            episode_info = info[0]
            done = done[0]  # Extract boolean from array
            
            # Print current status
            if episode_info.get('is_grasping', False):
                print(f"\rGrasping! Hold counter: {episode_info.get('hold_counter', 0)}/50", end='')
            else:
                print(f"\rDistance: {episode_info.get('distance', 0):.3f}, Contacts: {episode_info.get('num_contacts', 0):.1f}", end='')
        
        # Episode finished
        success = episode_info.get('success', False)
        episode_rewards.append(episode_reward)
        episode_successes.append(success)
        
        print(f"\nEpisode {episode + 1} finished:")
        print(f"  Total Reward: {episode_reward:.2f}")
        print(f"  Success: {'Yes' if success else 'No'}")
        print(f"  Episode Length: {episode_info.get('episode', {}).get('l', 0)}")
    
    # Print summary statistics
    print(f"\n--- Evaluation Summary ---")
    print(f"Average Reward: {np.mean(episode_rewards):.2f} ± {np.std(episode_rewards):.2f}")
    print(f"Success Rate: {np.mean(episode_successes) * 100:.1f}%")
    print(f"Best Episode Reward: {np.max(episode_rewards):.2f}")
    print(f"Worst Episode Reward: {np.min(episode_rewards):.2f}")
    
    env.close()
    return episode_rewards, episode_successes


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run trained Allegro Hand model")
    parser.add_argument("--model", type=str, default=None, 
                        help="Path to the trained model. If not specified, uses the latest model.")
    parser.add_argument("--episodes", type=int, default=10, 
                        help="Number of episodes to run")
    parser.add_argument("--render-sensor", action="store_true", 
                        help="Show tactile sensor views during execution")
    parser.add_argument("--stochastic", action="store_true", 
                        help="Use stochastic actions instead of deterministic")
    
    args = parser.parse_args()
    
    # Find model path
    if args.model is None:
        model_path = find_latest_model()
        if model_path is None:
            print("No trained model found! Please specify a model path or train a model first.")
            exit(1)
    else:
        model_path = args.model
        if not os.path.exists(model_path):
            print(f"Model file not found: {model_path}")
            exit(1)
    
    # Run the evaluation
    run_trained_model(
        model_path=model_path,
        num_episodes=args.episodes,
        render_sensor=args.render_sensor,
        deterministic=not args.stochastic
    )