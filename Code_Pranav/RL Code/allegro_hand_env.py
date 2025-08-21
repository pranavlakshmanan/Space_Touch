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
from types import SimpleNamespace # Import SimpleNamespace to create a mock object


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
                 sensor_res=(60, 80)):
        
        self.vis = vis
        self.max_steps = max_steps
        self.sim_freq = 240
        self.urdf_hand = urdf_hand
        self.urdf_sphere = urdf_sphere
        self.sensor_res = sensor_res
        
        self.tip_labels = {
            "joint_15.0_tip": "little", "joint_11.0_tip": "index", 
            "joint_7.0_tip": "middle", "joint_3.0_tip": "ring",
        }
        
        # Connect to PyBullet directly
        if self.vis: p.connect(p.GUI)
        else: p.connect(p.DIRECT)
        
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.setGravity(0, 0, 0)
        p.setRealTimeSimulation(0)
        p.loadURDF("plane.urdf", [0, 0, 0], useFixedBase=True)
        if self.vis: p.resetDebugVisualizerCamera(0.8, 60, -30, [0, 0, 0.25])
        
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
        
        # FIX: Create a mock object for Tacto that has the expected attributes
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
        return self._get_observation()

    def _get_observation(self):
        colors, _ = self.sensor.render()
        grayscale_images = [cv2.cvtColor(colors[i], cv2.COLOR_RGB2GRAY) for i in self.cam_order if i < len(colors)]
        if not grayscale_images: return np.zeros((1, *self.observation_space.shape), dtype=np.uint8)
        obs = np.stack(grayscale_images, axis=0)
        return np.expand_dims(obs, axis=0)

    def step_wait(self):
        self.step_counts += 1
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
        
        dones = np.array([self.step_counts[0] >= self.max_steps or self.hold_counters[0] >= 50 or distance > 1.5])
        infos = [{"success": self.hold_counters[0] >= 50, "distance": distance}]
        
        obs = self.reset_(dones) if dones[0] else self._get_observation()
        return obs, np.array([reward]), dones, infos

    def reset(self): return self.reset_(np.ones(self.num_envs, dtype=bool))
    def step_async(self, actions): self.actions = actions
    def close(self): p.disconnect()
    def seed(self, seed=None): np.random.seed(seed)
    def get_attr(self, attr_name, indices=None): return [getattr(self, attr_name, None)]
    def set_attr(self, attr_name, value, indices=None): pass
    def env_method(self, method_name, *method_args, indices=None, **method_kwargs): pass
    def env_is_wrapped(self, wrapper_class, indices=None): return [False] * self.num_envs

# --- Main Training Block ---
if __name__ == "__main__":
    env = AllegroHandTactileEnv(vis=True)

    log_dir = "./allegro_tensorboard_log/"
    model = PPO("CnnPolicy", env, verbose=1, tensorboard_log=log_dir, n_steps=1024)

    print("\n--- Starting Training ---")
    model.learn(total_timesteps=25000)
    
    model.save("ppo_allegro_hand")
    print("\n--- Training Finished. Model saved to ppo_allegro_hand.zip ---")

    env.close()