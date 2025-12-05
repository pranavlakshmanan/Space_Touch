#!/usr/bin/env python3
"""
V7_SC1.py - Improved Curriculum Space Debris Soft-Capture Training

Key improvements over V6:
- Cleaner 3-phase curriculum with skill isolation
- Phase 1: Pure approach (high proximity, low overlap)
- Phase 2: Pure envelopment (NO proximity, high overlap)
- Phase 3: Precision balance (overlap + clearance, harsh contact penalty)
- Optimized for 200K timesteps on CPU
- Retains all V6 bug fixes (memory, PyBullet reset, step counter)

Usage:
    python v7_sc1.py                    # Train new model
    python v7_sc1.py --resume PATH      # Resume from checkpoint
    python v7_sc1.py --test PATH        # Test trained model
"""

import os
import sys
import time
import gc
import argparse
from datetime import datetime
from pathlib import Path

import numpy as np
import pybullet as p
import pybullet_data
import gymnasium as gym
from gymnasium import spaces

import torch
from scipy import signal

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import VecEnv
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback
from stable_baselines3.common.logger import configure

import wandb

# Import V7 reward calculator
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from reward_functions.v7_reward import V7RewardCalculator

# Fix collections compatibility
import collections.abc
import collections
collections.Mapping = collections.abc.Mapping
collections.MutableMapping = collections.abc.MutableMapping
collections.Sequence = collections.abc.Sequence
collections.MutableSequence = collections.abc.MutableSequence


# =============================================================================
# TACTO Import (Optional)
# =============================================================================
try:
    import tacto
    TACTO_AVAILABLE = True
except ImportError:
    TACTO_AVAILABLE = False


# =============================================================================
# Utility Classes
# =============================================================================

class LowPassFilter:
    """Butterworth low-pass filter for control smoothing"""
    
    def __init__(self, cutoff_freq=8.0, sampling_freq=240.0, order=2):
        nyquist = sampling_freq / 2
        normalized_cutoff = cutoff_freq / nyquist
        self.b, self.a = signal.butter(order, normalized_cutoff, btype='low')
        self.zi = None
    
    def filter(self, data):
        if self.zi is None:
            self.zi = signal.lfilter_zi(self.b, self.a) * data
        filtered, self.zi = signal.lfilter(self.b, self.a, [data], zi=self.zi)
        return filtered[0]
    
    def reset(self):
        self.zi = None


class TendonController:
    """Tendon-based finger control for biomimetic actuation"""
    
    FINGER_CHAINS = {
        "index": [8, 9, 10, 11],
        "middle": [4, 5, 6, 7],
        "ring": [0, 1, 2, 3],
        "thumb": [12, 13, 14, 15]
    }
    
    TENDON_FORCE_GAIN = 3.0
    TENDON_DAMPING = 2.5
    MAX_TENDON_FORCE = 15.0
    
    def apply_tendon_forces(self, hand_id, tendon_commands):
        """Apply tendon forces to finger joints"""
        finger_names = ["ring", "middle", "index", "thumb"]
        
        for finger_idx, (name, cmd) in enumerate(zip(finger_names, tendon_commands)):
            if name in self.FINGER_CHAINS:
                force = np.clip(cmd, 0.0, 1.0) * self.MAX_TENDON_FORCE
                
                for i, joint_id in enumerate(self.FINGER_CHAINS[name]):
                    multiplier = 1.0 - (i * 0.1)
                    joint_force = force * multiplier * self.TENDON_FORCE_GAIN
                    
                    p.setJointMotorControl2(
                        hand_id, joint_id,
                        controlMode=p.TORQUE_CONTROL,
                        force=joint_force
                    )
                    p.changeDynamics(hand_id, joint_id, jointDamping=self.TENDON_DAMPING)


# =============================================================================
# V6 Environment
# =============================================================================

class V7Environment(VecEnv):
    """
    V7 Soft-Capture Environment with Improved Curriculum

    Based on V6 with all bug fixes:
    - Direct position control for hand-finger synchronization
    - Memory optimization (hull caching, PyBullet reset every 10 episodes)
    - Uses V7 reward calculator with cleaner phase progression
    """
    
    # Link IDs for Allegro hand
    FINGERTIP_LINKS = [11, 7, 3, 15]   # index, middle, ring, thumb fingertips
    FINGERBASE_LINKS = [8, 4, 0, 12]   # index, middle, ring, thumb bases
    
    def __init__(self, 
                 vis=False, 
                 max_steps=500,
                 urdf_path="/home/pralak/Space_Touch/examples/allegro_hand_description/allegro_hand_description_left_digit_fixed.urdf"):
        
        self.vis = vis
        self.max_steps = max_steps
        self.urdf_path = urdf_path
        self.sim_freq = 240.0
        
        # Action: 6 DOF base movement + 4 tendon commands = 10D
        action_space = spaces.Box(low=-1.0, high=1.0, shape=(10,), dtype=np.float32)
        
        # Observation: 28D total
        observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(28,), dtype=np.float32)
        
        super().__init__(1, observation_space, action_space)
        
        # Initialize V7 reward calculator with improved curriculum
        self.reward_calc = V7RewardCalculator({
            'object_radius': 0.05,
            'safety_margin': 0.025,
            'object_hull_points': 32,
            'hull_compute_freq': 24,  # 10Hz at 240Hz sim (good balance)
        })
        
        # PyBullet initialization
        self._init_pybullet()
        
        # Robot state
        self.hand_id = None
        self.target_id = None
        self.tendon_controller = TendonController()
        self.control_filters = [LowPassFilter() for _ in range(6)]
        
        # Episode state
        self.step_count = 0
        self.episode_reward = 0.0
        self.latest_info = {}
        
        # Target position
        self.target_pos = np.array([0.25, 0.15, 0.35])
        
        # Workspace bounds for safety
        self.workspace = {
            'x_min': -0.3, 'x_max': 0.6,
            'y_min': -0.3, 'y_max': 0.4,
            'z_min': 0.15, 'z_max': 0.55
        }
        
        # Movement scaling
        self.position_scale = 0.01  # 10mm per step at max action
        self.rotation_scale = 0.05  # radians per unit action
        
        # Tactile sensor
        self.tactile_sensor = None
        self._init_tactile()
        
        # Initial reset
        self.reset()
    
    def _init_pybullet(self):
        """Initialize PyBullet physics engine"""
        if self.vis:
            try:
                self.physics_client = p.connect(p.GUI)
                p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0)
            except:
                self.physics_client = p.connect(p.DIRECT)
        else:
            self.physics_client = p.connect(p.DIRECT)
        
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.setGravity(0, 0, 0)
        p.setRealTimeSimulation(0)
        p.setTimeStep(1.0 / self.sim_freq)
        p.loadURDF("plane.urdf", [0, 0, 0], useFixedBase=True)
    
    def _init_tactile(self):
        """Initialize TACTO tactile sensors if available"""
        if not TACTO_AVAILABLE:
            return
        
        try:
            self.tactile_sensor = tacto.Sensor(
                width=120, height=160,
                config_path=tacto.get_digit_config_path(),
                visualize_gui=False
            )
        except:
            self.tactile_sensor = None
    
    def _spawn_hand(self):
        """Spawn Allegro hand at initial position (phase-dependent)"""
        if self.hand_id is not None:
            p.removeBody(self.hand_id)

        # Phase-dependent starting distance
        current_phase = self.reward_calc.current_phase
        if current_phase == 0:
            # Phase 0: Very close (5cm away)
            offset = np.array([-0.05, 0.0, 0.03])
        elif current_phase == 1:
            # Phase 1: Close (8cm away)
            offset = np.array([-0.08, 0.0, 0.05])
        else:
            # Phase 2+: Standard distance (10cm away)
            offset = np.array([-0.10, 0.0, 0.05])

        init_pos = self.target_pos + offset
        init_orn = p.getQuaternionFromEuler([0, 0, 0])

        self.hand_id = p.loadURDF(
            self.urdf_path,
            init_pos.tolist(),
            init_orn,
            useFixedBase=False,
            flags=p.URDF_USE_SELF_COLLISION
        )
        
        p.changeDynamics(self.hand_id, -1,
                         linearDamping=0.0,
                         angularDamping=0.0,
                         mass=0.5)
        
        num_joints = p.getNumJoints(self.hand_id)
        for j in range(num_joints):
            p.resetJointState(self.hand_id, j, 0.0)
            p.setJointMotorControl2(self.hand_id, j, p.VELOCITY_CONTROL, force=0)
        
        if self.tactile_sensor is not None:
            for link_id in self.FINGERTIP_LINKS:
                try:
                    self.tactile_sensor.add_camera(self.hand_id, link_id)
                except:
                    pass
    
    def _spawn_target(self):
        """Spawn target sphere (static, zero mass)"""
        if self.target_id is not None:
            p.removeBody(self.target_id)
        
        radius = 0.05
        col_id = p.createCollisionShape(p.GEOM_SPHERE, radius=radius)
        vis_id = p.createVisualShape(p.GEOM_SPHERE, radius=radius, rgbaColor=[1, 0, 0, 0.8])
        
        self.target_id = p.createMultiBody(
            baseMass=0.0,
            baseCollisionShapeIndex=col_id,
            baseVisualShapeIndex=vis_id,
            basePosition=self.target_pos.tolist()
        )
        
        p.changeDynamics(self.target_id, -1,
                         linearDamping=0, angularDamping=0,
                         lateralFriction=0, spinningFriction=0, rollingFriction=0)
    
    def _get_finger_positions(self) -> np.ndarray:
        """Get fingertip positions with forced forward kinematics"""
        positions = []
        p.performCollisionDetection()
        
        for link_id in self.FINGERTIP_LINKS:
            state = p.getLinkState(self.hand_id, link_id, computeForwardKinematics=1)
            pos = np.array(state[0])
            positions.append(pos)
        
        return np.array(positions)
    
    def _get_finger_base_positions(self) -> np.ndarray:
        """Get finger base positions for 9-point hull"""
        positions = []
        
        for link_id in self.FINGERBASE_LINKS:
            state = p.getLinkState(self.hand_id, link_id, computeForwardKinematics=1)
            pos = np.array(state[0])
            positions.append(pos)
        
        return np.array(positions)
    
    def _get_palm_position(self) -> np.ndarray:
        """Calculate palm center from finger bases"""
        bases = self._get_finger_base_positions()
        return np.mean(bases, axis=0)
    
    def _get_tactile_contacts(self) -> np.ndarray:
        """Get binary contact flags for each finger"""
        contacts = np.zeros(4)
        
        if self.target_id is None:
            return contacts
        
        contact_points = p.getContactPoints(bodyA=self.hand_id, bodyB=self.target_id)
        
        for contact in contact_points:
            link_id = contact[3]
            if link_id in self.FINGERTIP_LINKS:
                finger_idx = self.FINGERTIP_LINKS.index(link_id)
                contacts[finger_idx] = 1.0
        
        if self.tactile_sensor is not None:
            try:
                _, depths = self.tactile_sensor.render()
                for i, depth in enumerate(depths[:4]):
                    if depth is not None and np.var(depth) > 0.001:
                        contacts[i] = 1.0
            except:
                pass
        
        return contacts
    
    def _get_observation(self) -> np.ndarray:
        """Build 28D observation vector"""
        obs = np.zeros(28, dtype=np.float32)
        
        if self.hand_id is None:
            return obs
        
        base_pos, base_orn = p.getBasePositionAndOrientation(self.hand_id)
        base_vel, base_ang_vel = p.getBaseVelocity(self.hand_id)
        
        base_pos = np.array(base_pos)
        base_vel = np.array(base_vel)
        base_ang_vel = np.array(base_ang_vel)
        
        target_pos = self.target_pos
        finger_pos = self._get_finger_positions().flatten()
        tactile = self._get_tactile_contacts()
        
        obs[0:3] = base_pos
        obs[3:6] = target_pos
        obs[6:9] = base_vel
        obs[9:12] = base_ang_vel
        obs[12:24] = finger_pos
        obs[24:28] = tactile
        
        return obs
    
    def _apply_action(self, action: np.ndarray):
        """Apply action using DIRECT POSITION CONTROL"""
        if self.hand_id is None:
            return
        
        current_pos, current_orn = p.getBasePositionAndOrientation(self.hand_id)
        current_pos = np.array(current_pos)
        current_euler = np.array(p.getEulerFromQuaternion(current_orn))
        
        linear_action = action[:3]
        angular_action = action[3:6]
        tendon_action = action[6:10]
        
        filtered_linear = np.array([
            self.control_filters[i].filter(linear_action[i]) 
            for i in range(3)
        ])
        filtered_angular = np.array([
            self.control_filters[i+3].filter(angular_action[i]) 
            for i in range(3)
        ])
        
        new_pos = current_pos + filtered_linear * self.position_scale
        new_euler = current_euler + filtered_angular * self.rotation_scale
        
        new_pos[0] = np.clip(new_pos[0], self.workspace['x_min'], self.workspace['x_max'])
        new_pos[1] = np.clip(new_pos[1], self.workspace['y_min'], self.workspace['y_max'])
        new_pos[2] = np.clip(new_pos[2], self.workspace['z_min'], self.workspace['z_max'])
        
        new_orn = p.getQuaternionFromEuler(new_euler.tolist())
        
        # DIRECT POSITION CONTROL - Key fix!
        p.resetBasePositionAndOrientation(self.hand_id, new_pos.tolist(), new_orn)
        p.resetBaseVelocity(self.hand_id, [0, 0, 0], [0, 0, 0])
        
        tendon_cmd = np.clip(tendon_action, -0.5, 0.5)
        self.tendon_controller.apply_tendon_forces(self.hand_id, tendon_cmd)
    
    def _calculate_reward(self, obs: np.ndarray) -> tuple:
        """Calculate reward using V6 reward calculator"""
        finger_positions = self._get_finger_positions()
        finger_bases = self._get_finger_base_positions()
        palm_position = self._get_palm_position()
        tactile = self._get_tactile_contacts()
        
        reward_obs = {
            'finger_positions': finger_positions,
            'finger_bases': finger_bases,
            'palm_position': palm_position,
            'object_pos': self.target_pos,
            'binary_contact': tactile,
        }
        
        reward, info = self.reward_calc.calculate_reward(reward_obs)
        self.latest_info = info
        
        return reward, info
    
    def _check_termination(self, info: dict) -> tuple:
        """Check if episode should terminate"""
        done = False

        if self.step_count >= self.max_steps:
            done = True

        # ADDED: Terminate if hand wanders too far (force learning to stay close)
        distance = info.get('distance_to_target', 0)
        if distance > 0.25:  # 25cm - too far from target
            done = True

        if (self.reward_calc.current_phase == 3 and
            info.get('consecutive_success_steps', 0) >= 50):
            done = True

        return done
    
    # VecEnv Interface
    def step_async(self, actions):
        self._actions = actions
    
    def step_wait(self):
        action = self._actions[0]
        
        self._apply_action(action)
        p.stepSimulation()
        
        obs = self._get_observation()
        reward, info = self._calculate_reward(obs)
        
        self.step_count += 1
        self.episode_reward += reward
        
        done = self._check_termination(info)
        
        step_info = {
            'episode_reward': self.episode_reward,
            'episode_length': self.step_count,
            'reward_info': info,
        }
        
        if done:
            step_info['terminal_observation'] = obs
            obs = self._reset_env()
        
        return np.array([obs]), np.array([reward]), np.array([done]), [step_info]
    
    def reset(self):
        obs = self._reset_env()
        return np.array([obs])
    
    def _reset_env(self) -> np.ndarray:
        """Reset environment for new episode"""
        self.step_count = 0
        self.episode_reward = 0.0

        # Track episodes for periodic full reset
        if not hasattr(self, '_episode_count'):
            self._episode_count = 0
        self._episode_count += 1

        # MEMORY FIX: Full PyBullet reset every 10 episodes to clear internal caches
        if self._episode_count % 10 == 0:
            # Save current phase
            current_phase = self.reward_calc.current_phase

            # Full reset clears PyBullet's collision cache, contact history, etc.
            p.resetSimulation()
            p.setGravity(0, 0, 0)
            p.setRealTimeSimulation(0)
            p.setTimeStep(1.0 / self.sim_freq)
            p.loadURDF("plane.urdf", [0, 0, 0], useFixedBase=True)

            # Force spawn (body IDs are now invalid)
            self.hand_id = None
            self.target_id = None

            # Restore phase
            self.reward_calc.current_phase = current_phase

            gc.collect()

        for f in self.control_filters:
            f.reset()

        self.reward_calc.reset()

        self._spawn_hand()
        self._spawn_target()

        self.target_pos = np.array([0.25, 0.15, 0.35])
        self.target_pos += np.random.uniform(-0.02, 0.02, 3)

        if self.target_id is not None:
            p.resetBasePositionAndOrientation(
                self.target_id,
                self.target_pos.tolist(),
                [0, 0, 0, 1]
            )

        for _ in range(10):
            p.stepSimulation()

        # MEMORY FIX: Aggressive garbage collection every episode
        # Now safe to run every episode since hull computation is reduced to 10Hz
        gc.collect()

        return self._get_observation()
    
    def close(self):
        """Clean up PyBullet resources"""
        try:
            # Remove bodies before disconnecting
            if self.hand_id is not None:
                p.removeBody(self.hand_id)
                self.hand_id = None
            if self.target_id is not None:
                p.removeBody(self.target_id)
                self.target_id = None

            # Disconnect physics
            if hasattr(self, 'physics_client') and self.physics_client >= 0:
                p.disconnect(self.physics_client)
                self.physics_client = -1
        except Exception as e:
            pass  # Ignore cleanup errors
    
    def render(self, mode='human'):
        pass
    
    def env_is_wrapped(self, wrapper_class, indices=None):
        return [False]
    
    def env_method(self, method_name, *args, indices=None, **kwargs):
        if hasattr(self, method_name):
            return [getattr(self, method_name)(*args, **kwargs)]
        return [None]
    
    def get_attr(self, attr_name, indices=None):
        if hasattr(self, attr_name):
            return [getattr(self, attr_name)]
        return [None]
    
    def set_attr(self, attr_name, value, indices=None):
        setattr(self, attr_name, value)
    
    def update_phase(self, phase: int):
        """Update curriculum phase"""
        self.reward_calc.update_phase(phase)


# =============================================================================
# Callbacks
# =============================================================================

class V7CurriculumCallback(BaseCallback):
    """V7 Performance-based curriculum progression for 200K training"""

    def __init__(self, check_freq=10000, verbose=0):
        super().__init__(verbose)
        self.check_freq = check_freq
        self.current_phase = 0  # Start at Phase 0
        self.phase_start_step = 0

        # V7 Phase thresholds optimized for 200K total steps
        # Phase 0: 30K, Phase 1: 60K, Phase 2: 70K, Phase 3: 40K
        self.phase_thresholds = {
            0: {  # Phase 0: Bootstrap (0-30K)
                'min_steps': 20000,          # At least 20K in phase
                'max_steps': 30000,          # Force advance at 30K
                'mean_distance': 0.15,       # Average distance < 15cm
                'mean_overlap': 0.00001,     # Average overlap > 10 cm³
                'window_size': 500,
            },
            1: {  # Phase 1: Approach (30K-90K)
                'min_steps': 50000,          # At least 50K in phase
                'max_steps': 60000,          # Force advance at 60K (total 90K)
                'mean_distance': 0.15,       # Average distance < 15cm
                'mean_overlap': 0.00001,     # Average overlap > 10 cm³
                'window_size': 1000,
            },
            2: {  # Phase 2: Envelopment (90K-160K)
                'min_steps': 60000,          # At least 60K in phase
                'max_steps': 70000,          # Force advance at 70K (total 160K)
                'mean_overlap': 0.0001,      # Average overlap > 100 cm³
                'mean_distance': 0.20,       # Within 20cm
                'window_size': 1000,
            },
            3: {  # Phase 3: Precision (160K-200K)
                'min_steps': float('inf'),
                'max_steps': float('inf'),   # Train to completion
            },
        }

        # FIXED: Limit list size to prevent memory leak
        self.max_history = 1500  # Maximum size for history lists
        self.recent_overlaps = []
        self.recent_distances = []
    
    def _on_step(self) -> bool:
        # CRITICAL MEMORY FIX: Only sample data every 100 steps (reduces from 240Hz to 2.4Hz)
        # This prevents accumulating 240 entries/second which causes laptop freeze
        if self.num_timesteps % 100 == 0:
            if 'reward_info' in self.locals.get('infos', [{}])[0]:
                info = self.locals['infos'][0]['reward_info']
                self.recent_overlaps.append(info.get('overlap_volume', 0))
                self.recent_distances.append(info.get('distance_to_target', 0))

                # FIXED: Trim lists immediately to prevent unbounded growth
                if len(self.recent_overlaps) > self.max_history:
                    self.recent_overlaps = self.recent_overlaps[-self.max_history:]
                if len(self.recent_distances) > self.max_history:
                    self.recent_distances = self.recent_distances[-self.max_history:]

        if self.num_timesteps % self.check_freq == 0:
            self._check_phase_transition()

        return True
    
    def _check_phase_transition(self):
        """Check if performance criteria met to advance phase"""
        if self.current_phase >= 3:
            return

        steps_in_phase = self.num_timesteps - self.phase_start_step
        threshold = self.phase_thresholds[self.current_phase]
        window = threshold.get('window_size', 500)

        # Need minimum time in phase before considering advancement
        if steps_in_phase < threshold['min_steps']:
            return

        # Force advancement if stuck too long
        if steps_in_phase >= threshold['max_steps']:
            print(f"[Step {self.num_timesteps:,}] Phase {self.current_phase} max steps reached - forcing advance")
            self._advance_phase()
            return

        # Check if we have enough data
        if len(self.recent_overlaps) < window or len(self.recent_distances) < window:
            return

        # Calculate performance metrics
        mean_overlap = np.mean(self.recent_overlaps[-window:])
        mean_distance = np.mean(self.recent_distances[-window:])

        # Check performance criteria
        performance_met = True

        if 'mean_overlap' in threshold:
            if mean_overlap < threshold['mean_overlap']:
                performance_met = False

        if 'mean_distance' in threshold:
            if mean_distance > threshold['mean_distance']:
                performance_met = False

        if performance_met:
            print(f"[Step {self.num_timesteps:,}] Phase {self.current_phase} performance goals met:")
            print(f"  Mean distance: {mean_distance:.4f}m (target: <{threshold.get('mean_distance', 'N/A')})")
            print(f"  Mean overlap: {mean_overlap*1e6:.4f} cm³ (target: >{threshold.get('mean_overlap', 0)*1e6:.4f} cm³)")
            self._advance_phase()
    
    def _advance_phase(self):
        old_phase = self.current_phase
        self.current_phase += 1
        self.phase_start_step = self.num_timesteps
        
        self.training_env.env_method('update_phase', self.current_phase)
        
        if wandb.run is not None:
            wandb.log({
                'curriculum/phase': self.current_phase,
                'curriculum/transition_step': self.num_timesteps,
            })
        
        print(f"[Step {self.num_timesteps:,}] Phase {old_phase} → {self.current_phase}")


class V6WandBCallback(BaseCallback):
    """Lightweight WandB logging"""

    def __init__(self, log_freq=480, verbose=0):  # TIMING FIX: 480 is divisible by 24 (hull compute freq)
        super().__init__(verbose)
        self.log_freq = log_freq
        self.recent_overlaps_for_stats = []
        self.max_stats_history = 200  # Only keep 200 for rolling statistics

    def _on_step(self) -> bool:
        # CRITICAL FIX: Only append data at log_freq intervals, not every step
        # This prevents accumulating 240 entries/second at 240Hz simulation
        if self.num_timesteps % self.log_freq == 0:
            infos = self.locals.get('infos', [{}])
            if infos and 'reward_info' in infos[0]:
                info = infos[0]['reward_info']
                self.recent_overlaps_for_stats.append(info.get('overlap_volume', 0))

                # Trim immediately
                if len(self.recent_overlaps_for_stats) > self.max_stats_history:
                    self.recent_overlaps_for_stats = self.recent_overlaps_for_stats[-self.max_stats_history:]

            self._log_metrics()

        return True
    
    def _log_metrics(self):
        if wandb.run is None:
            return

        infos = self.locals.get('infos', [{}])
        if not infos or 'reward_info' not in infos[0]:
            return

        info = infos[0]['reward_info']

        overlap_cm3 = info.get('overlap_volume', 0) * 1e6
        hand_vol_cm3 = info.get('hand_hull_volume', 0) * 1e6
        obj_vol_cm3 = info.get('object_hull_volume', 0) * 1e6

        log_dict = {
            'train/step': self.num_timesteps,
            'reward/overlap': info.get('overlap_reward', 0),
            'reward/proximity': info.get('proximity_reward', 0),
            'reward/contact_penalty': info.get('contact_penalty', 0),
            'reward/clearance': info.get('clearance_reward', 0),
            'reward/quality': info.get('quality_reward', 0),
            'hull/overlap_cm3': overlap_cm3,
            'hull/hand_volume_cm3': hand_vol_cm3,
            'hull/object_volume_cm3': obj_vol_cm3,
            'state/distance': info.get('distance_to_target', 0),
            'state/contacts': info.get('num_contacts', 0),
            'state/consecutive_success': info.get('consecutive_success_steps', 0),
            'state/phase': info.get('current_phase', 1),
        }

        # Use the new variable name
        if self.recent_overlaps_for_stats:
            log_dict['stats/mean_overlap_cm3'] = np.mean(self.recent_overlaps_for_stats) * 1e6
            log_dict['stats/max_overlap_cm3'] = np.max(self.recent_overlaps_for_stats) * 1e6

        wandb.log(log_dict)


# =============================================================================
# Training Functions
# =============================================================================

def create_model(env, device='cuda'):
    """Create PPO model with tuned hyperparameters"""
    return PPO(
        "MlpPolicy",
        env,
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.001,  # REDUCED: 0.01 → 0.001 to reduce random exploration
        vf_coef=0.5,
        max_grad_norm=0.5,
        device=device,  # MEMORY FIX: Default to CPU to avoid GPU memory issues
        verbose=0,
    )


def train(args):
    """Main V7 training function - optimized for 200K steps"""
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    run_name = f"V7_SC1_{timestamp}"
    log_dir = Path(f"SC1_Training_Runs/{run_name}")
    log_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir = log_dir / "checkpoints"
    checkpoint_dir.mkdir(exist_ok=True)

    wandb.init(
        project="sc1-v7-curriculum",
        name=run_name,
        config={
            'total_timesteps': args.timesteps,
            'algorithm': 'PPO',
            'curriculum_phases': 4,  # Phase 0-3
            'observation_dim': 28,
            'action_dim': 10,
            'position_control': True,
            'hull_points_hand': 9,
            'hull_points_object': 32,
            'overlap_method': 'bbox_fast',
            'phase_0_steps': 30000,
            'phase_1_steps': 60000,
            'phase_2_steps': 70000,
            'phase_3_steps': 40000,
        },
        tags=['v7', 'soft-capture', 'improved-curriculum', 'skill-isolation', '200k-optimized'],
    )

    print(f"Starting V7 training: {run_name}")
    print(f"  Total steps: {args.timesteps:,}")
    print(f"  Curriculum: Phase0(30K) → Phase1(60K) → Phase2(70K) → Phase3(40K)")
    print(f"  Log dir: {log_dir}")

    env = V7Environment(vis=args.vis, max_steps=500)

    if args.resume:
        print(f"  Resuming from: {args.resume}")
        model = PPO.load(args.resume, env=env)
    else:
        # CPU MODE: Using CPU for training
        model = create_model(env, device='cpu')

    model.set_logger(configure(str(log_dir), ["csv", "tensorboard"]))

    callbacks = [
        V7CurriculumCallback(check_freq=10000, verbose=0),
        V6WandBCallback(log_freq=480, verbose=0),  # TIMING FIX: 480 is divisible by 24 (hull freq)
        CheckpointCallback(
            save_freq=50000,
            save_path=str(checkpoint_dir),
            name_prefix="v7_sc1",
            save_replay_buffer=False,
            save_vecnormalize=False,
        ),
    ]
    
    try:
        model.learn(
            total_timesteps=args.timesteps,
            callback=callbacks,
            progress_bar=True,
            reset_num_timesteps=not args.resume,
        )
        
        final_path = log_dir / "final_model.zip"
        model.save(str(final_path))
        print(f"Training complete. Model saved to: {final_path}")
        
    except KeyboardInterrupt:
        print("\nTraining interrupted. Saving checkpoint...")
        interrupt_path = log_dir / "interrupted_model.zip"
        model.save(str(interrupt_path))
        print(f"Checkpoint saved to: {interrupt_path}")
    
    finally:
        env.close()
        wandb.finish()


def test(args):
    """
    Comprehensive model testing with multiple scenarios and visualization.
    Tests the model on various target positions and collects detailed metrics.
    """
    import pandas as pd
    import matplotlib.pyplot as plt
    from pathlib import Path

    print(f"="*60)
    print(f"V6 SC-1 COMPREHENSIVE TESTING")
    print(f"="*60)
    print(f"Model: {args.model}")
    print(f"Episodes per scenario: {args.episodes}")
    print(f"Visualization: {args.vis}")
    print()

    # Define test scenarios
    test_scenarios = {
        'close_easy': {
            'target_pos': np.array([0.20, 0.15, 0.30]),
            'description': 'Close target (easy)',
            'expected_success_rate': 0.7,
        },
        'medium_standard': {
            'target_pos': np.array([0.25, 0.15, 0.35]),
            'description': 'Standard distance (medium)',
            'expected_success_rate': 0.5,
        },
        'far_challenging': {
            'target_pos': np.array([0.35, 0.15, 0.40]),
            'description': 'Far target (hard)',
            'expected_success_rate': 0.2,
        },
        'side_reach': {
            'target_pos': np.array([0.25, 0.25, 0.35]),
            'description': 'Side reach (medium-hard)',
            'expected_success_rate': 0.3,
        },
        'precise_grasp': {
            'target_pos': np.array([0.22, 0.12, 0.32]),
            'description': 'Precision grasp (medium)',
            'expected_success_rate': 0.4,
        },
    }

    # Create test results directory
    model_path = Path(args.model)
    test_dir = model_path.parent / f"v6_test_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    test_dir.mkdir(parents=True, exist_ok=True)
    print(f"Test results directory: {test_dir}")
    print()

    # Load model
    env = V7Environment(vis=args.vis, max_steps=1000)
    model = PPO.load(args.model, env=env)

    # Storage for all test data
    all_results = []
    scenario_summaries = []

    # Test each scenario
    for scenario_name, scenario_config in test_scenarios.items():
        print(f"-" * 60)
        print(f"Testing: {scenario_config['description']}")
        print(f"Target: {scenario_config['target_pos']}")
        print(f"-" * 60)

        scenario_results = []

        for episode in range(args.episodes):
            # Override target position for this scenario
            env.target_pos = scenario_config['target_pos'].copy()

            obs = env.reset()
            episode_reward = 0
            episode_data = []

            for step in range(1000):
                action, _ = model.predict(obs, deterministic=True)
                obs, reward, done, info = env.step(action)

                episode_reward += reward[0]

                # Collect step data
                reward_info = info[0].get('reward_info', {})
                step_data = {
                    'scenario': scenario_name,
                    'episode': episode + 1,
                    'step': step + 1,
                    'reward': reward[0],
                    'cumulative_reward': episode_reward,
                    'overlap_volume_cm3': reward_info.get('overlap_volume', 0) * 1e6,
                    'hand_volume_cm3': reward_info.get('hand_hull_volume', 0) * 1e6,
                    'object_volume_cm3': reward_info.get('object_hull_volume', 0) * 1e6,
                    'distance': reward_info.get('distance_to_target', 0),
                    'num_contacts': reward_info.get('num_contacts', 0),
                    'is_success': reward_info.get('is_success', False),
                    'consecutive_success': reward_info.get('consecutive_success_steps', 0),
                    'phase': reward_info.get('current_phase', 1),
                    'overlap_reward': reward_info.get('overlap_reward', 0),
                    'proximity_reward': reward_info.get('proximity_reward', 0),
                    'contact_penalty': reward_info.get('contact_penalty', 0),
                }
                episode_data.append(step_data)

                if args.vis:
                    time.sleep(0.01)

                if done[0]:
                    break

            # Episode summary
            final_distance = episode_data[-1]['distance']
            max_overlap = max(d['overlap_volume_cm3'] for d in episode_data)
            success = episode_data[-1]['is_success']

            scenario_results.append({
                'episode': episode + 1,
                'total_reward': episode_reward,
                'steps': len(episode_data),
                'final_distance': final_distance,
                'max_overlap_cm3': max_overlap,
                'success': success,
            })

            all_results.extend(episode_data)

            print(f"  Episode {episode + 1:2d}: Reward={episode_reward:7.2f}, "
                  f"Steps={len(episode_data):4d}, Dist={final_distance:.4f}m, "
                  f"Overlap={max_overlap:.4f}cm³, Success={'✓' if success else '✗'}")

        # Scenario summary
        success_rate = sum(r['success'] for r in scenario_results) / len(scenario_results)
        avg_reward = np.mean([r['total_reward'] for r in scenario_results])
        avg_distance = np.mean([r['final_distance'] for r in scenario_results])
        avg_overlap = np.mean([r['max_overlap_cm3'] for r in scenario_results])

        scenario_summaries.append({
            'scenario': scenario_name,
            'description': scenario_config['description'],
            'success_rate': success_rate,
            'avg_reward': avg_reward,
            'avg_final_distance': avg_distance,
            'avg_max_overlap_cm3': avg_overlap,
            'expected_success_rate': scenario_config['expected_success_rate'],
        })

        print(f"  Summary: Success={success_rate:.1%}, "
              f"AvgReward={avg_reward:.2f}, "
              f"AvgDist={avg_distance:.4f}m, "
              f"AvgOverlap={avg_overlap:.4f}cm³")
        print()

    env.close()

    # Save detailed results
    results_df = pd.DataFrame(all_results)
    results_csv = test_dir / "test_results_detailed.csv"
    results_df.to_csv(results_csv, index=False)
    print(f"Saved detailed results: {results_csv}")

    # Save scenario summaries
    summary_df = pd.DataFrame(scenario_summaries)
    summary_csv = test_dir / "test_results_summary.csv"
    summary_df.to_csv(summary_csv, index=False)
    print(f"Saved summary: {summary_csv}")

    # Generate plots
    print()
    print("Generating visualization plots...")
    _plot_test_results(results_df, summary_df, test_dir)

    print()
    print(f"="*60)
    print(f"TESTING COMPLETE")
    print(f"="*60)
    print(f"Results saved to: {test_dir}")
    print()


def _plot_test_results(results_df, summary_df, output_dir):
    """Generate comprehensive test result plots"""
    import matplotlib.pyplot as plt

    fig = plt.figure(figsize=(16, 12))

    # Plot 1: Success rate by scenario
    ax1 = plt.subplot(2, 3, 1)
    scenarios = summary_df['scenario'].values
    success_rates = summary_df['success_rate'].values * 100
    expected_rates = summary_df['expected_success_rate'].values * 100

    x = np.arange(len(scenarios))
    width = 0.35
    ax1.bar(x - width/2, success_rates, width, label='Actual', color='blue', alpha=0.7)
    ax1.bar(x + width/2, expected_rates, width, label='Expected', color='green', alpha=0.7)
    ax1.set_xlabel('Scenario')
    ax1.set_ylabel('Success Rate (%)')
    ax1.set_title('Success Rate by Scenario')
    ax1.set_xticks(x)
    ax1.set_xticklabels([s.replace('_', '\n') for s in scenarios], fontsize=8)
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Plot 2: Average overlap volume by scenario
    ax2 = plt.subplot(2, 3, 2)
    avg_overlaps = summary_df['avg_max_overlap_cm3'].values
    ax2.bar(scenarios, avg_overlaps, color='orange', alpha=0.7)
    ax2.set_xlabel('Scenario')
    ax2.set_ylabel('Avg Max Overlap (cm³)')
    ax2.set_title('Average Maximum Overlap Volume')
    ax2.set_xticklabels([s.replace('_', '\n') for s in scenarios], fontsize=8)
    ax2.grid(True, alpha=0.3)

    # Plot 3: Average final distance by scenario
    ax3 = plt.subplot(2, 3, 3)
    avg_distances = summary_df['avg_final_distance'].values * 100  # Convert to cm
    ax3.bar(scenarios, avg_distances, color='red', alpha=0.7)
    ax3.set_xlabel('Scenario')
    ax3.set_ylabel('Avg Final Distance (cm)')
    ax3.set_title('Average Final Distance to Target')
    ax3.set_xticklabels([s.replace('_', '\n') for s in scenarios], fontsize=8)
    ax3.grid(True, alpha=0.3)

    # Plot 4: Overlap volume over time (first episode per scenario)
    ax4 = plt.subplot(2, 3, 4)
    for scenario in results_df['scenario'].unique():
        scenario_data = results_df[(results_df['scenario'] == scenario) & (results_df['episode'] == 1)]
        ax4.plot(scenario_data['step'], scenario_data['overlap_volume_cm3'],
                label=scenario.replace('_', ' '), alpha=0.7)
    ax4.set_xlabel('Step')
    ax4.set_ylabel('Overlap Volume (cm³)')
    ax4.set_title('Overlap Volume Over Time (Episode 1)')
    ax4.legend(fontsize=8)
    ax4.grid(True, alpha=0.3)

    # Plot 5: Distance over time (first episode per scenario)
    ax5 = plt.subplot(2, 3, 5)
    for scenario in results_df['scenario'].unique():
        scenario_data = results_df[(results_df['scenario'] == scenario) & (results_df['episode'] == 1)]
        ax5.plot(scenario_data['step'], scenario_data['distance'] * 100,
                label=scenario.replace('_', ' '), alpha=0.7)
    ax5.set_xlabel('Step')
    ax5.set_ylabel('Distance to Target (cm)')
    ax5.set_title('Distance to Target Over Time (Episode 1)')
    ax5.legend(fontsize=8)
    ax5.grid(True, alpha=0.3)

    # Plot 6: Reward components (averaged across all scenarios)
    ax6 = plt.subplot(2, 3, 6)
    components = ['overlap_reward', 'proximity_reward', 'contact_penalty']
    avg_components = [results_df[comp].mean() for comp in components]
    colors = ['blue', 'green', 'red']
    ax6.bar(components, avg_components, color=colors, alpha=0.7)
    ax6.set_xlabel('Reward Component')
    ax6.set_ylabel('Average Value')
    ax6.set_title('Average Reward Components')
    ax6.set_xticklabels(['Overlap', 'Proximity', 'Contact\nPenalty'], fontsize=9)
    ax6.grid(True, alpha=0.3)

    plt.tight_layout()
    plot_path = output_dir / "test_results_analysis.png"
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"  Saved plot: {plot_path}")

    # Generate per-scenario detailed plots
    for scenario in results_df['scenario'].unique():
        scenario_data = results_df[results_df['scenario'] == scenario]
        _plot_scenario_details(scenario_data, output_dir, scenario)


def _plot_scenario_details(scenario_data, output_dir, scenario_name):
    """Generate detailed plots for a specific scenario"""
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle(f"Scenario: {scenario_name.replace('_', ' ').title()}", fontsize=14, fontweight='bold')

    # Plot overlap volume for all episodes
    ax1 = axes[0, 0]
    for episode in scenario_data['episode'].unique():
        ep_data = scenario_data[scenario_data['episode'] == episode]
        ax1.plot(ep_data['step'], ep_data['overlap_volume_cm3'], alpha=0.6, linewidth=1)
    ax1.set_xlabel('Step')
    ax1.set_ylabel('Overlap Volume (cm³)')
    ax1.set_title('Overlap Volume Across All Episodes')
    ax1.grid(True, alpha=0.3)

    # Plot distance for all episodes
    ax2 = axes[0, 1]
    for episode in scenario_data['episode'].unique():
        ep_data = scenario_data[scenario_data['episode'] == episode]
        ax2.plot(ep_data['step'], ep_data['distance'] * 100, alpha=0.6, linewidth=1)
    ax2.set_xlabel('Step')
    ax2.set_ylabel('Distance (cm)')
    ax2.set_title('Distance to Target Across All Episodes')
    ax2.grid(True, alpha=0.3)

    # Plot cumulative reward
    ax3 = axes[1, 0]
    for episode in scenario_data['episode'].unique():
        ep_data = scenario_data[scenario_data['episode'] == episode]
        ax3.plot(ep_data['step'], ep_data['cumulative_reward'], alpha=0.6, linewidth=1)
    ax3.set_xlabel('Step')
    ax3.set_ylabel('Cumulative Reward')
    ax3.set_title('Cumulative Reward Across All Episodes')
    ax3.grid(True, alpha=0.3)

    # Plot contact statistics
    ax4 = axes[1, 1]
    episode_contacts = []
    for episode in scenario_data['episode'].unique():
        ep_data = scenario_data[scenario_data['episode'] == episode]
        total_contacts = ep_data['num_contacts'].sum()
        episode_contacts.append(total_contacts)
    ax4.bar(range(1, len(episode_contacts) + 1), episode_contacts, alpha=0.7, color='red')
    ax4.set_xlabel('Episode')
    ax4.set_ylabel('Total Contact Steps')
    ax4.set_title('Total Contact Steps Per Episode')
    ax4.grid(True, alpha=0.3)

    plt.tight_layout()
    plot_path = output_dir / f"scenario_{scenario_name}_detailed.png"
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"  Saved scenario plot: {plot_path}")


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description='V7 SC-1 Improved Curriculum Training')
    subparsers = parser.add_subparsers(dest='command', help='Command to run')

    train_parser = subparsers.add_parser('train', help='Train a new model')
    train_parser.add_argument('--timesteps', type=int, default=200000, help='Total training steps (default: 200K optimized for CPU)')
    train_parser.add_argument('--resume', type=str, default=None, help='Resume from checkpoint')
    train_parser.add_argument('--vis', action='store_true', help='Enable visualization')
    
    test_parser = subparsers.add_parser('test', help='Test a trained model')
    test_parser.add_argument('model', type=str, help='Path to model file')
    test_parser.add_argument('--episodes', type=int, default=5, help='Number of test episodes')
    
    args = parser.parse_args()
    
    if args.command == 'train':
        train(args)
    elif args.command == 'test':
        test(args)
    else:
        args.timesteps = 200000
        args.resume = None
        args.vis = False
        train(args)


if __name__ == "__main__":
    main()
