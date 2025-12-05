#!/usr/bin/env python3
"""
Simplified V5 Training Script
Focused on stable training without visualization complications
"""

import os
import sys
import numpy as np
from datetime import datetime
import wandb

# Add project path
sys.path.append('/home/pralak/Space_Touch')

# Set matplotlib to non-interactive backend
import matplotlib
matplotlib.use('Agg')

# Import training components
import pybullet as p
import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import VecEnv
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.logger import configure
import torch

# Import reward function
from reward_functions.convex_hull_overlap_reward import ConvexHullOverlapReward

print("🚀 V5 SIMPLE CONVEX HULL TRAINING")
print("=" * 50)

# Training configuration
TOTAL_TIMESTEPS = int(os.environ.get('TRAINING_STEPS', 5000))  # Short test by default
timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
LOG_DIR = f"SC1_Training_Runs/Run_{timestamp}_V5_Simple"
MODEL_NAME = f"v5_simple_model_{timestamp}"

print(f"📁 Log directory: {LOG_DIR}")
print(f"🎯 Total timesteps: {TOTAL_TIMESTEPS:,}")

# Create directories
os.makedirs(LOG_DIR, exist_ok=True)

# Simple WandB callback
class SimpleWandBCallback(BaseCallback):
    def __init__(self, log_freq=100, verbose=1):
        super().__init__(verbose)
        self.log_freq = log_freq
        self.last_logged_step = 0

    def _on_step(self) -> bool:
        if self.num_timesteps - self.last_logged_step >= self.log_freq:
            try:
                log_dict = {'timestep': self.num_timesteps}
                wandb.log(log_dict)
                self.last_logged_step = self.num_timesteps
            except Exception as e:
                if self.verbose > 0:
                    print(f"WandB logging error: {e}")
        return True

# Simple Environment class
class SimpleConvexHullEnv(gym.Env):
    def __init__(self):
        # Define spaces (28D to accommodate 4 tactile values)
        self.observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(28,), dtype=np.float32)
        self.action_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(10,), dtype=np.float32)

        # Initialize reward function (no visualization)
        reward_config = {
            'object_radius': 0.05,
            'safety_margin': 0.025,
            'overlap_scale': 1000.0,  # Lower scale for testing
            'contact_penalty': -1.0,   # Smaller penalty for testing
            'generate_vis': False,     # No visualization
        }
        self.reward_calculator = ConvexHullOverlapReward(config=reward_config)

        self.step_count = 0
        self.max_steps = 1000

        print("✅ Simple environment initialized")

    def reset(self, seed=None, options=None):
        self.step_count = 0
        # Return simple observation (28D)
        obs = np.zeros(28, dtype=np.float32)
        return obs, {}

    def step(self, action):
        self.step_count += 1

        # Generate simple observation with proper finger and contact data (28D)
        obs = np.zeros(28, dtype=np.float32)

        # Set reasonable values for reward calculation
        obs[:3] = [0.0, 0.0, 0.3]     # Hand position
        obs[3:6] = [0.25, 0.15, 0.35] # Target position

        # Finger positions (4 fingers × 3D = 12D)
        finger_data = np.random.randn(12) * 0.05 + np.tile([0.2, 0.15, 0.35], 4)
        obs[12:24] = finger_data

        # Binary contact (4D) - now we have space for all 4
        obs[24:28] = np.zeros(4)  # All 4 tactile values

        # Calculate reward
        try:
            finger_positions = obs[12:24].reshape(4, 3)
            palm_position = obs[:3]
            object_pos = obs[3:6]
            binary_contact = obs[24:28]

            reward_obs = {
                'finger_positions': finger_positions,
                'palm_position': palm_position,
                'object_pos': object_pos,
                'binary_contact': binary_contact,
                'episode_step': self.step_count,
            }

            reward, reward_info = self.reward_calculator.calculate_reward(reward_obs)
        except Exception as e:
            print(f"Reward calculation error: {e}")
            reward = 0.0

        # Check termination
        done = self.step_count >= self.max_steps
        truncated = False
        info = {}

        return obs, reward, done, truncated, info

def main():
    try:
        # Initialize WandB
        wandb.init(
            project="space-touch-v5-simple",
            name=f"V5_Simple_{timestamp}",
            config={
                "algorithm": "PPO",
                "total_timesteps": TOTAL_TIMESTEPS,
                "approach": "simple_convex_hull_test",
            },
            tags=["v5", "simple", "test"]
        )

        print("🔧 Creating simple environment...")
        env = SimpleConvexHullEnv()

        print("🧠 Creating PPO model...")
        model = PPO(
            "MlpPolicy",
            env,
            learning_rate=3e-4,
            n_steps=512,   # Smaller for testing
            batch_size=32, # Smaller for testing
            n_epochs=5,    # Fewer epochs for testing
            verbose=1,
            device="cpu"   # Use CPU to avoid GPU warnings
        )

        # Set up logging
        model.set_logger(configure(LOG_DIR, ["csv"]))

        # Create callback
        wandb_callback = SimpleWandBCallback(log_freq=100)

        print(f"🏋️  Starting training with {model.device} device...")
        print(f"⏱️  Training for {TOTAL_TIMESTEPS:,} steps...")

        # Training
        model.learn(
            total_timesteps=TOTAL_TIMESTEPS,
            callback=wandb_callback,
        )

        # Save final model
        final_model_path = os.path.join(LOG_DIR, f"{MODEL_NAME}.zip")
        model.save(final_model_path)
        print(f"🎯 Final model saved: {final_model_path}")

        wandb.log({
            "training_completed": True,
            "final_timesteps": model.num_timesteps,
        })

        print("\n" + "=" * 50)
        print("✅ SIMPLE TRAINING COMPLETED SUCCESSFULLY!")
        print("=" * 50)
        print(f"🎯 Final model: {final_model_path}")

    except Exception as e:
        print(f"❌ Training failed: {e}")
        import traceback
        traceback.print_exc()
    finally:
        wandb.finish()

if __name__ == "__main__":
    main()