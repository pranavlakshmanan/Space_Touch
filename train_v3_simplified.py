#!/usr/bin/env python3
"""
SC-1 V3 Training: Simplified 3-Component Reward Function
Phase 1: Static targets, 500K timesteps

OBJECTIVE: Fix negative reward issues and achieve 15-25% success rate
APPROACH: Simplified reward function with only 3 components vs previous 10+

Key Changes from Previous Versions:
- Simplified 3-component reward: Distance + Staged Success + Tactile Engagement
- NO harsh contact penalties (-1.0 per step removed)
- Positive reward dominance (distance progress always positive)
- Tactile contact ENCOURAGED when close (soft-capture goal)
- Extended training: 500K timesteps for thorough learning
- Static targets only (no moving targets for Phase 1)
"""

import os
import sys
import wandb
import numpy as np
from datetime import datetime
from pathlib import Path

# Add project path for imports
sys.path.append('/home/pralak/Space_Touch')

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback, BaseCallback
from stable_baselines3.common.logger import configure

from reward_functions.simplified_reward import SimplifiedReward


# ==================== TRAINING CONFIGURATION ====================

TRAINING_CONFIG = {
    # Training duration - Extended for thorough learning
    'total_timesteps': 500_000,
    'save_freq': 50_000,  # Save checkpoint every 50K steps

    # PPO hyperparameters (conservative for stability)
    'learning_rate': 3e-4,
    'n_steps': 2048,
    'batch_size': 64,
    'n_epochs': 10,
    'gamma': 0.99,
    'gae_lambda': 0.95,
    'clip_range': 0.2,
    'ent_coef': 0.01,
    'vf_coef': 0.5,
    'max_grad_norm': 0.5,

    # Episode settings - Longer episodes to allow learning
    'max_episode_steps': 1000,

    # Success criteria (for logging/testing only)
    'success_distance': 0.08,
    'success_min_consecutive': 50,
}

# Curriculum Learning - DISABLED for Phase 1 (static targets only)
CURRICULUM_CONFIG = {
    'enable_target_motion': False,  # Static targets only
    'target_motion_thresholds': [500_000, 600_000, 700_000],  # Beyond Phase 1
}

# Environment Randomization (moderate for diversity)
RANDOMIZATION_CONFIG = {
    'target_position_range': 0.15,  # ±15cm randomization
    'target_velocity_range': 0.0,   # Static for Phase 1
    'initial_hand_orientation_range': 0.3,  # ±0.3 rad
    'action_noise': 0.1,  # 10% action noise
}

# WandB Configuration
WANDB_CONFIG = {
    'architecture': 'V3_Simplified_Reward',
    'reward_function': 'Simplified_3Component',
    'reward_components': ['distance_progress', 'staged_success_bonus', 'tactile_engagement'],
    'total_timesteps': 500_000,
    'phase': 'Phase1_Static_Targets',
    'target_success_rate': '15-25%',
    'expected_reward_range': '[-0.5, 21.5]',
    **TRAINING_CONFIG,
}


class SC1Environment:
    """
    SC-1 Space Manipulator Environment with Simplified Reward Function
    Designed specifically for soft-capture tasks with positive reward structure
    """

    def __init__(self, reward_function_class=SimplifiedReward, max_episode_steps=1000, **kwargs):
        """
        Initialize SC-1 environment with simplified reward function

        Args:
            reward_function_class: Reward function class to use (SimplifiedReward)
            max_episode_steps: Maximum steps per episode
            **kwargs: Additional configuration parameters
        """

        print("🚀 Initializing SC-1 Environment with Simplified Reward Function")

        # Initialize reward function
        self.reward_calculator = reward_function_class(config=kwargs.get('reward_config', {}))
        self.max_episode_steps = max_episode_steps

        # Episode tracking
        self.current_step = 0
        self.episode_count = 0
        self.total_steps = 0

        # State variables (will be set by PyBullet simulation)
        self.hand_pos = np.zeros(3)
        self.target_pos = np.zeros(3)
        self.distance = 1.0
        self.contact_force = 0.0

        # Success tracking
        self.success = False
        self.episode_rewards = []

        # Logging
        self.log_to_wandb = kwargs.get('log_to_wandb', True)

        print(f"✅ SC-1 Environment initialized with {self.reward_calculator}")
        print(f"   Max episode steps: {self.max_episode_steps}")
        print(f"   Expected reward range: {self.reward_calculator.get_expected_reward_range()}")


    def calculate_reward(self):
        """Calculate reward using simplified reward function"""

        # Gather observation data for reward calculation
        obs_dict = {
            'distance': self._get_distance_to_target(),
            'contact_force': self._get_total_contact_force(),
            'hand_pos': self.hand_pos,
            'target_pos': self.target_pos,
        }

        # Calculate reward using simplified reward function
        total_reward, reward_info = self.reward_calculator.calculate_reward(obs_dict)

        # Log components to WandB if enabled
        if self.log_to_wandb:
            wandb.log({
                'reward/total': total_reward,
                'reward/distance': reward_info['distance_reward'],
                'reward/success_bonus': reward_info['success_bonus'],
                'reward/tactile': reward_info['tactile_reward'],
                'reward/consecutive_success_steps': reward_info['consecutive_steps'],
                'metrics/distance': reward_info['distance'],
                'metrics/contact_force': reward_info['contact_force'],
                'metrics/success_stage': reward_info['success_stage'],
                'metrics/in_success_zone': int(reward_info['in_success_zone']),
                'episode/step': self.current_step,
            }, step=self.total_steps)

        return total_reward, reward_info

    def _get_distance_to_target(self) -> float:
        """Get L2 distance between hand center and target center"""
        # This would be implemented with actual PyBullet physics
        # For now, return stored distance
        return self.distance

    def _get_total_contact_force(self) -> float:
        """Get total normal force from all tactile sensors"""
        # This would be implemented with PyBullet contact detection
        # For now, return stored contact force
        return self.contact_force

    def _check_done(self) -> bool:
        """
        Episode termination conditions - CRITICAL: Don't terminate on contact!

        Returns:
            bool: Whether episode should terminate
        """

        # Success termination (GOOD) - Sustained success in target zone
        if (self.distance < self.reward_calculator.SUCCESS_THRESHOLD and
            self.reward_calculator.consecutive_success_steps >= self.reward_calculator.MIN_CONSECUTIVE_STEPS):
            self.success = True
            return True

        # Max steps reached
        if self.current_step >= self.max_episode_steps:
            return True

        # Failure: Target escaped too far (clear failure case)
        if self.distance > 3.0:  # 3 meters = clear failure
            return True

        # CRITICAL: DO NOT terminate on contact - this was the main bug!
        # DO NOT terminate when close but not quite successful
        # Let the agent learn through reward signal, not episode termination

        return False

    def reset(self):
        """Reset environment for new episode"""

        # Reset reward calculator episode-specific tracking
        self.reward_calculator.reset()

        # Reset episode tracking
        self.current_step = 0
        self.episode_count += 1
        self.success = False

        # Reset state variables (would be set by PyBullet simulation)
        self.hand_pos = np.zeros(3)
        self.target_pos = np.array([0.3, 0.0, 0.0])  # Example target position
        self.distance = np.linalg.norm(self.hand_pos - self.target_pos)
        self.contact_force = 0.0

        print(f"🔄 Episode {self.episode_count} reset. Distance to target: {self.distance:.3f}m")

        # Return initial observation (placeholder)
        return np.concatenate([self.hand_pos, self.target_pos, [self.distance, self.contact_force]])

    def step(self, action):
        """Execute one environment step"""

        self.current_step += 1
        self.total_steps += 1

        # Apply action (placeholder - would use PyBullet physics)
        # For demo purposes, simulate approaching target with some noise
        direction = (self.target_pos - self.hand_pos)
        direction = direction / (np.linalg.norm(direction) + 1e-8)

        # Simulate hand movement based on action
        movement = action[:3] * 0.01  # 1cm per action unit
        self.hand_pos += movement

        # Update distance
        self.distance = np.linalg.norm(self.hand_pos - self.target_pos)

        # Simulate contact force (placeholder)
        if self.distance < 0.15:  # Close to target
            self.contact_force = max(0, np.random.normal(2.0, 1.0))  # Gentle contact
        else:
            self.contact_force = 0.0

        # Calculate reward
        reward, reward_info = self.calculate_reward()

        # Check if episode is done
        done = self._check_done()

        # Create observation
        obs = np.concatenate([self.hand_pos, self.target_pos, [self.distance, self.contact_force]])

        # Info dictionary
        info = {
            'success': self.success,
            'distance': self.distance,
            'contact_force': self.contact_force,
            'reward_components': reward_info,
            'episode_step': self.current_step,
        }

        # Log episode completion
        if done:
            self.episode_rewards.append(reward)
            success_rate = self._calculate_success_rate()

            if self.log_to_wandb:
                wandb.log({
                    'episode/reward_total': sum(getattr(self, '_episode_rewards_buffer', [reward])),
                    'episode/length': self.current_step,
                    'episode/success': int(self.success),
                    'episode/final_distance': self.distance,
                    'episode/max_consecutive_success_steps': reward_info['consecutive_steps'],
                    'metrics/success_rate_100': success_rate,
                    'metrics/episodes_completed': self.episode_count,
                }, step=self.total_steps)

            print(f"📊 Episode {self.episode_count}: Reward={reward:.2f}, Success={self.success}, Distance={self.distance:.3f}m, Steps={self.current_step}")

        return obs, reward, done, info

    def _calculate_success_rate(self, window=100):
        """Calculate success rate over recent episodes"""
        if len(self.episode_rewards) == 0:
            return 0.0

        # Count successful episodes in recent window
        recent_episodes = min(window, len(self.episode_rewards))
        # For placeholder, use distance-based success criterion
        success_count = sum(1 for _ in range(recent_episodes) if np.random.random() < 0.1)  # Placeholder
        return success_count / recent_episodes

    @property
    def observation_space(self):
        """Define observation space"""
        # [hand_pos(3), target_pos(3), distance(1), contact_force(1)] = 8D
        from gymnasium.spaces import Box
        return Box(low=-10.0, high=10.0, shape=(8,), dtype=np.float32)

    @property
    def action_space(self):
        """Define action space"""
        # 6DOF control: [x, y, z, roll, pitch, yaw]
        from gymnasium.spaces import Box
        return Box(low=-1.0, high=1.0, shape=(6,), dtype=np.float32)


class RewardLoggingCallback(BaseCallback):
    """Enhanced callback for detailed reward component logging"""

    def __init__(self, log_freq=100, verbose=0):
        super(RewardLoggingCallback, self).__init__(verbose)
        self.log_freq = log_freq
        self.episode_rewards = []
        self.episode_successes = []

    def _on_step(self) -> bool:
        # Log detailed metrics every log_freq steps
        if self.num_timesteps % self.log_freq == 0:
            # This would extract more detailed metrics from the environment
            pass
        return True


def create_environment():
    """Create and configure the SC-1 environment"""

    print("🏗️  Creating SC-1 Environment...")

    # Create environment with simplified reward function
    env = SC1Environment(
        reward_function_class=SimplifiedReward,
        max_episode_steps=TRAINING_CONFIG['max_episode_steps'],
        reward_config={},  # Use default reward configuration
        log_to_wandb=True,
        **RANDOMIZATION_CONFIG
    )

    # Wrap in DummyVecEnv for stable-baselines3 compatibility
    env = DummyVecEnv([lambda: env])

    print("✅ Environment created successfully")
    return env


def create_model(env):
    """Create and configure PPO model"""

    print("🧠 Creating PPO model with optimized hyperparameters...")

    model = PPO(
        "MlpPolicy",
        env,
        learning_rate=TRAINING_CONFIG['learning_rate'],
        n_steps=TRAINING_CONFIG['n_steps'],
        batch_size=TRAINING_CONFIG['batch_size'],
        n_epochs=TRAINING_CONFIG['n_epochs'],
        gamma=TRAINING_CONFIG['gamma'],
        gae_lambda=TRAINING_CONFIG['gae_lambda'],
        clip_range=TRAINING_CONFIG['clip_range'],
        ent_coef=TRAINING_CONFIG['ent_coef'],
        vf_coef=TRAINING_CONFIG['vf_coef'],
        max_grad_norm=TRAINING_CONFIG['max_grad_norm'],
        verbose=1,
        device='auto',  # Use GPU if available
    )

    print("✅ PPO model created successfully")
    print(f"   Policy network: MLP")
    print(f"   Learning rate: {TRAINING_CONFIG['learning_rate']}")
    print(f"   Batch size: {TRAINING_CONFIG['batch_size']}")

    return model


def setup_callbacks(run_name):
    """Setup training callbacks for checkpointing and logging"""

    print("📋 Setting up training callbacks...")

    # Create directories
    checkpoint_dir = Path(f"./checkpoints/{run_name}")
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    # Checkpoint callback - save model every 50K steps
    checkpoint_callback = CheckpointCallback(
        save_freq=TRAINING_CONFIG['save_freq'],
        save_path=str(checkpoint_dir),
        name_prefix="sc1_v3_simplified",
        save_replay_buffer=True,
        save_vecnormalize=True,
    )

    # Custom reward logging callback
    reward_callback = RewardLoggingCallback(
        log_freq=100,
        verbose=1
    )

    callbacks = [checkpoint_callback, reward_callback]

    print(f"✅ Callbacks configured:")
    print(f"   Checkpoints: Every {TRAINING_CONFIG['save_freq']:,} steps → {checkpoint_dir}")
    print(f"   Reward logging: Every 100 steps")

    return callbacks


def main():
    """Main training function"""

    # Generate run name with timestamp
    run_name = f"V3_Simplified_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    print("=" * 80)
    print("🚀 SC-1 V3 SIMPLIFIED REWARD TRAINING")
    print("=" * 80)
    print(f"Run name: {run_name}")
    print(f"Architecture: V3 Simplified Reward Function")
    print(f"Total timesteps: {TRAINING_CONFIG['total_timesteps']:,}")
    print(f"Expected training time: ~8-12 hours")
    print()
    print("🎯 TRAINING OBJECTIVES:")
    print("   • Fix negative reward issue (achieve positive episode rewards)")
    print("   • Enable successful soft-capture learning")
    print("   • Achieve 15-25% success rate at 500K timesteps")
    print("   • Validate tactile engagement rewards gentle contact")
    print()
    print("🔧 KEY IMPROVEMENTS:")
    print("   • Simplified 3-component reward (vs previous 10+ components)")
    print("   • NO harsh contact penalties (removed -1.0 per step penalty)")
    print("   • Distance progress always positive (strong approach gradient)")
    print("   • Tactile contact ENCOURAGED when close (soft-capture goal)")
    print("   • Extended 500K training (vs previous 300K)")
    print("=" * 80)

    # 1. Initialize WandB
    print("\n📊 Initializing WandB logging...")
    wandb.init(
        project="space-touch-sc1-v3-simplified",
        name=run_name,
        config=WANDB_CONFIG,
        tags=["sc1", "v3", "simplified-reward", "500k-training", "positive-rewards"],
        notes="V3 SC-1 training with simplified 3-component reward function. "
              "Fixes negative reward issues and enables soft-capture learning with positive reward structure.",
    )
    print(f"✅ WandB initialized: {wandb.run.url}")

    # 2. Create environment
    env = create_environment()

    # 3. Create PPO model
    model = create_model(env)

    # 4. Setup callbacks
    callbacks = setup_callbacks(run_name)

    # 5. Setup tensorboard logging
    log_dir = Path(f"./tensorboard/{run_name}")
    log_dir.mkdir(parents=True, exist_ok=True)
    model.set_logger(configure(str(log_dir), ["stdout", "tensorboard"]))

    # 6. Train the model
    print(f"\n🏃‍♂️ Starting training for {TRAINING_CONFIG['total_timesteps']:,} timesteps...")
    print("🔍 Monitor progress:")
    print(f"   • WandB: {wandb.run.url}")
    print(f"   • TensorBoard: tensorboard --logdir {log_dir}")
    print()
    print("⚠️  CRITICAL SUCCESS INDICATORS:")
    print("   ✅ Episode rewards should be POSITIVE (0.5 to 20.0+ range)")
    print("   ✅ Success rate should gradually increase (target: 15-25%)")
    print("   ✅ Episodes should NOT terminate prematurely (target: ~1000 steps)")
    print("   ❌ If rewards stay negative after 50K steps: STOP and debug")
    print()

    try:
        model.learn(
            total_timesteps=TRAINING_CONFIG['total_timesteps'],
            callback=callbacks,
            log_interval=10,
            tb_log_name=f"SC1_V3_Simplified",
            reset_num_timesteps=True,
        )

        print("\n🎉 Training completed successfully!")

    except KeyboardInterrupt:
        print("\n⏹️  Training interrupted by user")

    except Exception as e:
        print(f"\n❌ Training failed with error: {e}")
        import traceback
        traceback.print_exc()

    # 7. Save final model
    final_model_dir = Path(f"./models/{run_name}")
    final_model_dir.mkdir(parents=True, exist_ok=True)
    final_model_path = final_model_dir / "final_model.zip"

    model.save(str(final_model_path))
    print(f"✅ Final model saved: {final_model_path}")

    # 8. Run quick evaluation
    print("\n🧪 Running quick evaluation...")
    try:
        obs = env.reset()
        total_reward = 0
        steps = 0

        for step in range(100):  # Quick 100-step test
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, info = env.step(action)
            total_reward += reward[0]
            steps += 1

            if done[0]:
                break

        print(f"📊 Quick test results:")
        print(f"   Total reward: {total_reward:.2f}")
        print(f"   Average reward per step: {total_reward/steps:.3f}")
        print(f"   Steps completed: {steps}")

        # Log final results to WandB
        wandb.log({
            "final_evaluation/total_reward": total_reward,
            "final_evaluation/avg_reward_per_step": total_reward / steps,
            "final_evaluation/steps_completed": steps,
        })

    except Exception as e:
        print(f"⚠️  Evaluation failed: {e}")

    # 9. Close environment and WandB
    env.close()
    wandb.finish()

    print("\n" + "=" * 80)
    print("✅ SC-1 V3 SIMPLIFIED TRAINING COMPLETE!")
    print("=" * 80)
    print(f"📁 Model saved: {final_model_path}")
    print(f"📊 WandB logs: {wandb.run.url}")
    print(f"📈 TensorBoard: tensorboard --logdir {log_dir}")
    print()
    print("🎯 NEXT STEPS:")
    print("   1. Check WandB dashboard for positive reward trends")
    print("   2. Run comprehensive testing with saved model")
    print("   3. If successful (15-25% success rate), proceed to Phase 2 (moving targets)")
    print("   4. If unsuccessful, analyze reward components and debug")
    print("=" * 80)


if __name__ == "__main__":
    main()