#!/usr/bin/env python3
"""
Training script for Allegro Hand with tactile sensing using Stable Baselines3
"""

import numpy as np
from stable_baselines3 import PPO, SAC
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
import torch
import torch.nn as nn
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

# Import the custom environment
from allegro_hand_env import AllegroHandTactileEnv


class TactileFeatureExtractor(BaseFeaturesExtractor):
    """
    Custom CNN feature extractor for tactile images.
    Processes multiple grayscale tactile sensor images.
    """
    
    def __init__(self, observation_space, features_dim=256):
        super().__init__(observation_space, features_dim)
        
        n_sensors = observation_space.shape[0]
        height = observation_space.shape[1]
        width = observation_space.shape[2]
        
        # CNN for processing each tactile image
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Flatten()
        )
        
        # Calculate CNN output size
        with torch.no_grad():
            sample = torch.zeros(1, 1, height, width)
            cnn_out_size = self.cnn(sample).shape[1]
        
        # Combine features from all sensors
        self.fc = nn.Sequential(
            nn.Linear(cnn_out_size * n_sensors, 512),
            nn.ReLU(),
            nn.Linear(512, features_dim),
            nn.ReLU()
        )
        
        self.n_sensors = n_sensors

    def forward(self, observations):
        batch_size = observations.shape[0]
        
        # Process each sensor's image through CNN
        sensor_features = []
        for i in range(self.n_sensors):
            sensor_img = observations[:, i:i+1, :, :]  # Keep channel dimension
            features = self.cnn(sensor_img.float() / 255.0)  # Normalize to [0,1]
            sensor_features.append(features)
        
        # Concatenate all sensor features
        combined = torch.cat(sensor_features, dim=1)
        
        # Final processing
        return self.fc(combined)


def make_env(rank, seed=0, vis=False):
    """
    Utility function for multiprocessed env.
    """
    def _init():
        env = AllegroHandTactileEnv(
            num_envs=1,
            vis=vis and rank == 0,  # Only first env has visualization
            max_steps=500,
            sensor_res=(60, 80),
            reset_on_completion=True
        )
        env.seed(seed + rank)
        return env
    return _init


def train_allegro_hand(
    total_timesteps=1_000_000,
    n_envs=4,
    learning_rate=3e-4,
    batch_size=64,
    n_epochs=10,
    gamma=0.99,
    save_freq=10000,
    eval_freq=5000,
    vis_training=False,
    vis_eval=True,
    algorithm='ppo'  # 'ppo' or 'sac'
):
    """
    Train the Allegro hand to grasp objects using RL.
    """
    
    # Create training environments
    if n_envs > 1:
        env = SubprocVecEnv([make_env(i, vis=vis_training) for i in range(n_envs)])
    else:
        env = make_env(0, vis=vis_training)()
    
    # Create evaluation environment
    eval_env = make_env(0, seed=100, vis=vis_eval)()
    
    # Define policy kwargs with custom feature extractor
    policy_kwargs = dict(
        features_extractor_class=TactileFeatureExtractor,
        features_extractor_kwargs=dict(features_dim=256),
        net_arch=[256, 256]  # Actor and critic networks
    )
    
    # Create model
    if algorithm.lower() == 'ppo':
        model = PPO(
            "CnnPolicy",
            env,
            learning_rate=learning_rate,
            n_steps=2048,
            batch_size=batch_size,
            n_epochs=n_epochs,
            gamma=gamma,
            verbose=1,
            policy_kwargs=policy_kwargs,
            tensorboard_log="./allegro_hand_tensorboard/"
        )
    elif algorithm.lower() == 'sac':
        model = SAC(
            "CnnPolicy",
            env,
            learning_rate=learning_rate,
            buffer_size=100000,
            learning_starts=1000,
            batch_size=batch_size,
            gamma=gamma,
            verbose=1,
            policy_kwargs=policy_kwargs,
            tensorboard_log="./allegro_hand_tensorboard/"
        )
    else:
        raise ValueError(f"Unknown algorithm: {algorithm}")
    
    # Callbacks
    checkpoint_callback = CheckpointCallback(
        save_freq=save_freq,
        save_path="./allegro_hand_checkpoints/",
        name_prefix=f"allegro_hand_{algorithm}"
    )
    
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path="./allegro_hand_best_model/",
        log_path="./allegro_hand_eval_logs/",
        eval_freq=eval_freq,
        deterministic=True,
        render=False
    )
    
    # Train
    print(f"Starting training with {algorithm.upper()}...")
    model.learn(
        total_timesteps=total_timesteps,
        callback=[checkpoint_callback, eval_callback],
        progress_bar=True
    )
    
    # Save final model
    model.save(f"allegro_hand_{algorithm}_final")
    
    return model


def evaluate_model(model_path, n_episodes=10, vis=True):
    """
    Evaluate a trained model.
    """
    # Load model
    if 'ppo' in model_path.lower():
        model = PPO.load(model_path)
    else:
        model = SAC.load(model_path)
    
    # Create environment
    env = AllegroHandTactileEnv(
        num_envs=1,
        vis=vis,
        max_steps=500,
        sensor_res=(60, 80)
    )
    
    # Evaluate
    episode_rewards = []
    episode_successes = []
    
    for episode in range(n_episodes):
        obs = env.reset()
        done = False
        total_reward = 0
        
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, info = env.step(action)
            total_reward += reward[0]
        
        episode_rewards.append(total_reward)
        episode_successes.append(info[0]['success'])
        
        print(f"Episode {episode + 1}: "
              f"Reward = {total_reward:.2f}, "
              f"Success = {info[0]['success']}")
    
    print(f"\nEvaluation Summary:")
    print(f"Mean Reward: {np.mean(episode_rewards):.2f} ± {np.std(episode_rewards):.2f}")
    print(f"Success Rate: {np.mean(episode_successes) * 100:.1f}%")
    
    env.close()


if __name__ == "__main__":
    # Training configuration
    config = {
        'total_timesteps': 500_000,
        'n_envs': 4,  # Number of parallel environments
        'learning_rate': 3e-4,
        'batch_size': 64,
        'n_epochs': 10,
        'gamma': 0.99,
        'save_freq': 10000,
        'eval_freq': 5000,
        'vis_training': False,  # Set True to visualize during training
        'vis_eval': True,
        'algorithm': 'ppo'  # or 'sac'
    }
    
    # Train
    model = train_allegro_hand(**config)
    
    # Evaluate
    print("\nEvaluating trained model...")
    evaluate_model(f"allegro_hand_{config['algorithm']}_final", n_episodes=10, vis=True)