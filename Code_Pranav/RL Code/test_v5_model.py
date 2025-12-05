#!/usr/bin/env python3
"""
Test V5 ConvexHull trained models
"""

import sys
import numpy as np
import argparse
from pathlib import Path

sys.path.append('/home/pralak/Space_Touch')
sys.path.append('/home/pralak/Space_Touch/Code_Pranav/RL Code')

from stable_baselines3 import PPO
from V5_ConvexHull_Overlap_Training import ConvexHullOverlapEnv

def test_v5_model(model_path, episodes=5):
    """Test a V5 ConvexHull trained model"""

    print(f"🎯 Testing V5 Model: {model_path}")
    print(f"📊 Episodes: {episodes}")

    # Load model
    try:
        model = PPO.load(model_path)
        print("✅ Model loaded successfully")
    except Exception as e:
        print(f"❌ Failed to load model: {e}")
        return False

    # Create environment (same as training)
    try:
        env = ConvexHullOverlapEnv(num_envs=1, vis=False, max_steps=500)
        print("✅ Environment created")
    except Exception as e:
        print(f"❌ Failed to create environment: {e}")
        return False

    # Test episodes
    successes = 0
    total_rewards = []
    overlaps_detected = 0

    for episode in range(episodes):
        print(f"\n🧪 Episode {episode + 1}/{episodes}")

        obs = env.reset()
        total_reward = 0
        steps = 0
        max_overlap = 0
        episode_overlaps = 0

        for step in range(500):  # Max steps
            # Predict action
            action, _ = model.predict(obs, deterministic=True)

            # Take step
            obs, reward, done, info = env.step(action)

            total_reward += reward[0] if hasattr(reward, '__len__') else reward
            steps += 1

            # Check for overlap detection
            if hasattr(info, '__len__') and len(info) > 0:
                reward_info = info[0].get('reward_info', {})
                overlap_vol = reward_info.get('overlap_volume', 0)

                if overlap_vol > 0:
                    episode_overlaps += 1
                    max_overlap = max(max_overlap, overlap_vol)
                    if episode_overlaps == 1:  # First overlap of episode
                        print(f"   🎉 First overlap at step {step}: {overlap_vol:.9f} m³ ({overlap_vol*1e6:.4f} cm³)")

            if done[0] if hasattr(done, '__len__') else done:
                break

        # Episode summary
        total_rewards.append(total_reward)
        if episode_overlaps > 0:
            overlaps_detected += 1
            print(f"   ✅ SUCCESS: {episode_overlaps} overlaps detected, max: {max_overlap*1e6:.4f} cm³")
            successes += 1
        else:
            print(f"   ❌ No overlaps detected")

        print(f"   📊 Total reward: {total_reward:.2f}, Steps: {steps}")

    # Final results
    print(f"\n{'='*60}")
    print(f"🎯 FINAL RESULTS:")
    print(f"{'='*60}")
    print(f"   Episodes: {episodes}")
    print(f"   Successes (with overlap): {successes}")
    print(f"   Success rate: {successes/episodes*100:.1f}%")
    print(f"   Mean reward: {np.mean(total_rewards):.2f} ± {np.std(total_rewards):.2f}")
    print(f"   Episodes with overlap: {overlaps_detected}")

    if overlaps_detected > 0:
        print(f"   🎉 OVERLAP CAPABILITY CONFIRMED!")
    else:
        print(f"   ⚠️  No overlap detection - model may need more training")

    env.close()
    return successes > 0

def main():
    parser = argparse.ArgumentParser(description="Test V5 ConvexHull trained model")
    parser.add_argument("--model", "-m", required=True, help="Path to model file")
    parser.add_argument("--episodes", type=int, default=5, help="Number of test episodes")

    args = parser.parse_args()

    model_path = Path(args.model)
    if not model_path.exists():
        print(f"❌ Model file not found: {model_path}")
        return 1

    success = test_v5_model(str(model_path), args.episodes)
    return 0 if success else 1

if __name__ == "__main__":
    exit(main())