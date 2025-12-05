#!/usr/bin/env python3
"""Quick headless test of the latest model"""

import sys
import os
sys.path.append('/home/pralak/Space_Touch/Code_Pranav/RL Code')

# Import our test script but force headless mode
os.environ['DISPLAY'] = ''  # Force headless

from test_latest_model import *

def quick_test():
    """Run a quick headless test"""
    print("🚀 QUICK HEADLESS MODEL TEST")
    print("=" * 50)

    # Find latest model
    model_path = find_latest_model()
    if model_path is None:
        return

    # Load model
    try:
        print(f"📂 Loading: {model_path.name}")
        dummy_env = TendonAllegroTestEnv(vis=False)
        model = PPO.load(str(model_path), env=dummy_env)
        dummy_env.close()
        print("✅ Model loaded!")
    except Exception as e:
        print(f"❌ Error: {e}")
        return

    # Quick test - just one scenario, 3 episodes
    print(f"\n🧪 Running quick test (3 episodes)...")

    env_test = TendonAllegroTestEnv(vis=False, test_scenario="static_close")
    results = []

    for episode in range(3):
        print(f"  Episode {episode+1}/3... ", end="")

        obs = env_test.reset()
        episode_reward = 0
        episode_steps = 0

        while episode_steps < 500:  # Shorter episodes
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, info = env_test.step(action)
            episode_reward += reward[0]
            episode_steps += 1

            if done[0]:
                break

        # Get results
        distance = info[0].get('distance', float('inf'))
        success = info[0].get('success', False)
        num_fingers = info[0].get('num_active_fingers', 0)

        results.append({
            'distance': distance,
            'success': success,
            'reward': episode_reward,
            'steps': episode_steps,
            'fingers': num_fingers
        })

        status = "✅ SUCCESS" if success else "❌ FAILED"
        print(f"{status} | Dist: {distance:.3f}m | Fingers: {num_fingers}")

    env_test.close()

    # Summary
    successes = sum(r['success'] for r in results)
    avg_distance = np.mean([r['distance'] for r in results])
    avg_fingers = np.mean([r['fingers'] for r in results])
    avg_reward = np.mean([r['reward'] for r in results])

    print(f"\n📊 QUICK TEST RESULTS:")
    print(f"   Success Rate: {successes}/3 ({successes/3*100:.1f}%)")
    print(f"   Avg Distance: {avg_distance:.3f}m")
    print(f"   Avg Fingers: {avg_fingers:.1f}")
    print(f"   Avg Reward: {avg_reward:.1f}")

    # Assessment
    print(f"\n🎯 ASSESSMENT:")
    if successes >= 1:  # 33% for quick test
        print("✅ Model shows learning signs!")
    else:
        print("⚠️  Model may need more training")

    if avg_distance <= 0.4:
        print("✅ Distance performance reasonable")
    else:
        print("⚠️  Distance needs improvement")

if __name__ == "__main__":
    quick_test()