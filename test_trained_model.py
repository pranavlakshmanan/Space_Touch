#!/usr/bin/env python3
"""
Test trained SC-1 V3 model with simplified reward function
Evaluate performance across multiple scenarios and distances
"""

import os
import sys
import argparse
import numpy as np
from pathlib import Path

sys.path.append('/home/pralak/Space_Touch')

from stable_baselines3 import PPO
from reward_functions.simplified_reward import SimplifiedReward
from train_v3_simplified import SC1Environment, create_environment


def test_scenarios():
    """Define test scenarios for comprehensive evaluation"""
    return {
        "close_easy": {
            "target_pos": [0.20, 0.00, 0.30],
            "description": "Close target - should achieve >50% success",
            "expected_success_rate": 0.5,
            "difficulty": "Easy"
        },
        "medium_standard": {
            "target_pos": [0.30, 0.10, 0.35],
            "description": "Medium distance - standard scenario",
            "expected_success_rate": 0.3,
            "difficulty": "Medium"
        },
        "far_challenging": {
            "target_pos": [0.45, 0.15, 0.40],
            "description": "Far target - challenging but possible",
            "expected_success_rate": 0.1,
            "difficulty": "Hard"
        },
        "precision_grasp": {
            "target_pos": [0.25, -0.10, 0.32],
            "description": "Precision placement test",
            "expected_success_rate": 0.2,
            "difficulty": "Medium-Hard"
        },
        "off_axis": {
            "target_pos": [0.35, 0.25, 0.45],
            "description": "Off-axis approach test",
            "expected_success_rate": 0.15,
            "difficulty": "Hard"
        }
    }


def run_single_episode(model, env, target_pos, max_steps=1000, verbose=False):
    """
    Run a single episode with specified target position

    Returns:
        dict: Episode results including success, reward, distance, etc.
    """

    # Set target position (this would be done through env interface)
    obs = env.reset()

    episode_rewards = []
    episode_distances = []
    episode_contacts = []
    success = False
    final_distance = 1.0

    for step in range(max_steps):
        # Get action from model
        action, _ = model.predict(obs, deterministic=True)

        # Take environment step
        obs, reward, done, info = env.step(action)

        # Track metrics
        episode_rewards.append(reward[0])

        if isinstance(info, list) and len(info) > 0:
            info_dict = info[0]
            episode_distances.append(info_dict.get('distance', 1.0))
            episode_contacts.append(info_dict.get('contact_force', 0.0))
            final_distance = info_dict.get('distance', 1.0)
            success = info_dict.get('success', False)

        if verbose and step % 100 == 0:
            print(f"   Step {step:3d}: Reward={reward[0]:6.2f}, Distance={final_distance:.3f}m")

        if done[0]:
            break

    # Calculate episode statistics
    total_reward = sum(episode_rewards)
    avg_reward = total_reward / len(episode_rewards) if episode_rewards else 0
    min_distance = min(episode_distances) if episode_distances else final_distance
    max_contact = max(episode_contacts) if episode_contacts else 0

    return {
        'success': success,
        'total_reward': total_reward,
        'avg_reward': avg_reward,
        'final_distance': final_distance,
        'min_distance': min_distance,
        'max_contact_force': max_contact,
        'episode_length': len(episode_rewards),
        'rewards': episode_rewards,
        'distances': episode_distances,
    }


def evaluate_scenario(model, env, scenario_name, scenario_config, num_episodes=20):
    """
    Evaluate model on a specific scenario across multiple episodes

    Returns:
        dict: Aggregated results for the scenario
    """

    print(f"\n📋 Testing scenario: {scenario_name}")
    print(f"   Description: {scenario_config['description']}")
    print(f"   Target position: {scenario_config['target_pos']}")
    print(f"   Expected success rate: {scenario_config['expected_success_rate']:.1%}")
    print(f"   Running {num_episodes} episodes...")

    results = []

    for episode in range(num_episodes):
        result = run_single_episode(
            model, env, scenario_config['target_pos'],
            max_steps=1000, verbose=(episode == 0)  # Verbose for first episode only
        )
        results.append(result)

        # Print progress every 5 episodes
        if (episode + 1) % 5 == 0:
            success_so_far = sum(1 for r in results if r['success'])
            print(f"   Progress: {episode + 1}/{num_episodes} episodes, "
                  f"{success_so_far} successes ({success_so_far/(episode+1):.1%})")

    # Calculate aggregated statistics
    successes = sum(1 for r in results if r['success'])
    success_rate = successes / num_episodes

    total_rewards = [r['total_reward'] for r in results]
    final_distances = [r['final_distance'] for r in results]
    episode_lengths = [r['episode_length'] for r in results]
    min_distances = [r['min_distance'] for r in results]

    aggregated = {
        'scenario_name': scenario_name,
        'num_episodes': num_episodes,
        'success_count': successes,
        'success_rate': success_rate,
        'expected_success_rate': scenario_config['expected_success_rate'],
        'difficulty': scenario_config['difficulty'],

        # Reward statistics
        'avg_total_reward': np.mean(total_rewards),
        'std_total_reward': np.std(total_rewards),
        'min_total_reward': np.min(total_rewards),
        'max_total_reward': np.max(total_rewards),

        # Distance statistics
        'avg_final_distance': np.mean(final_distances),
        'std_final_distance': np.std(final_distances),
        'avg_min_distance': np.mean(min_distances),
        'best_min_distance': np.min(min_distances),

        # Episode length statistics
        'avg_episode_length': np.mean(episode_lengths),
        'std_episode_length': np.std(episode_lengths),

        # Performance assessment
        'performance_vs_expected': success_rate / scenario_config['expected_success_rate'],
    }

    # Print results
    print(f"\n📊 Results for {scenario_name}:")
    print(f"   Success rate: {success_rate:.1%} (expected: {scenario_config['expected_success_rate']:.1%})")
    print(f"   Performance ratio: {aggregated['performance_vs_expected']:.2f}")
    print(f"   Average total reward: {aggregated['avg_total_reward']:.2f} ± {aggregated['std_total_reward']:.2f}")
    print(f"   Average final distance: {aggregated['avg_final_distance']:.3f}m ± {aggregated['std_final_distance']:.3f}m")
    print(f"   Best approach distance: {aggregated['best_min_distance']:.3f}m")
    print(f"   Average episode length: {aggregated['avg_episode_length']:.0f} ± {aggregated['std_episode_length']:.0f} steps")

    return aggregated


def evaluate_model(model_path, num_episodes_per_scenario=20):
    """
    Comprehensive evaluation of trained model across all test scenarios

    Args:
        model_path: Path to trained model (.zip file)
        num_episodes_per_scenario: Number of episodes to run per scenario

    Returns:
        dict: Complete evaluation results
    """

    print("=" * 80)
    print("🧪 COMPREHENSIVE MODEL EVALUATION")
    print("=" * 80)
    print(f"Model path: {model_path}")
    print(f"Episodes per scenario: {num_episodes_per_scenario}")

    # Load trained model
    print(f"\n📂 Loading trained model...")
    try:
        model = PPO.load(model_path)
        print(f"✅ Model loaded successfully")
    except Exception as e:
        print(f"❌ Failed to load model: {e}")
        return None

    # Create environment
    print(f"\n🏗️  Creating evaluation environment...")
    env = create_environment()
    print(f"✅ Environment created")

    # Get test scenarios
    scenarios = test_scenarios()

    # Run evaluation for each scenario
    scenario_results = {}
    overall_successes = 0
    overall_episodes = 0

    for scenario_name, scenario_config in scenarios.items():
        result = evaluate_scenario(
            model, env, scenario_name, scenario_config,
            num_episodes_per_scenario
        )
        scenario_results[scenario_name] = result
        overall_successes += result['success_count']
        overall_episodes += result['num_episodes']

    # Calculate overall statistics
    overall_success_rate = overall_successes / overall_episodes

    # Performance assessment
    print("\n" + "=" * 80)
    print("📊 OVERALL EVALUATION SUMMARY")
    print("=" * 80)

    print(f"Overall success rate: {overall_success_rate:.1%} ({overall_successes}/{overall_episodes})")
    print()

    # Scenario-by-scenario summary
    print("📋 Scenario-by-scenario results:")
    print(f"{'Scenario':<15} {'Success Rate':<12} {'Expected':<10} {'Ratio':<8} {'Avg Reward':<12} {'Difficulty'}")
    print("-" * 75)

    for scenario_name, result in scenario_results.items():
        print(f"{scenario_name:<15} "
              f"{result['success_rate']:.1%}{'':>7} "
              f"{result['expected_success_rate']:.1%}{'':>5} "
              f"{result['performance_vs_expected']:.2f}{'':>4} "
              f"{result['avg_total_reward']:.1f}{'':>8} "
              f"{result['difficulty']}")

    # Performance assessment
    print(f"\n🎯 PERFORMANCE ASSESSMENT:")

    # Check if model meets success criteria
    phase1_target_success_rate = 0.20  # 20% target for Phase 1

    if overall_success_rate >= phase1_target_success_rate:
        print(f"✅ SUCCESS: Model exceeds Phase 1 target ({overall_success_rate:.1%} >= {phase1_target_success_rate:.1%})")
        print(f"🚀 Ready to proceed to Phase 2 (moving targets)")
    elif overall_success_rate >= phase1_target_success_rate * 0.7:
        print(f"⚠️  PARTIAL SUCCESS: Model shows promise ({overall_success_rate:.1%})")
        print(f"🔧 Consider additional training or hyperparameter tuning")
    else:
        print(f"❌ NEEDS IMPROVEMENT: Model below target ({overall_success_rate:.1%} < {phase1_target_success_rate:.1%})")
        print(f"🔧 Requires debugging or reward function adjustment")

    # Check for positive rewards
    avg_rewards = [r['avg_total_reward'] for r in scenario_results.values()]
    overall_avg_reward = np.mean(avg_rewards)

    if overall_avg_reward > 5.0:
        print(f"✅ REWARD STRUCTURE: Positive rewards achieved (avg: {overall_avg_reward:.1f})")
    elif overall_avg_reward > 0:
        print(f"⚠️  REWARD STRUCTURE: Barely positive rewards (avg: {overall_avg_reward:.1f})")
    else:
        print(f"❌ REWARD STRUCTURE: Still negative rewards (avg: {overall_avg_reward:.1f})")

    # Environment close
    env.close()

    # Return complete results
    evaluation_results = {
        'model_path': model_path,
        'overall_success_rate': overall_success_rate,
        'overall_avg_reward': overall_avg_reward,
        'total_episodes': overall_episodes,
        'total_successes': overall_successes,
        'scenario_results': scenario_results,
        'meets_phase1_target': overall_success_rate >= phase1_target_success_rate,
    }

    return evaluation_results


def main():
    """Main evaluation function"""

    parser = argparse.ArgumentParser(description='Test trained SC-1 V3 model')
    parser.add_argument('--model', '-m', type=str, required=True,
                        help='Path to trained model (.zip file)')
    parser.add_argument('--episodes', '-e', type=int, default=20,
                        help='Number of episodes per scenario (default: 20)')
    parser.add_argument('--output', '-o', type=str, default=None,
                        help='Output file for results (JSON format)')

    args = parser.parse_args()

    # Validate model path
    model_path = Path(args.model)
    if not model_path.exists():
        print(f"❌ Model file not found: {model_path}")
        return 1

    # Run evaluation
    results = evaluate_model(str(model_path), args.episodes)

    if results is None:
        print(f"❌ Evaluation failed")
        return 1

    # Save results if requested
    if args.output:
        import json
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Convert numpy types for JSON serialization
        def convert_numpy(obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.integer):
                return int(obj)
            return obj

        # Recursively convert numpy types
        def convert_dict(d):
            if isinstance(d, dict):
                return {k: convert_dict(v) for k, v in d.items()}
            elif isinstance(d, list):
                return [convert_dict(v) for v in d]
            else:
                return convert_numpy(d)

        results_json = convert_dict(results)

        with open(output_path, 'w') as f:
            json.dump(results_json, f, indent=2)

        print(f"\n💾 Results saved to: {output_path}")

    print(f"\n🏁 Evaluation complete!")

    return 0


if __name__ == "__main__":
    exit(main())