#!/usr/bin/env python3
"""
V7.4 Testing Script - Comprehensive Model Evaluation

Tests trained V7.4 or V7.4.1 models across multiple scenarios.
Outputs console statistics for easy EC2 monitoring.

Usage:
    python v7_4_test.py MODEL_PATH --episodes 10
    python v7_4_test.py MODEL_PATH --episodes 10 --version 7.4.1
"""

import os
import sys
import argparse
import numpy as np
from pathlib import Path

# Import the environment (try both V7.4 and V7.4.1)
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def import_environment(version='7.4'):
    """Import the correct environment based on version"""
    if version == '7.4':
        from v7_4_sc1 import V741Environment
        return V741Environment
    elif version == '7.4.1':
        from v7_4_1_sc1 import V7411Environment
        return V7411Environment
    else:
        raise ValueError(f"Unknown version: {version}")


def test_model(model_path: str, episodes: int = 10, version: str = '7.4'):
    """
    Test a trained model across multiple scenarios

    Args:
        model_path: Path to model checkpoint
        episodes: Number of episodes per scenario
        version: Model version ('7.4' or '7.4.1')
    """
    from stable_baselines3 import PPO

    print("=" * 70)
    print(f"V7.4 MODEL TESTING - Version {version}")
    print("=" * 70)
    print(f"Model: {model_path}")
    print(f"Episodes per scenario: {episodes}")
    print(f"Version: {version}")
    print()

    # Import correct environment
    if version == '7.4':
        from v7_4_sc1 import V741Environment as EnvClass
    else:
        from v7_4_1_sc1 import V7411Environment as EnvClass

    # Load model
    env = EnvClass(vis=False, max_steps=1000)
    model = PPO.load(model_path, env=env)

    print("Model loaded successfully!")
    print()

    # Define test scenarios
    scenarios = [
        {"name": "Standard (10cm)", "description": "Normal starting distance"},
        {"name": "Close (5cm)", "description": "Close starting distance"},
        {"name": "Far (15cm)", "description": "Far starting distance"},
        {"name": "Very Far (20cm)", "description": "Very far starting distance"},
    ]

    all_results = {}

    # Test each scenario
    for scenario_idx, scenario in enumerate(scenarios):
        print(f"[Scenario {scenario_idx+1}/{len(scenarios)}] {scenario['name']}")
        print(f"  Description: {scenario['description']}")
        print(f"  Running {episodes} episodes...")
        print()

        # Run episodes
        episode_results = []

        for ep in range(episodes):
            obs = env.reset()
            done = False
            step_count = 0
            total_reward = 0
            max_overlap = 0
            min_distance = float('inf')
            contact_steps = 0
            final_overlap = 0
            final_distance = 0

            while not done and step_count < 1000:
                action, _ = model.predict(obs, deterministic=True)
                obs, reward, done, info = env.step(action)

                if 'reward_info' in info:
                    r_info = info['reward_info']
                    overlap_cm3 = r_info.get('overlap_volume', 0) * 1e6
                    distance = r_info.get('distance_to_target', 0)
                    contacts = r_info.get('num_contacts', 0)

                    max_overlap = max(max_overlap, overlap_cm3)
                    min_distance = min(min_distance, distance)
                    if contacts > 0:
                        contact_steps += 1

                    final_overlap = overlap_cm3
                    final_distance = distance

                total_reward += reward
                step_count += 1

            success = (max_overlap > 50.0 and contact_steps < step_count * 0.1)

            episode_results.append({
                'episode': ep + 1,
                'total_reward': total_reward,
                'max_overlap_cm3': max_overlap,
                'final_overlap_cm3': final_overlap,
                'min_distance': min_distance,
                'final_distance': final_distance,
                'contact_steps': contact_steps,
                'total_steps': step_count,
                'success': success,
            })

            # Print episode summary
            status = "✓ SUCCESS" if success else "✗ FAIL"
            print(f"    Ep {ep+1:2d}: {status} | Overlap: {max_overlap:6.1f}cm³ | "
                  f"Distance: {min_distance:.3f}m | Contacts: {contact_steps:3d}/{step_count}")

        # Calculate scenario statistics
        avg_reward = np.mean([r['total_reward'] for r in episode_results])
        avg_max_overlap = np.mean([r['max_overlap_cm3'] for r in episode_results])
        avg_final_overlap = np.mean([r['final_overlap_cm3'] for r in episode_results])
        avg_min_distance = np.mean([r['min_distance'] for r in episode_results])
        avg_final_distance = np.mean([r['final_distance'] for r in episode_results])
        avg_contact_steps = np.mean([r['contact_steps'] for r in episode_results])
        success_rate = np.mean([r['success'] for r in episode_results]) * 100

        std_max_overlap = np.std([r['max_overlap_cm3'] for r in episode_results])
        std_min_distance = np.std([r['min_distance'] for r in episode_results])

        all_results[scenario['name']] = {
            'episodes': episode_results,
            'avg_reward': avg_reward,
            'avg_max_overlap': avg_max_overlap,
            'std_max_overlap': std_max_overlap,
            'avg_final_overlap': avg_final_overlap,
            'avg_min_distance': avg_min_distance,
            'std_min_distance': std_min_distance,
            'avg_final_distance': avg_final_distance,
            'avg_contact_steps': avg_contact_steps,
            'success_rate': success_rate,
        }

        print()
        print(f"  Scenario Summary:")
        print(f"    Success Rate:     {success_rate:5.1f}%")
        print(f"    Avg Max Overlap:  {avg_max_overlap:6.1f} ± {std_max_overlap:5.1f} cm³")
        print(f"    Avg Final Overlap: {avg_final_overlap:6.1f} cm³")
        print(f"    Avg Min Distance:  {avg_min_distance:.3f} ± {std_min_distance:.3f} m")
        print(f"    Avg Final Distance: {avg_final_distance:.3f} m")
        print(f"    Avg Contact Steps: {avg_contact_steps:.1f}")
        print(f"    Avg Reward:        {avg_reward:7.1f}")
        print()
        print("-" * 70)
        print()

    # Overall summary
    print("=" * 70)
    print("OVERALL SUMMARY")
    print("=" * 70)
    print()

    overall_success = np.mean([s['success_rate'] for s in all_results.values()])
    overall_overlap = np.mean([s['avg_max_overlap'] for s in all_results.values()])
    overall_distance = np.mean([s['avg_min_distance'] for s in all_results.values()])

    print(f"Average across all scenarios:")
    print(f"  Overall Success Rate:  {overall_success:5.1f}%")
    print(f"  Overall Max Overlap:   {overall_overlap:6.1f} cm³")
    print(f"  Overall Min Distance:  {overall_distance:.3f} m")
    print()

    print("Per-scenario breakdown:")
    print(f"  {'Scenario':<20} {'Success %':>10} {'Max Overlap':>12} {'Min Distance':>13}")
    print(f"  {'-'*20} {'-'*10} {'-'*12} {'-'*13}")
    for name, results in all_results.items():
        print(f"  {name:<20} {results['success_rate']:>9.1f}% "
              f"{results['avg_max_overlap']:>11.1f}cm³ "
              f"{results['avg_min_distance']:>12.3f}m")
    print()

    # Key findings
    print("KEY FINDINGS:")
    best_scenario = max(all_results.items(), key=lambda x: x[1]['success_rate'])
    worst_scenario = min(all_results.items(), key=lambda x: x[1]['success_rate'])
    print(f"  Best performance:  {best_scenario[0]} ({best_scenario[1]['success_rate']:.1f}% success)")
    print(f"  Worst performance: {worst_scenario[0]} ({worst_scenario[1]['success_rate']:.1f}% success)")
    print()

    if overall_overlap > 100:
        print(f"  ✓ Good overlap achieved! ({overall_overlap:.1f} cm³)")
    else:
        print(f"  ✗ Low overlap. ({overall_overlap:.1f} cm³)")

    if overall_distance < 0.15:
        print(f"  ✓ Good distance control! ({overall_distance:.3f}m)")
    else:
        print(f"  ✗ Poor distance control. ({overall_distance:.3f}m)")

    print()
    print("=" * 70)
    print("Testing complete!")
    print("=" * 70)

    return all_results


def main():
    parser = argparse.ArgumentParser(description='V7.4 Model Testing')
    parser.add_argument('model', type=str, help='Path to model checkpoint (e.g., final_model.zip)')
    parser.add_argument('--episodes', type=int, default=10, help='Episodes per scenario (default: 10)')
    parser.add_argument('--version', type=str, default='7.4', choices=['7.4', '7.4.1'],
                        help='Model version (default: 7.4)')

    args = parser.parse_args()

    if not os.path.exists(args.model):
        print(f"Error: Model file not found: {args.model}")
        sys.exit(1)

    test_model(args.model, args.episodes, args.version)


if __name__ == "__main__":
    main()
