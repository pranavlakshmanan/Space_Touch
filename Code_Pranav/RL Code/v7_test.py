#!/usr/bin/env python3
"""
V7 Testing Script - Comprehensive Model Evaluation

Auto-finds latest trained V7 model and tests across multiple scenarios.
Generates detailed plots and CSV exports for analysis.

Usage:
    python v7_test.py                           # Auto-find latest model
    python v7_test.py --model path/to/model.zip # Test specific model
    python v7_test.py --episodes 10 --vis       # 10 episodes with visualization
"""

import os
import sys
import time
import argparse
from datetime import datetime
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

# Import V7 environment
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from v7_sc1 import V7Environment

from stable_baselines3 import PPO


def find_latest_v7_model():
    """Find the most recently trained V7 model"""
    runs_dir = Path("SC1_Training_Runs")

    if not runs_dir.exists():
        raise FileNotFoundError("No training runs found. Train a model first with: python v7_sc1.py train")

    # Find all V7 run directories
    v7_runs = [d for d in runs_dir.iterdir() if d.is_dir() and d.name.startswith("V7_SC1_")]

    if not v7_runs:
        raise FileNotFoundError("No V7 training runs found")

    # Sort by modification time (most recent first)
    v7_runs.sort(key=lambda x: x.stat().st_mtime, reverse=True)
    latest_run = v7_runs[0]

    # Check for final model
    final_model = latest_run / "final_model.zip"
    if final_model.exists():
        return str(final_model), latest_run

    # Check for interrupted model
    interrupted_model = latest_run / "interrupted_model.zip"
    if interrupted_model.exists():
        return str(interrupted_model), latest_run

    # Check for latest checkpoint
    checkpoint_dir = latest_run / "checkpoints"
    if checkpoint_dir.exists():
        checkpoints = list(checkpoint_dir.glob("*.zip"))
        if checkpoints:
            checkpoints.sort(key=lambda x: x.stat().st_mtime, reverse=True)
            return str(checkpoints[0]), latest_run

    raise FileNotFoundError(f"No model files found in {latest_run}")


def get_test_scenarios():
    """Define comprehensive test scenarios"""
    return {
        'ultra_close': {
            'target_pos': np.array([0.18, 0.15, 0.28]),
            'description': 'Ultra-close (Phase 0 difficulty)',
            'expected_success_rate': 0.8,
            'difficulty': 'Easy',
        },
        'close_standard': {
            'target_pos': np.array([0.22, 0.15, 0.32]),
            'description': 'Close standard position',
            'expected_success_rate': 0.7,
            'difficulty': 'Easy',
        },
        'medium_distance': {
            'target_pos': np.array([0.25, 0.15, 0.35]),
            'description': 'Medium distance (training position)',
            'expected_success_rate': 0.6,
            'difficulty': 'Medium',
        },
        'far_reach': {
            'target_pos': np.array([0.30, 0.15, 0.40]),
            'description': 'Far reach',
            'expected_success_rate': 0.4,
            'difficulty': 'Hard',
        },
        'side_reach_left': {
            'target_pos': np.array([0.25, 0.10, 0.35]),
            'description': 'Side reach (left)',
            'expected_success_rate': 0.5,
            'difficulty': 'Medium',
        },
        'side_reach_right': {
            'target_pos': np.array([0.25, 0.20, 0.35]),
            'description': 'Side reach (right)',
            'expected_success_rate': 0.5,
            'difficulty': 'Medium',
        },
        'high_target': {
            'target_pos': np.array([0.25, 0.15, 0.42]),
            'description': 'High target',
            'expected_success_rate': 0.4,
            'difficulty': 'Hard',
        },
        'low_target': {
            'target_pos': np.array([0.25, 0.15, 0.30]),
            'description': 'Low target',
            'expected_success_rate': 0.6,
            'difficulty': 'Medium',
        },
    }


def test_model(model_path, episodes_per_scenario=5, visualize=False):
    """Test model across all scenarios"""

    print(f"{'='*80}")
    print(f"V7 COMPREHENSIVE MODEL TESTING")
    print(f"{'='*80}")
    print(f"Model: {model_path}")
    print(f"Episodes per scenario: {episodes_per_scenario}")
    print(f"Visualization: {visualize}")
    print()

    # Create test results directory
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    test_dir = Path(f"V7_Test_Results_{timestamp}")
    test_dir.mkdir(parents=True, exist_ok=True)
    print(f"Results directory: {test_dir}")
    print()

    # Load model and environment
    env = V7Environment(vis=visualize, max_steps=1000)
    model = PPO.load(model_path, env=env)

    scenarios = get_test_scenarios()

    # Storage for all test data
    all_episodes_data = []
    scenario_summaries = []

    # Test each scenario
    for scenario_name, scenario_config in scenarios.items():
        print(f"{'-'*80}")
        print(f"Testing: {scenario_config['description']} [{scenario_config['difficulty']}]")
        print(f"Target: {scenario_config['target_pos']}")
        print(f"{'-'*80}")

        scenario_episodes = []

        for episode in range(episodes_per_scenario):
            # Override target position
            env.target_pos = scenario_config['target_pos'].copy()

            obs = env.reset()
            episode_data = []
            episode_reward = 0

            for step in range(1000):
                action, _ = model.predict(obs, deterministic=True)
                obs, reward, done, info = env.step(action)

                episode_reward += reward[0]

                # Collect detailed step data
                reward_info = info[0].get('reward_info', {})
                step_data = {
                    'scenario': scenario_name,
                    'difficulty': scenario_config['difficulty'],
                    'episode': episode + 1,
                    'step': step + 1,
                    'reward': reward[0],
                    'cumulative_reward': episode_reward,

                    # Hull metrics
                    'overlap_volume_cm3': reward_info.get('overlap_volume', 0) * 1e6,
                    'hand_volume_cm3': reward_info.get('hand_hull_volume', 0) * 1e6,
                    'object_volume_cm3': reward_info.get('object_hull_volume', 0) * 1e6,

                    # State metrics
                    'distance_m': reward_info.get('distance_to_target', 0),
                    'distance_cm': reward_info.get('distance_to_target', 0) * 100,
                    'num_contacts': reward_info.get('num_contacts', 0),
                    'consecutive_success': reward_info.get('consecutive_success_steps', 0),
                    'is_success': reward_info.get('is_success', False),

                    # Reward components
                    'reward_overlap': reward_info.get('overlap_reward', 0),
                    'reward_proximity': reward_info.get('proximity_reward', 0),
                    'reward_contact': reward_info.get('contact_penalty', 0),
                    'reward_clearance': reward_info.get('clearance_reward', 0),
                    'reward_quality': reward_info.get('quality_reward', 0),

                    # Phase info
                    'phase': reward_info.get('current_phase', 0),
                }

                episode_data.append(step_data)

                if visualize:
                    time.sleep(0.01)

                if done[0]:
                    break

            # Episode summary
            final_step = episode_data[-1]
            max_overlap = max(d['overlap_volume_cm3'] for d in episode_data)
            min_distance = min(d['distance_cm'] for d in episode_data)
            total_contacts = sum(d['num_contacts'] for d in episode_data)
            max_consecutive_success = max(d['consecutive_success'] for d in episode_data)

            episode_summary = {
                'scenario': scenario_name,
                'difficulty': scenario_config['difficulty'],
                'episode': episode + 1,
                'total_reward': episode_reward,
                'steps': len(episode_data),
                'final_distance_cm': final_step['distance_cm'],
                'min_distance_cm': min_distance,
                'max_overlap_cm3': max_overlap,
                'final_overlap_cm3': final_step['overlap_volume_cm3'],
                'total_contacts': total_contacts,
                'max_consecutive_success': max_consecutive_success,
                'success': final_step['is_success'],
            }

            scenario_episodes.append(episode_summary)
            all_episodes_data.extend(episode_data)

            # Print episode summary
            success_icon = '✓' if episode_summary['success'] else '✗'
            print(f"  Ep {episode+1:2d}: Reward={episode_reward:7.2f}, Steps={len(episode_data):4d}, "
                  f"FinalDist={final_step['distance_cm']:5.2f}cm, MaxOverlap={max_overlap:6.2f}cm³, "
                  f"Contacts={total_contacts:3d}, Success={success_icon}")

        # Scenario summary
        success_rate = sum(ep['success'] for ep in scenario_episodes) / len(scenario_episodes)
        avg_reward = np.mean([ep['total_reward'] for ep in scenario_episodes])
        avg_final_distance = np.mean([ep['final_distance_cm'] for ep in scenario_episodes])
        avg_max_overlap = np.mean([ep['max_overlap_cm3'] for ep in scenario_episodes])
        avg_contacts = np.mean([ep['total_contacts'] for ep in scenario_episodes])

        scenario_summary = {
            'scenario': scenario_name,
            'description': scenario_config['description'],
            'difficulty': scenario_config['difficulty'],
            'success_rate': success_rate,
            'expected_success_rate': scenario_config['expected_success_rate'],
            'avg_reward': avg_reward,
            'avg_final_distance_cm': avg_final_distance,
            'avg_max_overlap_cm3': avg_max_overlap,
            'avg_total_contacts': avg_contacts,
            'episodes_tested': episodes_per_scenario,
        }

        scenario_summaries.append(scenario_summary)

        print(f"  Summary: Success={success_rate:.1%}, AvgReward={avg_reward:.2f}, "
              f"AvgDist={avg_final_distance:.2f}cm, AvgOverlap={avg_max_overlap:.2f}cm³")
        print()

    env.close()

    # Save data
    print(f"Saving results...")

    # Detailed step data
    detailed_df = pd.DataFrame(all_episodes_data)
    detailed_csv = test_dir / "test_results_detailed.csv"
    detailed_df.to_csv(detailed_csv, index=False)
    print(f"  Detailed data: {detailed_csv}")

    # Scenario summaries
    summary_df = pd.DataFrame(scenario_summaries)
    summary_csv = test_dir / "test_results_summary.csv"
    summary_df.to_csv(summary_csv, index=False)
    print(f"  Summary data: {summary_csv}")

    # Generate plots
    print(f"\nGenerating plots...")
    generate_comprehensive_plots(detailed_df, summary_df, test_dir)

    print(f"\n{'='*80}")
    print(f"TESTING COMPLETE")
    print(f"{'='*80}")
    print(f"Results saved to: {test_dir}")
    print(f"\nOverall Performance:")
    print(f"  Average success rate: {summary_df['success_rate'].mean():.1%}")
    print(f"  Average reward: {summary_df['avg_reward'].mean():.2f}")
    print(f"  Average max overlap: {summary_df['avg_max_overlap_cm3'].mean():.2f} cm³")
    print()

    return test_dir, summary_df


def generate_comprehensive_plots(detailed_df, summary_df, output_dir):
    """Generate comprehensive analysis plots"""

    # Plot 1: Overview Dashboard (4x3 grid)
    fig = plt.figure(figsize=(20, 16))
    gs = GridSpec(4, 3, figure=fig, hspace=0.3, wspace=0.3)

    # 1. Success Rate by Scenario
    ax1 = fig.add_subplot(gs[0, 0])
    scenarios = summary_df['scenario'].values
    success_rates = summary_df['success_rate'].values * 100
    expected_rates = summary_df['expected_success_rate'].values * 100
    x = np.arange(len(scenarios))
    width = 0.35
    ax1.bar(x - width/2, success_rates, width, label='Actual', color='#2ecc71', alpha=0.8)
    ax1.bar(x + width/2, expected_rates, width, label='Expected', color='#3498db', alpha=0.8)
    ax1.set_xlabel('Scenario', fontsize=10)
    ax1.set_ylabel('Success Rate (%)', fontsize=10)
    ax1.set_title('Success Rate by Scenario', fontsize=12, fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels([s.replace('_', '\n') for s in scenarios], fontsize=8, rotation=0)
    ax1.legend(fontsize=9)
    ax1.grid(True, alpha=0.3, axis='y')

    # 2. Success Rate by Difficulty
    ax2 = fig.add_subplot(gs[0, 1])
    difficulty_stats = summary_df.groupby('difficulty')['success_rate'].agg(['mean', 'std']).reset_index()
    difficulties = ['Easy', 'Medium', 'Hard']
    difficulty_stats['difficulty'] = pd.Categorical(difficulty_stats['difficulty'], categories=difficulties, ordered=True)
    difficulty_stats = difficulty_stats.sort_values('difficulty')
    colors = {'Easy': '#2ecc71', 'Medium': '#f39c12', 'Hard': '#e74c3c'}
    bars = ax2.bar(difficulty_stats['difficulty'], difficulty_stats['mean'] * 100,
                   color=[colors[d] for d in difficulty_stats['difficulty']], alpha=0.8)
    ax2.errorbar(difficulty_stats['difficulty'], difficulty_stats['mean'] * 100,
                 yerr=difficulty_stats['std'] * 100, fmt='none', ecolor='black', capsize=5)
    ax2.set_xlabel('Difficulty', fontsize=10)
    ax2.set_ylabel('Success Rate (%)', fontsize=10)
    ax2.set_title('Success Rate by Difficulty', fontsize=12, fontweight='bold')
    ax2.grid(True, alpha=0.3, axis='y')

    # 3. Average Overlap by Scenario
    ax3 = fig.add_subplot(gs[0, 2])
    avg_overlaps = summary_df['avg_max_overlap_cm3'].values
    bars = ax3.bar(scenarios, avg_overlaps, color='#9b59b6', alpha=0.8)
    ax3.set_xlabel('Scenario', fontsize=10)
    ax3.set_ylabel('Avg Max Overlap (cm³)', fontsize=10)
    ax3.set_title('Average Maximum Overlap Volume', fontsize=12, fontweight='bold')
    ax3.set_xticklabels([s.replace('_', '\n') for s in scenarios], fontsize=8, rotation=0)
    ax3.grid(True, alpha=0.3, axis='y')

    # 4. Overlap Volume Over Time (First Episode Each Scenario)
    ax4 = fig.add_subplot(gs[1, :])
    for scenario in detailed_df['scenario'].unique():
        scenario_data = detailed_df[(detailed_df['scenario'] == scenario) & (detailed_df['episode'] == 1)]
        ax4.plot(scenario_data['step'], scenario_data['overlap_volume_cm3'],
                label=scenario.replace('_', ' ').title(), alpha=0.7, linewidth=2)
    ax4.set_xlabel('Step', fontsize=10)
    ax4.set_ylabel('Overlap Volume (cm³)', fontsize=10)
    ax4.set_title('Overlap Volume Over Time (Episode 1 per Scenario)', fontsize=12, fontweight='bold')
    ax4.legend(fontsize=8, ncol=4, loc='upper left')
    ax4.grid(True, alpha=0.3)

    # 5. Distance Over Time (First Episode Each Scenario)
    ax5 = fig.add_subplot(gs[2, :])
    for scenario in detailed_df['scenario'].unique():
        scenario_data = detailed_df[(detailed_df['scenario'] == scenario) & (detailed_df['episode'] == 1)]
        ax5.plot(scenario_data['step'], scenario_data['distance_cm'],
                label=scenario.replace('_', ' ').title(), alpha=0.7, linewidth=2)
    ax5.set_xlabel('Step', fontsize=10)
    ax5.set_ylabel('Distance to Target (cm)', fontsize=10)
    ax5.set_title('Distance to Target Over Time (Episode 1 per Scenario)', fontsize=12, fontweight='bold')
    ax5.legend(fontsize=8, ncol=4, loc='upper right')
    ax5.grid(True, alpha=0.3)

    # 6. Reward Components (Averaged Across All Scenarios)
    ax6 = fig.add_subplot(gs[3, 0])
    components = ['reward_overlap', 'reward_proximity', 'reward_quality', 'reward_contact']
    labels = ['Overlap', 'Proximity', 'Quality', 'Contact']
    avg_components = [detailed_df[comp].mean() for comp in components]
    colors_comp = ['#9b59b6', '#3498db', '#2ecc71', '#e74c3c']
    bars = ax6.bar(labels, avg_components, color=colors_comp, alpha=0.8)
    ax6.set_xlabel('Reward Component', fontsize=10)
    ax6.set_ylabel('Average Value', fontsize=10)
    ax6.set_title('Average Reward Components', fontsize=12, fontweight='bold')
    ax6.grid(True, alpha=0.3, axis='y')

    # 7. Contact Analysis
    ax7 = fig.add_subplot(gs[3, 1])
    avg_contacts = summary_df['avg_total_contacts'].values
    bars = ax7.bar(scenarios, avg_contacts, color='#e74c3c', alpha=0.8)
    ax7.set_xlabel('Scenario', fontsize=10)
    ax7.set_ylabel('Avg Total Contacts', fontsize=10)
    ax7.set_title('Average Total Contact Steps', fontsize=12, fontweight='bold')
    ax7.set_xticklabels([s.replace('_', '\n') for s in scenarios], fontsize=8, rotation=0)
    ax7.grid(True, alpha=0.3, axis='y')

    # 8. Final Distance by Scenario
    ax8 = fig.add_subplot(gs[3, 2])
    avg_distances = summary_df['avg_final_distance_cm'].values
    bars = ax8.bar(scenarios, avg_distances, color='#f39c12', alpha=0.8)
    ax8.set_xlabel('Scenario', fontsize=10)
    ax8.set_ylabel('Avg Final Distance (cm)', fontsize=10)
    ax8.set_title('Average Final Distance to Target', fontsize=12, fontweight='bold')
    ax8.set_xticklabels([s.replace('_', '\n') for s in scenarios], fontsize=8, rotation=0)
    ax8.grid(True, alpha=0.3, axis='y')

    plt.suptitle('V7 Model Test Results - Comprehensive Overview', fontsize=16, fontweight='bold', y=0.995)
    plot_path = output_dir / "test_overview.png"
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Overview plot: {plot_path}")

    # Plot 2: Per-Scenario Detailed Analysis
    for scenario in detailed_df['scenario'].unique():
        generate_scenario_detail_plot(detailed_df, scenario, output_dir)


def generate_scenario_detail_plot(detailed_df, scenario, output_dir):
    """Generate detailed plot for a specific scenario"""
    scenario_data = detailed_df[detailed_df['scenario'] == scenario]

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle(f"Scenario: {scenario.replace('_', ' ').title()}", fontsize=14, fontweight='bold')

    # 1. Overlap Volume Across Episodes
    ax = axes[0, 0]
    for episode in scenario_data['episode'].unique():
        ep_data = scenario_data[scenario_data['episode'] == episode]
        ax.plot(ep_data['step'], ep_data['overlap_volume_cm3'], alpha=0.6, linewidth=1.5)
    ax.set_xlabel('Step')
    ax.set_ylabel('Overlap Volume (cm³)')
    ax.set_title('Overlap Volume Across All Episodes')
    ax.grid(True, alpha=0.3)

    # 2. Distance Across Episodes
    ax = axes[0, 1]
    for episode in scenario_data['episode'].unique():
        ep_data = scenario_data[scenario_data['episode'] == episode]
        ax.plot(ep_data['step'], ep_data['distance_cm'], alpha=0.6, linewidth=1.5)
    ax.set_xlabel('Step')
    ax.set_ylabel('Distance (cm)')
    ax.set_title('Distance to Target Across All Episodes')
    ax.grid(True, alpha=0.3)

    # 3. Cumulative Reward
    ax = axes[0, 2]
    for episode in scenario_data['episode'].unique():
        ep_data = scenario_data[scenario_data['episode'] == episode]
        ax.plot(ep_data['step'], ep_data['cumulative_reward'], alpha=0.6, linewidth=1.5)
    ax.set_xlabel('Step')
    ax.set_ylabel('Cumulative Reward')
    ax.set_title('Cumulative Reward Across All Episodes')
    ax.grid(True, alpha=0.3)

    # 4. Contact Steps Per Episode
    ax = axes[1, 0]
    episode_contacts = []
    episodes = scenario_data['episode'].unique()
    for episode in episodes:
        ep_data = scenario_data[scenario_data['episode'] == episode]
        total_contact_steps = (ep_data['num_contacts'] > 0).sum()
        episode_contacts.append(total_contact_steps)
    ax.bar(episodes, episode_contacts, alpha=0.7, color='#e74c3c')
    ax.set_xlabel('Episode')
    ax.set_ylabel('Contact Steps')
    ax.set_title('Total Contact Steps Per Episode')
    ax.grid(True, alpha=0.3, axis='y')

    # 5. Reward Components Over Time (Episode 1)
    ax = axes[1, 1]
    ep1_data = scenario_data[scenario_data['episode'] == 1]
    ax.plot(ep1_data['step'], ep1_data['reward_overlap'], label='Overlap', linewidth=2)
    ax.plot(ep1_data['step'], ep1_data['reward_proximity'], label='Proximity', linewidth=2)
    ax.plot(ep1_data['step'], ep1_data['reward_quality'], label='Quality', linewidth=2)
    ax.plot(ep1_data['step'], ep1_data['reward_contact'], label='Contact', linewidth=2)
    ax.set_xlabel('Step')
    ax.set_ylabel('Reward Value')
    ax.set_title('Reward Components (Episode 1)')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 6. Success Consistency
    ax = axes[1, 2]
    for episode in scenario_data['episode'].unique():
        ep_data = scenario_data[scenario_data['episode'] == episode]
        ax.plot(ep_data['step'], ep_data['consecutive_success'], alpha=0.6, linewidth=1.5)
    ax.set_xlabel('Step')
    ax.set_ylabel('Consecutive Success Steps')
    ax.set_title('Success Consistency Across Episodes')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plot_path = output_dir / f"scenario_{scenario}_detailed.png"
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Scenario plot: {plot_path}")


def main():
    parser = argparse.ArgumentParser(description='V7 SC-1 Comprehensive Testing')
    parser.add_argument('--model', type=str, default=None,
                       help='Path to model file (auto-finds latest if not specified)')
    parser.add_argument('--episodes', type=int, default=5,
                       help='Number of episodes per scenario (default: 5)')
    parser.add_argument('--vis', action='store_true',
                       help='Enable visualization (slower)')

    args = parser.parse_args()

    # Find model
    if args.model:
        model_path = args.model
        run_dir = Path(model_path).parent
    else:
        print("Auto-detecting latest V7 model...")
        model_path, run_dir = find_latest_v7_model()
        print(f"Found: {model_path}")
        print(f"From run: {run_dir.name}")
        print()

    # Test model
    test_dir, summary_df = test_model(model_path, args.episodes, args.vis)

    print(f"\nTo view results:")
    print(f"  Test overview: {test_dir}/test_overview.png")
    print(f"  Detailed CSVs: {test_dir}/test_results_*.csv")
    print()


if __name__ == "__main__":
    main()
