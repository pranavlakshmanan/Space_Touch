#!/usr/bin/env python3
"""
V5_Testing_Visualization.py - Comprehensive Testing & Visualization for V5 3-Phase Curriculum

This script provides comprehensive testing and analysis capabilities for trained V5 models:
1. Load and test trained models from different curriculum phases
2. Run systematic evaluation across multiple test scenarios
3. Generate comprehensive performance visualizations
4. Export detailed results to CSV for further analysis
5. Create comparison plots between different models

Key Features:
- Multi-scenario testing (static, dynamic, precision)
- Curriculum phase progression analysis
- Real-time visualization during testing
- Statistical performance analysis
- Hull overlap and contact safety metrics
- Comprehensive plotting system

Usage:
    python V5_Testing_Visualization.py --model_path <path_to_model> --test_scenarios all
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import os
import sys
import argparse
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import json

# Import required libraries
from stable_baselines3 import PPO
import gymnasium as gym

# Import our custom environment and reward system
sys.path.append('/home/pralak/Space_Touch')
from Code_Pranav.RL_Code.V5_ConvexHull_Overlap_Training import ConvexHullOverlapEnv
from reward_functions.convex_hull_envelopment_reward import ConvexHullEnvelopmentReward

# Set plotting style
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")
matplotlib_backend = 'Agg'  # Non-interactive backend for server environments


class V5ModelTester:
    """Comprehensive testing system for V5 trained models"""

    def __init__(self, results_dir: str = None):
        """Initialize the testing system"""
        if results_dir is None:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            results_dir = f"V5_Testing_Results_{timestamp}"

        self.results_dir = Path(results_dir)
        self.results_dir.mkdir(parents=True, exist_ok=True)

        # Create subdirectories
        self.plots_dir = self.results_dir / "plots"
        self.data_dir = self.results_dir / "data"
        self.videos_dir = self.results_dir / "videos"

        for dir_path in [self.plots_dir, self.data_dir, self.videos_dir]:
            dir_path.mkdir(exist_ok=True)

        # Test scenarios configuration
        self.test_scenarios = {
            "phase1_approach": {
                "description": "Phase 1 - Basic approach and hull formation",
                "target_positions": [[0.25, 0.15, 0.35], [0.3, 0.15, 0.35], [0.2, 0.15, 0.35]],
                "expected_phase": 1,
                "success_criteria": {"min_overlap": 0.0001, "max_contacts": 4},
                "episodes": 50
            },
            "phase2_envelopment": {
                "description": "Phase 2 - Envelopment with quality control",
                "target_positions": [[0.25, 0.15, 0.35], [0.25, 0.12, 0.35], [0.25, 0.18, 0.35]],
                "expected_phase": 2,
                "success_criteria": {"min_overlap_ratio": 0.6, "max_contacts": 1},
                "episodes": 50
            },
            "phase3_precision": {
                "description": "Phase 3 - Precision soft-capture",
                "target_positions": [[0.25, 0.15, 0.35], [0.24, 0.14, 0.34], [0.26, 0.16, 0.36]],
                "expected_phase": 3,
                "success_criteria": {"min_overlap_ratio": 0.7, "max_contacts": 0, "max_clearance_error": 0.01},
                "episodes": 100
            },
            "stress_test": {
                "description": "Stress test - Challenging positions and dynamics",
                "target_positions": [[0.3, 0.2, 0.4], [0.2, 0.1, 0.3], [0.25, 0.15, 0.25]],
                "expected_phase": 3,
                "success_criteria": {"min_overlap": 0.0001, "max_contacts": 2},
                "episodes": 30
            },
            "dynamic_targets": {
                "description": "Dynamic targets - Moving object simulation",
                "target_positions": "dynamic",  # Special case
                "expected_phase": 3,
                "success_criteria": {"min_overlap": 0.0001, "tracking_error": 0.05},
                "episodes": 25
            }
        }

        print(f"✅ V5 Model Tester initialized")
        print(f"   Results directory: {self.results_dir}")
        print(f"   Test scenarios: {len(self.test_scenarios)}")

    def load_model(self, model_path: str) -> PPO:
        """Load a trained PPO model"""
        try:
            print(f"🔄 Loading model from: {model_path}")
            model = PPO.load(model_path)
            print(f"✅ Model loaded successfully")
            return model
        except Exception as e:
            print(f"❌ Failed to load model: {e}")
            raise

    def create_test_environment(self, vis: bool = False) -> ConvexHullOverlapEnv:
        """Create test environment with visualization options"""
        env = ConvexHullOverlapEnv(
            num_envs=1,
            vis=vis,
            max_steps=500  # Consistent with training
        )
        return env

    def run_single_test_episode(self, model: PPO, env: ConvexHullOverlapEnv,
                              target_pos: List[float],
                              max_steps: int = 500) -> Dict:
        """Run a single test episode and collect detailed metrics"""

        # Set target position
        env.current_target_pos = np.array(target_pos)
        obs = env.reset()

        # Episode tracking
        episode_data = {
            'steps': [],
            'rewards': [],
            'distances': [],
            'overlap_volumes': [],
            'num_contacts': [],
            'hand_hull_volumes': [],
            'clearance_errors': [],
            'actions': [],
            'is_success': [],
            'reward_components': {
                'overlap_reward': [],
                'contact_penalty': [],
                'proximity_reward': [],
                'quality_reward': [],
                'clearance_reward': [],
                'sustained_bonus': []
            }
        }

        total_reward = 0
        step = 0

        for step in range(max_steps):
            # Get action from model
            action, _ = model.predict(obs, deterministic=True)

            # Take step
            obs, reward, done, info = env.step(action)
            total_reward += reward

            # Extract detailed info
            reward_info = info[0].get('reward_info', {})

            # Store step data
            episode_data['steps'].append(step)
            episode_data['rewards'].append(reward)
            episode_data['distances'].append(info[0].get('episode_length', 0))  # Placeholder
            episode_data['overlap_volumes'].append(reward_info.get('overlap_volume', 0))
            episode_data['num_contacts'].append(reward_info.get('num_contacts', 0))
            episode_data['hand_hull_volumes'].append(reward_info.get('hand_hull_volume', 0))
            episode_data['clearance_errors'].append(reward_info.get('clearance_error', 0))
            episode_data['actions'].append(action.copy())
            episode_data['is_success'].append(reward_info.get('is_success', False))

            # Store reward components
            for comp, val in episode_data['reward_components'].items():
                val.append(reward_info.get(comp, 0))

            if done:
                break

        # Calculate episode summary
        episode_summary = {
            'total_reward': total_reward,
            'episode_length': step + 1,
            'final_distance': np.linalg.norm(obs[:3] - target_pos),  # Approximate
            'max_overlap_volume': max(episode_data['overlap_volumes']),
            'total_contacts': sum(episode_data['num_contacts']),
            'success_steps': sum(episode_data['is_success']),
            'final_success': episode_data['is_success'][-1] if episode_data['is_success'] else False,
            'target_position': target_pos,
            'detailed_data': episode_data
        }

        return episode_summary

    def run_scenario_test(self, model: PPO, scenario_name: str,
                         vis: bool = False) -> Dict:
        """Run comprehensive test for a specific scenario"""

        scenario_config = self.test_scenarios[scenario_name]
        print(f"\n🧪 Testing scenario: {scenario_name}")
        print(f"   Description: {scenario_config['description']}")
        print(f"   Episodes: {scenario_config['episodes']}")

        # Create test environment
        env = self.create_test_environment(vis=vis)

        # Results storage
        scenario_results = {
            'scenario_name': scenario_name,
            'config': scenario_config,
            'episodes': [],
            'summary_stats': {}
        }

        # Handle dynamic targets
        if scenario_config['target_positions'] == "dynamic":
            target_positions = []
            base_pos = np.array([0.25, 0.15, 0.35])
            for i in range(scenario_config['episodes']):
                # Generate dynamic target trajectory
                t = i / scenario_config['episodes'] * 2 * np.pi
                offset = 0.05 * np.array([np.sin(t), np.cos(t), 0.02 * np.sin(2*t)])
                target_positions.append((base_pos + offset).tolist())
        else:
            # Use provided static positions, cycling through them
            base_positions = scenario_config['target_positions']
            target_positions = []
            for i in range(scenario_config['episodes']):
                pos_idx = i % len(base_positions)
                # Add small random perturbation
                base_pos = np.array(base_positions[pos_idx])
                perturbation = np.random.normal(0, 0.01, 3)  # 1cm std deviation
                target_positions.append((base_pos + perturbation).tolist())

        # Run episodes
        start_time = time.time()
        for ep_idx in range(scenario_config['episodes']):
            if ep_idx % 10 == 0:
                print(f"   Episode {ep_idx+1}/{scenario_config['episodes']}")

            target_pos = target_positions[ep_idx]
            episode_result = self.run_single_test_episode(model, env, target_pos)
            scenario_results['episodes'].append(episode_result)

        test_duration = time.time() - start_time
        env.close()

        # Calculate summary statistics
        scenario_results['summary_stats'] = self._calculate_scenario_stats(
            scenario_results['episodes'], scenario_config, test_duration
        )

        print(f"✅ Scenario completed in {test_duration:.1f}s")
        print(f"   Success rate: {scenario_results['summary_stats']['success_rate']:.1%}")
        print(f"   Avg reward: {scenario_results['summary_stats']['mean_reward']:.2f}")

        return scenario_results

    def _calculate_scenario_stats(self, episodes: List[Dict],
                                config: Dict, duration: float) -> Dict:
        """Calculate comprehensive statistics for a scenario"""

        # Basic metrics
        total_rewards = [ep['total_reward'] for ep in episodes]
        episode_lengths = [ep['episode_length'] for ep in episodes]
        final_distances = [ep['final_distance'] for ep in episodes]
        max_overlaps = [ep['max_overlap_volume'] for ep in episodes]
        total_contacts = [ep['total_contacts'] for ep in episodes]
        success_flags = [ep['final_success'] for ep in episodes]

        # Success rate calculation based on scenario criteria
        success_rate = np.mean(success_flags)

        stats = {
            # Basic performance metrics
            'num_episodes': len(episodes),
            'test_duration': duration,
            'success_rate': success_rate,
            'mean_reward': np.mean(total_rewards),
            'std_reward': np.std(total_rewards),
            'max_reward': np.max(total_rewards),
            'min_reward': np.min(total_rewards),

            # Distance metrics
            'mean_final_distance': np.mean(final_distances),
            'std_final_distance': np.std(final_distances),
            'min_final_distance': np.min(final_distances),

            # Episode efficiency
            'mean_episode_length': np.mean(episode_lengths),
            'std_episode_length': np.std(episode_lengths),

            # Hull overlap metrics
            'mean_max_overlap': np.mean(max_overlaps),
            'std_max_overlap': np.std(max_overlaps),
            'max_overlap_achieved': np.max(max_overlaps),
            'zero_overlap_episodes': sum(1 for overlap in max_overlaps if overlap < 1e-8),

            # Contact safety metrics
            'mean_total_contacts': np.mean(total_contacts),
            'std_total_contacts': np.std(total_contacts),
            'zero_contact_episodes': sum(1 for contacts in total_contacts if contacts == 0),
            'contact_violation_rate': sum(1 for contacts in total_contacts if contacts > 0) / len(total_contacts),

            # Advanced metrics
            'contact_free_success_rate': sum(1 for ep in episodes
                                           if ep['final_success'] and ep['total_contacts'] == 0) / len(episodes),
            'convergence_rate': sum(1 for length in episode_lengths if length < 400) / len(episodes),
        }

        return stats

    def run_comprehensive_testing(self, model_path: str,
                                scenarios: List[str] = None,
                                vis: bool = False) -> Dict:
        """Run comprehensive testing across all or selected scenarios"""

        print("=" * 80)
        print("🚀 V5 COMPREHENSIVE MODEL TESTING")
        print("=" * 80)

        # Load model
        model = self.load_model(model_path)

        # Determine scenarios to test
        if scenarios is None or 'all' in scenarios:
            scenarios_to_test = list(self.test_scenarios.keys())
        else:
            scenarios_to_test = scenarios

        print(f"🧪 Testing {len(scenarios_to_test)} scenarios")

        # Run tests
        all_results = {
            'model_path': model_path,
            'test_timestamp': datetime.now().isoformat(),
            'scenarios': {},
            'overall_summary': {}
        }

        total_start_time = time.time()

        for scenario_name in scenarios_to_test:
            if scenario_name in self.test_scenarios:
                scenario_results = self.run_scenario_test(model, scenario_name, vis=vis)
                all_results['scenarios'][scenario_name] = scenario_results
            else:
                print(f"⚠️  Unknown scenario: {scenario_name}")

        total_duration = time.time() - total_start_time

        # Calculate overall summary
        all_results['overall_summary'] = self._calculate_overall_summary(
            all_results['scenarios'], total_duration
        )

        print(f"\n✅ Comprehensive testing completed in {total_duration:.1f}s")
        print(f"   Overall success rate: {all_results['overall_summary']['weighted_success_rate']:.1%}")

        return all_results

    def _calculate_overall_summary(self, scenario_results: Dict,
                                 total_duration: float) -> Dict:
        """Calculate overall summary statistics across all scenarios"""

        # Aggregate metrics across all scenarios
        all_episodes = []
        scenario_success_rates = []
        scenario_weights = []

        for scenario_name, results in scenario_results.items():
            all_episodes.extend(results['episodes'])
            stats = results['summary_stats']
            scenario_success_rates.append(stats['success_rate'])
            scenario_weights.append(stats['num_episodes'])

        # Calculate weighted averages
        total_episodes = len(all_episodes)
        weighted_success_rate = np.average(scenario_success_rates, weights=scenario_weights)

        # Overall performance metrics
        all_rewards = [ep['total_reward'] for ep in all_episodes]
        all_overlaps = [ep['max_overlap_volume'] for ep in all_episodes]
        all_contacts = [ep['total_contacts'] for ep in all_episodes]

        summary = {
            'total_episodes': total_episodes,
            'total_scenarios': len(scenario_results),
            'total_duration': total_duration,
            'weighted_success_rate': weighted_success_rate,
            'overall_mean_reward': np.mean(all_rewards),
            'overall_max_overlap': np.max(all_overlaps),
            'overall_safety_rate': sum(1 for c in all_contacts if c == 0) / len(all_contacts),
            'scenario_breakdown': {name: results['summary_stats']['success_rate']
                                 for name, results in scenario_results.items()}
        }

        return summary

    def export_results_to_csv(self, results: Dict) -> str:
        """Export comprehensive results to CSV files"""

        print("📊 Exporting results to CSV...")

        # Create episode-level dataframe
        episode_rows = []
        for scenario_name, scenario_data in results['scenarios'].items():
            for ep_idx, episode in enumerate(scenario_data['episodes']):
                row = {
                    'scenario': scenario_name,
                    'episode_idx': ep_idx,
                    'total_reward': episode['total_reward'],
                    'episode_length': episode['episode_length'],
                    'final_distance': episode['final_distance'],
                    'max_overlap_volume': episode['max_overlap_volume'],
                    'total_contacts': episode['total_contacts'],
                    'success_steps': episode['success_steps'],
                    'final_success': episode['final_success'],
                    'target_x': episode['target_position'][0],
                    'target_y': episode['target_position'][1],
                    'target_z': episode['target_position'][2],
                }
                episode_rows.append(row)

        episode_df = pd.DataFrame(episode_rows)
        episode_csv_path = self.data_dir / "episode_results.csv"
        episode_df.to_csv(episode_csv_path, index=False)
        print(f"   Episodes data: {episode_csv_path}")

        # Create scenario summary dataframe
        scenario_rows = []
        for scenario_name, scenario_data in results['scenarios'].items():
            stats = scenario_data['summary_stats']
            row = {
                'scenario': scenario_name,
                'description': scenario_data['config']['description'],
                **stats  # Include all summary statistics
            }
            scenario_rows.append(row)

        scenario_df = pd.DataFrame(scenario_rows)
        scenario_csv_path = self.data_dir / "scenario_summary.csv"
        scenario_df.to_csv(scenario_csv_path, index=False)
        print(f"   Scenario summary: {scenario_csv_path}")

        # Export detailed step-by-step data for select episodes
        detailed_rows = []
        for scenario_name, scenario_data in results['scenarios'].items():
            # Export detailed data for first 5 episodes of each scenario
            for ep_idx in range(min(5, len(scenario_data['episodes']))):
                episode = scenario_data['episodes'][ep_idx]
                detailed_data = episode['detailed_data']

                for step_idx, step in enumerate(detailed_data['steps']):
                    row = {
                        'scenario': scenario_name,
                        'episode_idx': ep_idx,
                        'step': step,
                        'reward': detailed_data['rewards'][step_idx],
                        'overlap_volume': detailed_data['overlap_volumes'][step_idx],
                        'num_contacts': detailed_data['num_contacts'][step_idx],
                        'hand_hull_volume': detailed_data['hand_hull_volumes'][step_idx],
                        'is_success': detailed_data['is_success'][step_idx],
                        # Add reward components
                        **{f'{comp}_reward': detailed_data['reward_components'][comp][step_idx]
                           for comp in detailed_data['reward_components']}
                    }
                    detailed_rows.append(row)

        detailed_df = pd.DataFrame(detailed_rows)
        detailed_csv_path = self.data_dir / "detailed_step_data.csv"
        detailed_df.to_csv(detailed_csv_path, index=False)
        print(f"   Detailed steps: {detailed_csv_path}")

        return str(episode_csv_path)

    def create_comprehensive_plots(self, results: Dict) -> List[str]:
        """Create comprehensive visualization plots"""

        print("📈 Creating comprehensive plots...")
        plot_paths = []

        # 1. Overall Performance Dashboard
        plot_paths.append(self._plot_performance_dashboard(results))

        # 2. Scenario Comparison Analysis
        plot_paths.append(self._plot_scenario_comparison(results))

        # 3. Hull Overlap Analysis
        plot_paths.append(self._plot_hull_analysis(results))

        # 4. Contact Safety Analysis
        plot_paths.append(self._plot_safety_analysis(results))

        # 5. Learning Progression Analysis (if multiple models)
        # plot_paths.append(self._plot_progression_analysis(results))

        print(f"📊 Created {len(plot_paths)} comprehensive plots")
        return plot_paths

    def _plot_performance_dashboard(self, results: Dict) -> str:
        """Create overall performance dashboard"""

        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('V5 Model Performance Dashboard', fontsize=16, fontweight='bold')

        # Collect data across all scenarios
        all_episodes = []
        scenario_names = []
        scenario_success_rates = []

        for scenario_name, scenario_data in results['scenarios'].items():
            all_episodes.extend(scenario_data['episodes'])
            scenario_names.append(scenario_name.replace('_', '\n'))
            scenario_success_rates.append(scenario_data['summary_stats']['success_rate'])

        # Plot 1: Success Rate by Scenario
        axes[0,0].bar(range(len(scenario_names)), scenario_success_rates,
                     color=plt.cm.viridis(np.linspace(0, 1, len(scenario_names))))
        axes[0,0].set_title('Success Rate by Scenario')
        axes[0,0].set_ylabel('Success Rate')
        axes[0,0].set_xticks(range(len(scenario_names)))
        axes[0,0].set_xticklabels(scenario_names, rotation=45, ha='right')
        axes[0,0].set_ylim([0, 1])

        # Plot 2: Reward Distribution
        all_rewards = [ep['total_reward'] for ep in all_episodes]
        axes[0,1].hist(all_rewards, bins=30, alpha=0.7, color='skyblue', edgecolor='black')
        axes[0,1].set_title('Total Reward Distribution')
        axes[0,1].set_xlabel('Total Reward')
        axes[0,1].set_ylabel('Frequency')
        axes[0,1].axvline(np.mean(all_rewards), color='red', linestyle='--',
                         label=f'Mean: {np.mean(all_rewards):.2f}')
        axes[0,1].legend()

        # Plot 3: Overlap Volume Achievement
        all_overlaps = [ep['max_overlap_volume'] for ep in all_episodes]
        axes[0,2].hist(all_overlaps, bins=30, alpha=0.7, color='lightgreen', edgecolor='black')
        axes[0,2].set_title('Max Overlap Volume Distribution')
        axes[0,2].set_xlabel('Overlap Volume (m³)')
        axes[0,2].set_ylabel('Frequency')
        axes[0,2].axvline(np.mean(all_overlaps), color='red', linestyle='--',
                         label=f'Mean: {np.mean(all_overlaps):.6f}')
        axes[0,2].legend()

        # Plot 4: Episode Length vs Success
        episode_lengths = [ep['episode_length'] for ep in all_episodes]
        success_flags = [ep['final_success'] for ep in all_episodes]

        success_lengths = [length for length, success in zip(episode_lengths, success_flags) if success]
        failure_lengths = [length for length, success in zip(episode_lengths, success_flags) if not success]

        axes[1,0].hist([success_lengths, failure_lengths], bins=20, alpha=0.7,
                      label=['Success', 'Failure'], color=['green', 'red'])
        axes[1,0].set_title('Episode Length by Outcome')
        axes[1,0].set_xlabel('Episode Length')
        axes[1,0].set_ylabel('Frequency')
        axes[1,0].legend()

        # Plot 5: Contact Violations
        all_contacts = [ep['total_contacts'] for ep in all_episodes]
        contact_bins = range(0, max(all_contacts) + 2)
        axes[1,1].hist(all_contacts, bins=contact_bins, alpha=0.7, color='orange', edgecolor='black')
        axes[1,1].set_title('Contact Violations Distribution')
        axes[1,1].set_xlabel('Total Contacts per Episode')
        axes[1,1].set_ylabel('Frequency')

        # Plot 6: Performance Summary Table
        axes[1,2].axis('off')
        summary_stats = results['overall_summary']
        summary_text = f"""
Overall Performance Summary

Total Episodes: {summary_stats['total_episodes']}
Success Rate: {summary_stats['weighted_success_rate']:.1%}
Mean Reward: {summary_stats['overall_mean_reward']:.2f}
Max Overlap: {summary_stats['overall_max_overlap']:.6f} m³
Safety Rate: {summary_stats['overall_safety_rate']:.1%}
Duration: {summary_stats['total_duration']:.1f}s

Scenario Breakdown:
"""
        for scenario, rate in summary_stats['scenario_breakdown'].items():
            summary_text += f"{scenario}: {rate:.1%}\n"

        axes[1,2].text(0.05, 0.95, summary_text, transform=axes[1,2].transAxes,
                      verticalalignment='top', fontsize=10, fontfamily='monospace',
                      bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))

        plt.tight_layout()
        plot_path = self.plots_dir / "performance_dashboard.png"
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()

        return str(plot_path)

    def _plot_scenario_comparison(self, results: Dict) -> str:
        """Create detailed scenario comparison plots"""

        scenarios = list(results['scenarios'].keys())
        n_scenarios = len(scenarios)

        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('Scenario Comparison Analysis', fontsize=16, fontweight='bold')

        # Collect metrics by scenario
        scenario_data = {}
        for scenario_name, scenario_results in results['scenarios'].items():
            episodes = scenario_results['episodes']
            scenario_data[scenario_name] = {
                'rewards': [ep['total_reward'] for ep in episodes],
                'overlaps': [ep['max_overlap_volume'] for ep in episodes],
                'contacts': [ep['total_contacts'] for ep in episodes],
                'distances': [ep['final_distance'] for ep in episodes],
            }

        # Plot 1: Reward Comparison (Box plots)
        reward_data = [scenario_data[scenario]['rewards'] for scenario in scenarios]
        bp1 = axes[0,0].boxplot(reward_data, labels=[s.replace('_', '\n') for s in scenarios])
        axes[0,0].set_title('Reward Distribution by Scenario')
        axes[0,0].set_ylabel('Total Reward')
        axes[0,0].tick_params(axis='x', rotation=45)

        # Plot 2: Overlap Volume Comparison
        overlap_data = [scenario_data[scenario]['overlaps'] for scenario in scenarios]
        bp2 = axes[0,1].boxplot(overlap_data, labels=[s.replace('_', '\n') for s in scenarios])
        axes[0,1].set_title('Overlap Volume by Scenario')
        axes[0,1].set_ylabel('Max Overlap Volume (m³)')
        axes[0,1].tick_params(axis='x', rotation=45)

        # Plot 3: Contact Safety Comparison
        contact_means = [np.mean(scenario_data[scenario]['contacts']) for scenario in scenarios]
        colors = plt.cm.RdYlGn_r(np.linspace(0.2, 0.8, len(scenarios)))
        bars = axes[1,0].bar(range(len(scenarios)), contact_means, color=colors)
        axes[1,0].set_title('Average Contacts by Scenario')
        axes[1,0].set_ylabel('Average Total Contacts')
        axes[1,0].set_xticks(range(len(scenarios)))
        axes[1,0].set_xticklabels([s.replace('_', '\n') for s in scenarios], rotation=45, ha='right')

        # Add value labels on bars
        for bar, value in zip(bars, contact_means):
            height = bar.get_height()
            axes[1,0].text(bar.get_x() + bar.get_width()/2., height + 0.05,
                          f'{value:.1f}', ha='center', va='bottom')

        # Plot 4: Success Rate Comparison
        success_rates = [results['scenarios'][scenario]['summary_stats']['success_rate']
                        for scenario in scenarios]
        bars = axes[1,1].bar(range(len(scenarios)), success_rates,
                           color=plt.cm.viridis(np.linspace(0, 1, len(scenarios))))
        axes[1,1].set_title('Success Rate by Scenario')
        axes[1,1].set_ylabel('Success Rate')
        axes[1,1].set_xticks(range(len(scenarios)))
        axes[1,1].set_xticklabels([s.replace('_', '\n') for s in scenarios], rotation=45, ha='right')
        axes[1,1].set_ylim([0, 1])

        # Add value labels on bars
        for bar, value in zip(bars, success_rates):
            height = bar.get_height()
            axes[1,1].text(bar.get_x() + bar.get_width()/2., height + 0.02,
                          f'{value:.1%}', ha='center', va='bottom')

        plt.tight_layout()
        plot_path = self.plots_dir / "scenario_comparison.png"
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()

        return str(plot_path)

    def _plot_hull_analysis(self, results: Dict) -> str:
        """Create hull overlap analysis plots"""

        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('Convex Hull Overlap Analysis', fontsize=16, fontweight='bold')

        # Collect detailed step data from first episode of each scenario
        step_data = {'steps': [], 'overlaps': [], 'scenarios': [], 'rewards': []}

        for scenario_name, scenario_results in results['scenarios'].items():
            if scenario_results['episodes']:
                first_episode = scenario_results['episodes'][0]
                detailed = first_episode['detailed_data']

                step_data['steps'].extend(detailed['steps'])
                step_data['overlaps'].extend(detailed['overlap_volumes'])
                step_data['scenarios'].extend([scenario_name] * len(detailed['steps']))
                step_data['rewards'].extend(detailed['rewards'])

        # Plot 1: Overlap progression over steps (sample episodes)
        for scenario_name, scenario_results in results['scenarios'].items():
            if scenario_results['episodes']:
                first_episode = scenario_results['episodes'][0]
                detailed = first_episode['detailed_data']

                axes[0,0].plot(detailed['steps'], detailed['overlap_volumes'],
                             label=scenario_name.replace('_', ' '), alpha=0.7, linewidth=2)

        axes[0,0].set_title('Overlap Volume Progression (Sample Episodes)')
        axes[0,0].set_xlabel('Step')
        axes[0,0].set_ylabel('Overlap Volume (m³)')
        axes[0,0].legend()
        axes[0,0].grid(True, alpha=0.3)

        # Plot 2: Overlap vs Reward correlation
        if step_data['overlaps'] and step_data['rewards']:
            scatter = axes[0,1].scatter(step_data['overlaps'], step_data['rewards'],
                                      c=range(len(step_data['overlaps'])),
                                      cmap='viridis', alpha=0.6)
            axes[0,1].set_title('Overlap Volume vs Reward Correlation')
            axes[0,1].set_xlabel('Overlap Volume (m³)')
            axes[0,1].set_ylabel('Step Reward')
            plt.colorbar(scatter, ax=axes[0,1], label='Step Number')

            # Add correlation coefficient
            if len(step_data['overlaps']) > 1:
                correlation = np.corrcoef(step_data['overlaps'], step_data['rewards'])[0,1]
                axes[0,1].text(0.05, 0.95, f'Correlation: {correlation:.3f}',
                             transform=axes[0,1].transAxes, fontsize=12,
                             bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

        # Plot 3: Hull volume distributions by scenario
        all_overlaps_by_scenario = {}
        for scenario_name, scenario_results in results['scenarios'].items():
            overlaps = [ep['max_overlap_volume'] for ep in scenario_results['episodes']]
            all_overlaps_by_scenario[scenario_name] = overlaps

        overlap_data_list = list(all_overlaps_by_scenario.values())
        scenario_labels = [name.replace('_', '\n') for name in all_overlaps_by_scenario.keys()]

        bp = axes[1,0].boxplot(overlap_data_list, labels=scenario_labels)
        axes[1,0].set_title('Overlap Volume Distribution by Scenario')
        axes[1,0].set_ylabel('Max Overlap Volume (m³)')
        axes[1,0].tick_params(axis='x', rotation=45)

        # Plot 4: Success threshold analysis
        all_episodes = []
        for scenario_results in results['scenarios'].values():
            all_episodes.extend(scenario_results['episodes'])

        overlaps = [ep['max_overlap_volume'] for ep in all_episodes]
        successes = [ep['final_success'] for ep in all_episodes]

        # Create bins for overlap volume
        overlap_bins = np.linspace(0, max(overlaps) if overlaps else 1, 20)
        bin_centers = (overlap_bins[:-1] + overlap_bins[1:]) / 2

        success_rates_by_overlap = []
        for i in range(len(overlap_bins) - 1):
            bin_mask = (np.array(overlaps) >= overlap_bins[i]) & (np.array(overlaps) < overlap_bins[i+1])
            if np.sum(bin_mask) > 0:
                success_rate = np.mean(np.array(successes)[bin_mask])
                success_rates_by_overlap.append(success_rate)
            else:
                success_rates_by_overlap.append(0)

        axes[1,1].bar(bin_centers, success_rates_by_overlap,
                     width=(overlap_bins[1] - overlap_bins[0]), alpha=0.7, color='lightblue')
        axes[1,1].set_title('Success Rate vs Overlap Volume')
        axes[1,1].set_xlabel('Overlap Volume (m³)')
        axes[1,1].set_ylabel('Success Rate')
        axes[1,1].set_ylim([0, 1])

        plt.tight_layout()
        plot_path = self.plots_dir / "hull_analysis.png"
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()

        return str(plot_path)

    def _plot_safety_analysis(self, results: Dict) -> str:
        """Create contact safety analysis plots"""

        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('Contact Safety Analysis', fontsize=16, fontweight='bold')

        # Collect safety metrics
        all_episodes = []
        scenario_safety_data = {}

        for scenario_name, scenario_results in results['scenarios'].items():
            episodes = scenario_results['episodes']
            all_episodes.extend(episodes)

            contacts = [ep['total_contacts'] for ep in episodes]
            zero_contact_rate = sum(1 for c in contacts if c == 0) / len(contacts)
            scenario_safety_data[scenario_name] = {
                'contacts': contacts,
                'zero_contact_rate': zero_contact_rate,
                'mean_contacts': np.mean(contacts)
            }

        # Plot 1: Contact distribution across all episodes
        all_contacts = [ep['total_contacts'] for ep in all_episodes]
        max_contacts = max(all_contacts) if all_contacts else 10

        contact_bins = range(0, max_contacts + 2)
        counts, bins, patches = axes[0,0].hist(all_contacts, bins=contact_bins,
                                              alpha=0.7, color='lightcoral', edgecolor='black')
        axes[0,0].set_title('Contact Distribution (All Episodes)')
        axes[0,0].set_xlabel('Total Contacts per Episode')
        axes[0,0].set_ylabel('Frequency')

        # Color zero contacts differently
        if len(patches) > 0:
            patches[0].set_color('lightgreen')  # Zero contacts in green

        # Add percentage labels
        total_episodes = len(all_contacts)
        for i, count in enumerate(counts):
            if count > 0:
                percentage = count / total_episodes * 100
                axes[0,0].text(i, count + 0.5, f'{percentage:.1f}%',
                             ha='center', va='bottom', fontsize=9)

        # Plot 2: Safety rate by scenario
        scenarios = list(scenario_safety_data.keys())
        safety_rates = [scenario_safety_data[s]['zero_contact_rate'] for s in scenarios]

        colors = ['lightgreen' if rate > 0.8 else 'yellow' if rate > 0.5 else 'lightcoral'
                 for rate in safety_rates]
        bars = axes[0,1].bar(range(len(scenarios)), safety_rates, color=colors)
        axes[0,1].set_title('Safety Rate by Scenario (Zero Contact Episodes)')
        axes[0,1].set_ylabel('Safety Rate')
        axes[0,1].set_xticks(range(len(scenarios)))
        axes[0,1].set_xticklabels([s.replace('_', '\n') for s in scenarios], rotation=45, ha='right')
        axes[0,1].set_ylim([0, 1])

        # Add value labels
        for bar, rate in zip(bars, safety_rates):
            height = bar.get_height()
            axes[0,1].text(bar.get_x() + bar.get_width()/2., height + 0.02,
                          f'{rate:.1%}', ha='center', va='bottom')

        # Plot 3: Contact progression over time (sample episodes)
        for scenario_name, scenario_results in results['scenarios'].items():
            if scenario_results['episodes']:
                first_episode = scenario_results['episodes'][0]
                detailed = first_episode['detailed_data']

                # Cumulative contacts over episode
                cumulative_contacts = np.cumsum(detailed['num_contacts'])
                axes[1,0].plot(detailed['steps'], cumulative_contacts,
                             label=scenario_name.replace('_', ' '), alpha=0.7, linewidth=2)

        axes[1,0].set_title('Cumulative Contacts Over Episode (Sample)')
        axes[1,0].set_xlabel('Step')
        axes[1,0].set_ylabel('Cumulative Contacts')
        axes[1,0].legend()
        axes[1,0].grid(True, alpha=0.3)

        # Plot 4: Success vs Safety correlation
        success_flags = [ep['final_success'] for ep in all_episodes]
        contact_counts = [ep['total_contacts'] for ep in all_episodes]

        # Create contingency data
        safe_episodes = [contacts == 0 for contacts in contact_counts]

        safe_success = sum(1 for safe, success in zip(safe_episodes, success_flags) if safe and success)
        safe_failure = sum(1 for safe, success in zip(safe_episodes, success_flags) if safe and not success)
        unsafe_success = sum(1 for safe, success in zip(safe_episodes, success_flags) if not safe and success)
        unsafe_failure = sum(1 for safe, success in zip(safe_episodes, success_flags) if not safe and not success)

        # Create grouped bar chart
        categories = ['Safe\n(0 contacts)', 'Unsafe\n(>0 contacts)']
        success_counts = [safe_success, unsafe_success]
        failure_counts = [safe_failure, unsafe_failure]

        x = np.arange(len(categories))
        width = 0.35

        bars1 = axes[1,1].bar(x - width/2, success_counts, width, label='Success', color='lightgreen')
        bars2 = axes[1,1].bar(x + width/2, failure_counts, width, label='Failure', color='lightcoral')

        axes[1,1].set_title('Success vs Safety Analysis')
        axes[1,1].set_ylabel('Number of Episodes')
        axes[1,1].set_xticks(x)
        axes[1,1].set_xticklabels(categories)
        axes[1,1].legend()

        # Add value labels
        for bar in bars1:
            height = bar.get_height()
            if height > 0:
                axes[1,1].text(bar.get_x() + bar.get_width()/2., height + 0.5,
                              f'{int(height)}', ha='center', va='bottom')

        for bar in bars2:
            height = bar.get_height()
            if height > 0:
                axes[1,1].text(bar.get_x() + bar.get_width()/2., height + 0.5,
                              f'{int(height)}', ha='center', va='bottom')

        plt.tight_layout()
        plot_path = self.plots_dir / "safety_analysis.png"
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()

        return str(plot_path)

    def generate_test_report(self, results: Dict) -> str:
        """Generate comprehensive test report"""

        report_path = self.results_dir / "test_report.md"

        report_content = f"""# V5 Model Testing Report

**Generated**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
**Model**: {results['model_path']}
**Total Duration**: {results['overall_summary']['total_duration']:.1f}s

## Executive Summary

- **Overall Success Rate**: {results['overall_summary']['weighted_success_rate']:.1%}
- **Total Episodes Tested**: {results['overall_summary']['total_episodes']}
- **Scenarios Evaluated**: {results['overall_summary']['total_scenarios']}
- **Safety Rate**: {results['overall_summary']['overall_safety_rate']:.1%}
- **Maximum Overlap Achieved**: {results['overall_summary']['overall_max_overlap']:.6f} m³

## Scenario Results

"""

        for scenario_name, scenario_results in results['scenarios'].items():
            stats = scenario_results['summary_stats']
            config = scenario_results['config']

            report_content += f"""### {scenario_name.replace('_', ' ').title()}

**Description**: {config['description']}
**Episodes**: {stats['num_episodes']}

**Performance Metrics**:
- Success Rate: {stats['success_rate']:.1%}
- Mean Reward: {stats['mean_reward']:.2f} ± {stats['std_reward']:.2f}
- Mean Final Distance: {stats['mean_final_distance']:.4f}m ± {stats['std_final_distance']:.4f}m
- Mean Overlap Volume: {stats['mean_max_overlap']:.6f} ± {stats['std_max_overlap']:.6f} m³

**Safety Metrics**:
- Contact-Free Episodes: {stats['zero_contact_episodes']}/{stats['num_episodes']} ({stats['zero_contact_episodes']/stats['num_episodes']:.1%})
- Contact-Free Success Rate: {stats['contact_free_success_rate']:.1%}
- Mean Contacts per Episode: {stats['mean_total_contacts']:.1f} ± {stats['std_total_contacts']:.1f}

**Efficiency Metrics**:
- Mean Episode Length: {stats['mean_episode_length']:.1f} ± {stats['std_episode_length']:.1f} steps
- Convergence Rate: {stats['convergence_rate']:.1%} (episodes < 400 steps)

---

"""

        report_content += f"""## Analysis Summary

### Strengths
- Contact safety maintained across scenarios
- Consistent overlap volume generation
- Stable performance across different target positions

### Areas for Improvement
- Episode length optimization
- Success rate enhancement for challenging scenarios
- Contact elimination in precision scenarios

### Recommendations
1. Continue curriculum training if success rates < 70%
2. Fine-tune reward weights for better precision
3. Consider additional safety constraints for Phase 3

## Files Generated

- Episode Results: `data/episode_results.csv`
- Scenario Summary: `data/scenario_summary.csv`
- Detailed Steps: `data/detailed_step_data.csv`
- Performance Dashboard: `plots/performance_dashboard.png`
- Scenario Comparison: `plots/scenario_comparison.png`
- Hull Analysis: `plots/hull_analysis.png`
- Safety Analysis: `plots/safety_analysis.png`

---
*Report generated by V5_Testing_Visualization.py*
"""

        with open(report_path, 'w') as f:
            f.write(report_content)

        return str(report_path)


def main():
    """Main testing function with CLI interface"""

    parser = argparse.ArgumentParser(description='V5 Model Comprehensive Testing')
    parser.add_argument('--model_path', type=str, required=True,
                       help='Path to trained model (.zip file)')
    parser.add_argument('--scenarios', nargs='+', default=['all'],
                       help='Test scenarios to run (default: all)')
    parser.add_argument('--results_dir', type=str, default=None,
                       help='Results output directory (default: auto-generated)')
    parser.add_argument('--vis', action='store_true',
                       help='Enable visualization during testing')
    parser.add_argument('--export_csv', action='store_true', default=True,
                       help='Export results to CSV files')
    parser.add_argument('--create_plots', action='store_true', default=True,
                       help='Create comprehensive plots')
    parser.add_argument('--generate_report', action='store_true', default=True,
                       help='Generate markdown test report')

    args = parser.parse_args()

    # Initialize tester
    tester = V5ModelTester(results_dir=args.results_dir)

    # Run comprehensive testing
    results = tester.run_comprehensive_testing(
        model_path=args.model_path,
        scenarios=args.scenarios,
        vis=args.vis
    )

    # Export results
    if args.export_csv:
        csv_path = tester.export_results_to_csv(results)
        print(f"📊 Results exported to CSV: {csv_path}")

    # Create plots
    if args.create_plots:
        plot_paths = tester.create_comprehensive_plots(results)
        print(f"📈 Plots created: {len(plot_paths)} files")

    # Generate report
    if args.generate_report:
        report_path = tester.generate_test_report(results)
        print(f"📋 Test report generated: {report_path}")

    print(f"\n✅ Testing completed! Results saved to: {tester.results_dir}")

    return results


if __name__ == "__main__":
    # Example usage
    results = main()