#!/usr/bin/env python3
"""
Comprehensive V7.6 Model Testing Script with Plots

Tests the trained model across multiple scenarios and generates 
visualization plots for analysis.

Usage:
    python test_trained_model.py <model_path>
"""

import sys
import os
import numpy as np
import time
import matplotlib
# Set backend to 'Agg' to write to file without a display (crucial for EC2)
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from datetime import datetime
from pathlib import Path

sys.path.append("/home/ubuntu/workspace/Space_Touch/Code_Pranav/RL Code")
sys.path.append("/home/ubuntu/workspace/Space_Touch")

from stable_baselines3 import PPO
from v7_6_sc1 import V76Environment

# --- Plotting Configuration ---
PLOT_DIR = "test_results_plots"
os.makedirs(PLOT_DIR, exist_ok=True)

def print_header(title):
    print("\n" + "="*70)
    print(f"  {title}")
    print("="*70)

def print_subheader(title):
    print(f"\n--- {title} ---")

def save_plot(fig, filename):
    """Save plot to the defined directory"""
    path = os.path.join(PLOT_DIR, filename)
    fig.savefig(path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"  >> Plot saved: {path}")

def run_episode(env, model, max_steps=500, deterministic=True):
    """Run a single episode and collect metrics"""
    obs = env.reset()
    
    episode_data = {
        'rewards': [],
        'palm_distances': [],
        'overlaps': [],
        'contacts': [],
        'consecutive_success': [],
    }
    
    total_reward = 0
    
    for step in range(max_steps):
        action, _ = model.predict(obs, deterministic=deterministic)
        obs, reward, done, info = env.step(action)
        
        ri = info[0].get('reward_info', {})
        
        total_reward += reward[0]
        episode_data['rewards'].append(reward[0])
        episode_data['palm_distances'].append(ri.get('palm_to_target_distance', 0) * 100)  # cm
        episode_data['overlaps'].append(ri.get('overlap_volume', 0) * 1e6)  # cm³
        episode_data['contacts'].append(ri.get('num_contacts', 0))
        episode_data['consecutive_success'].append(ri.get('consecutive_success_steps', 0))
        
        if done[0]:
            break
    
    # Pad simple data for easier plotting if needed, or just return lists
    return {
        'total_reward': total_reward,
        'steps': len(episode_data['rewards']),
        'final_distance_cm': episode_data['palm_distances'][-1],
        'min_distance_cm': min(episode_data['palm_distances']),
        'final_overlap_cm3': episode_data['overlaps'][-1],
        'max_overlap_cm3': max(episode_data['overlaps']),
        'total_contacts': sum(episode_data['contacts']),
        'max_consecutive_success': max(episode_data['consecutive_success']),
        'mean_distance_cm': np.mean(episode_data['palm_distances']),
        'mean_overlap_cm3': np.mean(episode_data['overlaps']),
        'trajectory': episode_data,
    }

def test_standard_positions(env, model, num_trials=3):
    print_header("TEST 1: Standard Target Positions")
    scenarios = [
        {'name': 'Close (5cm)', 'offset': np.array([0.03, 0.02, 0.02])},
        {'name': 'Medium (8cm)', 'offset': np.array([0.05, 0.03, 0.04])},
        {'name': 'Far (12cm)', 'offset': np.array([0.08, 0.05, 0.06])},
        {'name': 'Very Far (15cm)', 'offset': np.array([0.10, 0.07, 0.08])},
    ]
    
    results = []
    for scenario in scenarios:
        print_subheader(scenario['name'])
        scenario_results = []
        for trial in range(num_trials):
            env.reset()
            env.reward_calc.phase_offsets[env.reward_calc.current_phase] = scenario['offset']
            env.reset()
            result = run_episode(env, model)
            scenario_results.append(result)
            print(f"  Trial {trial+1}: Reward={result['total_reward']:+.1f}, FinalDist={result['final_distance_cm']:.1f}cm")
        
        avg_reward = np.mean([r['total_reward'] for r in scenario_results])
        avg_dist = np.mean([r['final_distance_cm'] for r in scenario_results])
        results.append({
            'scenario': scenario['name'], 
            'avg_reward': avg_reward, 
            'avg_final_distance_cm': avg_dist
        })
    return results

def test_approach_angles(env, model, num_trials=3):
    print_header("TEST 2: Different Approach Angles")
    base_dist = 0.10
    angles = [
        {'name': 'Front (+X)', 'offset': np.array([base_dist, 0, 0])},
        {'name': 'Back (-X)', 'offset': np.array([-base_dist, 0, 0])},
        {'name': 'Left (+Y)', 'offset': np.array([0, base_dist, 0])},
        {'name': 'Right (-Y)', 'offset': np.array([0, -base_dist, 0])},
        {'name': 'Above (+Z)', 'offset': np.array([0, 0, base_dist])},
        {'name': 'Below (-Z)', 'offset': np.array([0, 0, -base_dist])},
    ]
    
    results = []
    for angle in angles:
        print_subheader(angle['name'])
        angle_results = []
        for trial in range(num_trials):
            env.reset()
            env.reward_calc.phase_offsets[env.reward_calc.current_phase] = angle['offset']
            env.reset()
            result = run_episode(env, model)
            angle_results.append(result)
            print(f"  Trial {trial+1}: Reward={result['total_reward']:+.1f}, FinalDist={result['final_distance_cm']:.1f}cm")
        
        avg_dist = np.mean([r['final_distance_cm'] for r in angle_results])
        results.append({'angle': angle['name'], 'avg_final_distance_cm': avg_dist})
    return results

def test_edge_cases(env, model, num_trials=2):
    print_header("TEST 3: Edge Cases")
    edge_cases = [
        {'name': 'Close Start', 'offset': np.array([0.01, 0.01, 0.01])},
        {'name': 'Far Start', 'offset': np.array([0.12, 0.08, 0.10])},
        {'name': 'Workspace Edge', 'offset': np.array([0.15, 0.10, 0.05])},
    ]
    results = []
    for case in edge_cases:
        print_subheader(case['name'])
        case_results = []
        for trial in range(num_trials):
            env.reset()
            env.reward_calc.phase_offsets[env.reward_calc.current_phase] = case['offset']
            env.reset()
            result = run_episode(env, model)
            case_results.append(result)
            print(f"  Trial {trial+1}: Reward={result['total_reward']:+.1f}, FinalDist={result['final_distance_cm']:.1f}cm")
        results.append({'case': case['name'], 'trials': case_results})
    return results

def test_consistency(env, model, num_trials=10):
    print_header("TEST 4: Consistency Test")
    env.reset()
    env.reward_calc.current_phase = 1
    results = []
    for trial in range(num_trials):
        env.reset()
        result = run_episode(env, model)
        results.append(result)
        print(f"  Trial {trial+1}: FinalDist={result['final_distance_cm']:.1f}cm")
    
    rewards = [r['total_reward'] for r in results]
    distances = [r['final_distance_cm'] for r in results]
    return {
        'trials': results, 
        'reward_mean': np.mean(rewards), 
        'distance_mean': np.mean(distances),
        'distance_std': np.std(distances)
    }

def test_stochastic_vs_deterministic(env, model, num_trials=5):
    print_header("TEST 5: Deterministic vs Stochastic")
    det_results = []
    stoch_results = []
    
    print("  Running Deterministic...")
    for _ in range(num_trials):
        env.reset()
        det_results.append(run_episode(env, model, deterministic=True))
        
    print("  Running Stochastic...")
    for _ in range(num_trials):
        env.reset()
        stoch_results.append(run_episode(env, model, deterministic=False))
        
    return {'deterministic': det_results, 'stochastic': stoch_results}

def test_long_episode(env, model):
    print_header("TEST 6: Long Episode Analysis")
    env.reset()
    result = run_episode(env, model, max_steps=500)
    print(f"  Steps: {result['steps']}, Final Dist: {result['final_distance_cm']:.1f}cm")
    return result

def test_robustness_to_target_variation(env, model, num_trials=3):
    print_header("TEST 7: Target Variation")
    base_target = np.array([0.25, 0.15, 0.35])
    variations = [
        {'name': 'Default', 'target': base_target},
        {'name': 'Left Shift', 'target': base_target + np.array([0, 0.05, 0])},
        {'name': 'Right Shift', 'target': base_target + np.array([0, -0.05, 0])},
    ]
    results = []
    for var in variations:
        var_results = []
        for _ in range(num_trials):
            env.reset()
            env.target_pos = var['target'].copy()
            var_results.append(run_episode(env, model))
        avg_dist = np.mean([r['final_distance_cm'] for r in var_results])
        results.append({'variation': var['name'], 'avg_distance': avg_dist})
        print(f"  {var['name']}: Avg Dist {avg_dist:.1f}cm")
    return results

def generate_plots(all_results):
    print_header("GENERATING PLOTS")
    
    # --- Plot 1: Long Episode Trajectory ---
    if 'long_episode' in all_results:
        res = all_results['long_episode']
        traj = res['trajectory']
        steps = range(len(traj['palm_distances']))
        
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
        
        # Distance Plot
        ax1.plot(steps, traj['palm_distances'], color='blue', label='Distance')
        ax1.axhline(y=0, color='k', linestyle='--', alpha=0.3)
        ax1.set_ylabel('Palm Distance (cm)')
        ax1.set_title(f'Trajectory Analysis (Total Reward: {res["total_reward"]:.1f})')
        ax1.grid(True, alpha=0.3)
        
        # Overlap Plot
        ax2.plot(steps, traj['overlaps'], color='green', label='Overlap')
        ax2.set_xlabel('Step')
        ax2.set_ylabel('Overlap Volume (cm³)')
        ax2.grid(True, alpha=0.3)
        
        save_plot(fig, 'trajectory_analysis.png')

    # --- Plot 2: Scenario Comparison (Bar Chart) ---
    if 'standard' in all_results and 'angles' in all_results:
        # Combine data
        scenarios = [r['scenario'] for r in all_results['standard']]
        std_dists = [r['avg_final_distance_cm'] for r in all_results['standard']]
        
        angles = [r['angle'] for r in all_results['angles']]
        ang_dists = [r['avg_final_distance_cm'] for r in all_results['angles']]
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
        
        # Standard Positions
        bars1 = ax1.bar(scenarios, std_dists, color='skyblue')
        ax1.set_title('Performance vs Start Distance')
        ax1.set_ylabel('Final Distance (cm)')
        ax1.axhline(y=10.0, color='r', linestyle='--', label='Threshold (10cm)')
        ax1.legend()
        plt.setp(ax1.get_xticklabels(), rotation=45, ha='right')
        
        # Angles
        bars2 = ax2.bar(angles, ang_dists, color='lightgreen')
        ax2.set_title('Performance vs Approach Angle')
        ax2.axhline(y=10.0, color='r', linestyle='--')
        plt.setp(ax2.get_xticklabels(), rotation=45, ha='right')
        
        plt.tight_layout()
        save_plot(fig, 'scenario_performance.png')

    # --- Plot 3: Consistency Distribution ---
    if 'consistency' in all_results:
        cons = all_results['consistency']
        trials = cons['trials']
        final_dists = [t['final_distance_cm'] for t in trials]
        rewards = [t['total_reward'] for t in trials]
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        # Distance Boxplot
        ax1.boxplot(final_dists, patch_artist=True, boxprops=dict(facecolor="lightblue"))
        ax1.set_title('Consistency: Final Distance (10 Trials)')
        ax1.set_ylabel('Distance (cm)')
        ax1.set_xticklabels(['Model'])
        
        # Reward Scatter
        ax2.plot(range(1, 11), rewards, 'o-', color='purple')
        ax2.set_title('Reward Consistency')
        ax2.set_xlabel('Trial #')
        ax2.set_ylabel('Total Reward')
        ax2.grid(True)
        
        save_plot(fig, 'consistency_metrics.png')

    # --- Plot 4: Stochastic vs Deterministic ---
    if 'det_vs_stoch' in all_results:
        d_vs_s = all_results['det_vs_stoch']
        det_dists = [t['final_distance_cm'] for t in d_vs_s['deterministic']]
        stoch_dists = [t['final_distance_cm'] for t in d_vs_s['stochastic']]
        
        fig, ax = plt.subplots(figsize=(8, 6))
        data = [det_dists, stoch_dists]
        ax.boxplot(data, labels=['Deterministic', 'Stochastic'], patch_artist=True)
        ax.set_title('Policy Type Comparison')
        ax.set_ylabel('Final Distance (cm)')
        ax.grid(True, axis='y', alpha=0.3)
        
        save_plot(fig, 'policy_comparison.png')

def main():
    if len(sys.argv) < 2:
        print("Usage: python test_trained_model.py <model_path>")
        sys.exit(1)
    
    model_path = sys.argv[1]
    
    print("\n" + "#"*70)
    print("#  V7.6 SC-1 COMPREHENSIVE MODEL TESTING & PLOTTING")
    print("#"*70)
    print(f"  Model: {model_path}")
    
    env = V76Environment(vis=False, max_steps=500)
    model = PPO.load(model_path, env=env)
    
    all_results = {}
    
    try:
        all_results['standard'] = test_standard_positions(env, model)
        all_results['angles'] = test_approach_angles(env, model)
        all_results['edge_cases'] = test_edge_cases(env, model)
        all_results['consistency'] = test_consistency(env, model)
        all_results['det_vs_stoch'] = test_stochastic_vs_deterministic(env, model)
        all_results['long_episode'] = test_long_episode(env, model)
        all_results['target_variation'] = test_robustness_to_target_variation(env, model)
        
        # Generate plots
        generate_plots(all_results)
        
    except KeyboardInterrupt:
        print("\nTesting interrupted.")
    finally:
        env.close()
    
    print("\n" + "#"*70)
    print(f"#  TESTING COMPLETE. Plots saved in '{PLOT_DIR}/'")
    print("#"*70 + "\n")

if __name__ == "__main__":
    main()
