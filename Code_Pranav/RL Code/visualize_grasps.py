#!/usr/bin/env python3
"""
PRIORITY 7: Visualization Script for Gut-Check Verification
Provides visual verification of learned grasping behavior with annotations
"""

import time
import numpy as np
import pybullet as p
import pybullet_data
from pathlib import Path
import argparse
from stable_baselines3 import PPO
from Wandb_SC_1_Enhanced_V2 import TendonAllegroReachingEnv, TEST_SCENARIOS
import sys
import os

# Fix for attrdict Python 3.13 compatibility
import collections.abc
import collections
collections.Mapping = collections.abc.Mapping
collections.MutableMapping = collections.abc.MutableMapping
collections.Sequence = collections.abc.Sequence
collections.MutableSequence = collections.abc.MutableSequence


def draw_contact_indicators(env, binary_tactile, finger_positions):
    """Draw visual indicators for tactile contacts"""
    contact_markers = []

    finger_positions_reshaped = finger_positions.reshape(4, 3)
    finger_names = ["Thumb", "Index", "Middle", "Ring"]

    for i, (pos, contact, name) in enumerate(zip(finger_positions_reshaped, binary_tactile, finger_names)):
        color = [0, 1, 0] if contact > 0.5 else [0.5, 0.5, 0.5]  # Green if contact, gray if not

        # Draw sphere at fingertip
        marker_id = p.addUserDebugText(
            text=f"{name}: {'ON' if contact > 0.5 else 'OFF'}",
            textPosition=pos + np.array([0, 0, 0.03]),
            textColorRGB=color,
            textSize=1.0,
            lifeTime=0.1
        )
        contact_markers.append(marker_id)

        # Draw small sphere at fingertip
        if contact > 0.5:
            sphere_id = p.addUserDebugText(
                text="●",
                textPosition=pos,
                textColorRGB=[0, 1, 0],
                textSize=2.0,
                lifeTime=0.1
            )
            contact_markers.append(sphere_id)

    return contact_markers


def draw_convex_hull_wireframe(env):
    """Draw wireframe visualization of convex hull"""
    try:
        from scipy.spatial import ConvexHull

        # Get fingertip positions
        finger_positions = env._get_finger_positions().reshape(4, 3)

        # Get palm position
        palm_pos, _ = p.getBasePositionAndOrientation(env.hand)
        palm_pos = np.array(palm_pos)

        # Create convex hull from 5 points (4 fingertips + palm)
        points = np.vstack([finger_positions, palm_pos])

        hull = ConvexHull(points)
        hull_lines = []

        # Draw hull edges in yellow
        for simplex in hull.simplices:
            for i in range(len(simplex)):
                start_point = points[simplex[i]]
                end_point = points[simplex[(i + 1) % len(simplex)]]

                line_id = p.addUserDebugLine(
                    lineFromXYZ=start_point,
                    lineToXYZ=end_point,
                    lineColorRGB=[1, 1, 0],
                    lineWidth=2.0,
                    lifeTime=0.1
                )
                hull_lines.append(line_id)

        return hull_lines, hull.volume

    except Exception as e:
        print(f"Could not draw convex hull: {e}")
        return [], 0.0


def clear_debug_items(item_ids):
    """Clear debug visualization items"""
    for item_id in item_ids:
        try:
            p.removeUserDebugItem(item_id)
        except:
            pass


def visualize_trained_policy(model_path, num_episodes=5, scenario="static_close", slow_motion_factor=0.02, record_video=False):
    """
    Visualize trained grasping policy with comprehensive annotations

    Args:
        model_path: Path to trained model
        num_episodes: Number of episodes to visualize
        scenario: Test scenario to run
        slow_motion_factor: Time delay between steps (seconds)
        record_video: Whether to record video
    """

    print("=" * 80)
    print("SC-1 GRASP VISUALIZATION - GUT CHECK VERIFICATION")
    print("=" * 80)
    print(f"Model: {model_path}")
    print(f"Scenario: {scenario}")
    print(f"Episodes: {num_episodes}")
    print(f"Slow motion factor: {slow_motion_factor}s per step")
    print("=" * 80)

    # Load model
    if not Path(model_path).exists():
        print(f"Error: Model file not found: {model_path}")
        return

    try:
        print("Loading trained model...")
        model = PPO.load(model_path)
        print("✓ Model loaded successfully")
    except Exception as e:
        print(f"Error loading model: {e}")
        return

    # Create environment with GUI visualization
    print("Setting up visualization environment...")
    env = TendonAllegroReachingEnv(
        vis=True,  # Enable GUI
        control_smoothing=True,
        filter_cutoff=15.0,
        test_scenario=scenario
    )
    env.set_test_mode(True)

    # Configure camera for better viewing
    p.resetDebugVisualizerCamera(
        cameraDistance=0.8,
        cameraYaw=45,
        cameraPitch=-30,
        cameraTargetPosition=[0.2, 0.1, 0.3]
    )

    print("✓ Visualization environment ready")

    if record_video:
        # Setup video recording
        video_path = Path(model_path).parent / f"visualization_{scenario}_{int(time.time())}.mp4"
        p.startStateLogging(p.STATE_LOGGING_VIDEO_MP4, str(video_path))
        print(f"📹 Recording video to: {video_path}")

    episode_results = []

    for episode in range(num_episodes):
        print(f"\n{'='*40}")
        print(f"EPISODE {episode + 1}/{num_episodes}")
        print(f"{'='*40}")

        # Reset environment with verification
        obs = env.reset()
        print("✓ Environment reset complete")

        # Verify reset worked
        assert env.hand is not None, "Hand not created after reset"
        assert env.target_sphere is not None, "Target not created after reset"

        # Get initial state
        base_pos, _ = p.getBasePositionAndOrientation(env.hand)
        target_pos, _ = p.getBasePositionAndOrientation(env.target_sphere)
        initial_distance = np.linalg.norm(np.array(base_pos) - np.array(target_pos))

        print(f"Initial distance: {initial_distance:.3f}m")
        print(f"Hand position: {base_pos}")
        print(f"Target position: {target_pos}")

        episode_reward = 0
        episode_steps = 0
        contact_history = []
        hull_history = []

        # Episode loop
        while episode_steps < 1000:  # Max steps per episode
            # Predict action
            action, _ = model.predict(obs, deterministic=True)

            # Execute step
            obs, reward, done, info = env.step(action)
            episode_reward += reward[0]
            episode_steps += 1

            # Get current state for visualization
            current_distance = info[0].get('distance', float('inf'))
            success = info[0].get('success', False)
            tactile_contacts = info[0].get('tactile_contacts', [0,0,0,0])
            multi_finger_contacts = info[0].get('multi_finger_contact_count', 0)
            inside_hull = info[0].get('inside_hull', False)
            hull_volume = info[0].get('hull_volume', 0.0)

            # Track history
            contact_history.append(multi_finger_contacts)
            hull_history.append(inside_hull)

            # Get finger positions for visualization
            finger_positions = env._get_finger_positions()
            binary_tactile = np.array(tactile_contacts)

            # Clear previous debug items
            if episode_steps > 1:
                clear_debug_items(getattr(env, '_last_contact_markers', []))
                clear_debug_items(getattr(env, '_last_hull_lines', []))
                clear_debug_items(getattr(env, '_last_text_ids', []))

            # Draw contact indicators
            contact_markers = draw_contact_indicators(env, binary_tactile, finger_positions)
            env._last_contact_markers = contact_markers

            # Draw convex hull wireframe
            hull_lines, current_hull_volume = draw_convex_hull_wireframe(env)
            env._last_hull_lines = hull_lines

            # Create main status text overlay
            status_text = f"""STEP {episode_steps}
Distance: {current_distance:.3f}m
Active Fingers: {multi_finger_contacts}/4
Inside Hull: {'YES' if inside_hull else 'NO'}
Hull Volume: {hull_volume:.6f}m³
Success: {'YES' if success else 'NO'}
Reward: {reward[0]:.2f}
Total Reward: {episode_reward:.1f}"""

            # Position text in upper left of view
            text_id = p.addUserDebugText(
                text=status_text,
                textPosition=[0, 0, 0.8],
                textColorRGB=[1, 1, 1],  # White text
                textSize=1.2,
                lifeTime=0.1
            )

            # Individual finger status
            finger_status_text = f"""TACTILE STATUS:
Thumb: {'●' if binary_tactile[0] > 0.5 else '○'}
Index: {'●' if binary_tactile[1] > 0.5 else '○'}
Middle: {'●' if binary_tactile[2] > 0.5 else '○'}
Ring: {'●' if binary_tactile[3] > 0.5 else '○'}"""

            finger_text_id = p.addUserDebugText(
                text=finger_status_text,
                textPosition=[0, 0, 0.6],
                textColorRGB=[0, 1, 0] if multi_finger_contacts >= 2 else [1, 0, 0],
                textSize=1.0,
                lifeTime=0.1
            )

            env._last_text_ids = [text_id, finger_text_id]

            # Print step summary
            if episode_steps % 50 == 0 or success or done[0]:
                status = "SUCCESS" if success else "ONGOING" if not done[0] else "FAILED"
                tactile_status = f"{multi_finger_contacts}F" if multi_finger_contacts >= 2 else f"{multi_finger_contacts}F!"
                hull_status = "IN" if inside_hull else "OUT"
                print(f"  Step {episode_steps:3d}: {status:7s} | Dist={current_distance:.3f}m | Tactile={tactile_status} | Hull={hull_status} | R={reward[0]:+.2f}")

            # Slow motion delay
            time.sleep(slow_motion_factor)

            # Check termination
            if done[0]:
                break

        # Episode summary
        final_distance = info[0].get('distance', float('inf'))
        final_success = info[0].get('success', False)
        avg_contacts = np.mean(contact_history) if contact_history else 0
        hull_rate = np.mean(hull_history) if hull_history else 0

        episode_result = {
            'episode': episode + 1,
            'success': final_success,
            'final_distance': final_distance,
            'steps': episode_steps,
            'total_reward': episode_reward,
            'avg_contacts': avg_contacts,
            'hull_containment_rate': hull_rate,
            'initial_distance': initial_distance
        }

        episode_results.append(episode_result)

        print(f"\n📊 EPISODE {episode + 1} SUMMARY:")
        print(f"   Success: {'✓' if final_success else '✗'}")
        print(f"   Final distance: {final_distance:.3f}m (started at {initial_distance:.3f}m)")
        print(f"   Steps taken: {episode_steps}")
        print(f"   Total reward: {episode_reward:.1f}")
        print(f"   Avg finger contacts: {avg_contacts:.1f}")
        print(f"   Hull containment rate: {hull_rate:.1%}")

        # Pause between episodes
        if episode < num_episodes - 1:
            print(f"\nPress ENTER for next episode...")
            input()

    # Final cleanup
    clear_debug_items(getattr(env, '_last_contact_markers', []))
    clear_debug_items(getattr(env, '_last_hull_lines', []))
    clear_debug_items(getattr(env, '_last_text_ids', []))

    if record_video:
        p.stopStateLogging(p.STATE_LOGGING_VIDEO_MP4)
        print(f"✓ Video saved to: {video_path}")

    env.close()

    # Overall summary
    print("\n" + "=" * 80)
    print("VISUALIZATION SUMMARY")
    print("=" * 80)

    successes = sum(r['success'] for r in episode_results)
    success_rate = successes / len(episode_results)
    avg_final_distance = np.mean([r['final_distance'] for r in episode_results])
    avg_steps = np.mean([r['steps'] for r in episode_results])
    avg_reward = np.mean([r['total_reward'] for r in episode_results])
    avg_contacts = np.mean([r['avg_contacts'] for r in episode_results])
    avg_hull_rate = np.mean([r['hull_containment_rate'] for r in episode_results])

    print(f"Overall Success Rate: {successes}/{len(episode_results)} ({success_rate:.1%})")
    print(f"Average Final Distance: {avg_final_distance:.3f}m")
    print(f"Average Episode Steps: {avg_steps:.1f}")
    print(f"Average Episode Reward: {avg_reward:.1f}")
    print(f"Average Finger Contacts: {avg_contacts:.1f}")
    print(f"Average Hull Containment: {avg_hull_rate:.1%}")

    # Gut-check verification
    print(f"\n🔍 GUT-CHECK VERIFICATION:")
    tactile_engaged = avg_contacts >= 2.0
    spatial_engulfment = avg_hull_rate >= 0.5

    print(f"   Tactile Engagement: {'✓ PASS' if tactile_engaged else '✗ FAIL'} (avg {avg_contacts:.1f} fingers, need 2+)")
    print(f"   Spatial Engulfment: {'✓ PASS' if spatial_engulfment else '✗ FAIL'} (hull rate {avg_hull_rate:.1%}, need 50%+)")
    print(f"   Overall Assessment: {'✓ LEARNED GRASPING' if (tactile_engaged and spatial_engulfment and success_rate > 0.3) else '✗ NEEDS IMPROVEMENT'}")

    print("=" * 80)

    return episode_results


def compare_models(model_paths, scenario="static_close", episodes_per_model=3):
    """Compare multiple models side by side"""
    print("=" * 80)
    print("MODEL COMPARISON VISUALIZATION")
    print("=" * 80)

    all_results = {}

    for i, model_path in enumerate(model_paths):
        print(f"\nTesting Model {i+1}: {Path(model_path).name}")
        results = visualize_trained_policy(
            model_path=model_path,
            num_episodes=episodes_per_model,
            scenario=scenario,
            slow_motion_factor=0.01  # Faster for comparison
        )
        all_results[Path(model_path).name] = results

    # Comparison summary
    print("\n" + "=" * 80)
    print("MODEL COMPARISON SUMMARY")
    print("=" * 80)

    for model_name, results in all_results.items():
        successes = sum(r['success'] for r in results)
        success_rate = successes / len(results)
        avg_contacts = np.mean([r['avg_contacts'] for r in results])
        avg_hull_rate = np.mean([r['hull_containment_rate'] for r in results])

        print(f"{model_name}:")
        print(f"  Success: {success_rate:.1%} | Contacts: {avg_contacts:.1f} | Hull: {avg_hull_rate:.1%}")

    print("=" * 80)


def main():
    """Main function with command line arguments"""
    parser = argparse.ArgumentParser(description="Visualize trained SC-1 grasping policy")
    parser.add_argument("model_path", help="Path to trained model (.zip file)")
    parser.add_argument("--episodes", type=int, default=5, help="Number of episodes to visualize")
    parser.add_argument("--scenario", choices=list(TEST_SCENARIOS.keys()), default="static_close",
                       help="Test scenario to run")
    parser.add_argument("--speed", type=float, default=0.02, help="Slow motion factor (seconds per step)")
    parser.add_argument("--record", action="store_true", help="Record video")
    parser.add_argument("--compare", nargs="+", help="Compare multiple models (provide multiple paths)")

    args = parser.parse_args()

    if args.compare:
        # Compare multiple models
        compare_models(args.compare, scenario=args.scenario, episodes_per_model=args.episodes)
    else:
        # Visualize single model
        visualize_trained_policy(
            model_path=args.model_path,
            num_episodes=args.episodes,
            scenario=args.scenario,
            slow_motion_factor=args.speed,
            record_video=args.record
        )


if __name__ == "__main__":
    # If no command line args, run with defaults for quick testing
    if len(sys.argv) == 1:
        print("No arguments provided. Example usage:")
        print("python visualize_grasps.py path/to/model.zip --episodes 3 --scenario static_close --record")
        print("\nAvailable scenarios:", list(TEST_SCENARIOS.keys()))
        sys.exit(1)

    main()