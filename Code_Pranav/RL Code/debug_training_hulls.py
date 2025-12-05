#!/usr/bin/env python3
"""
Debug script to capture what's happening during actual training steps
"""

import numpy as np
import sys
import pybullet as p
import os

# Add the correct paths
sys.path.append('/home/pralak/Space_Touch')
sys.path.append('/home/pralak/Space_Touch/Code_Pranav/RL Code')

# Import directly from the current directory since we'll run from there
from V5_ConvexHull_Overlap_Training import ConvexHullOverlapEnv
from reward_functions.convex_hull_envelopment_reward import ConvexHullEnvelopmentReward

def debug_training_hull_calculation():
    """Debug hull calculation during actual environment steps"""

    print("🔍 Debugging Hull Calculation During Training Steps...")

    # Create environment exactly like training
    env = ConvexHullOverlapEnv(num_envs=1, vis=False, max_steps=500)

    print("\n📋 Environment Information:")
    print(f"   Observation space: {env.observation_space}")
    print(f"   Action space: {env.action_space}")

    # Reset environment
    obs = env.reset()
    print(f"\n🔄 After Reset - Observation shape: {obs.shape}")

    # Take a few training-like steps and debug each one
    for step in range(5):
        print(f"\n{'='*50}")
        print(f"🔍 DEBUG STEP {step + 1}")
        print(f"{'='*50}")

        # Generate random action (like during training)
        # VecEnv expects batch of actions, so wrap in array
        single_action = env.action_space.sample()
        action_batch = np.array([single_action])  # Batch format for VecEnv
        print(f"Action: {single_action}")

        # Take step
        obs, reward, done, info = env.step(action_batch)

        # Handle VecEnv batch returns
        reward_val = reward[0] if hasattr(reward, '__len__') else reward
        done_val = done[0] if hasattr(done, '__len__') else done
        info_val = info[0] if hasattr(info, '__len__') else info

        print(f"Reward: {reward_val}, Done: {done_val}")

        # Get detailed observation breakdown
        print(f"\n📊 Observation Breakdown (Shape: {obs.shape}):")
        if len(obs.shape) > 1:
            obs_flat = obs[0]  # Handle VecEnv format
        else:
            obs_flat = obs

        print(f"   Base pos (0:3): {obs_flat[:3]}")
        print(f"   Target pos (3:6): {obs_flat[3:6]}")
        print(f"   Base vel (6:9): {obs_flat[6:9]}")
        print(f"   Base ang vel (9:12): {obs_flat[9:12]}")
        print(f"   Finger positions (12:24): {obs_flat[12:24]}")
        print(f"   Tactile (24:28): {obs_flat[24:28]}")

        # Check for invalid values
        has_nan = np.any(np.isnan(obs_flat))
        has_inf = np.any(np.isinf(obs_flat))
        print(f"   Contains NaN: {has_nan}, Contains Inf: {has_inf}")

        if has_nan or has_inf:
            print("   ⚠️ FOUND INVALID VALUES IN OBSERVATIONS!")
            invalid_indices = np.where(np.isnan(obs_flat) | np.isinf(obs_flat))[0]
            print(f"   Invalid at indices: {invalid_indices}")

        # Manually calculate reward to see what's happening
        print(f"\n🎯 Manual Reward Calculation:")

        try:
            # Extract data like the environment does
            base_pos = obs_flat[:3]
            target_pos = obs_flat[3:6]
            finger_data = obs_flat[12:24]
            tactile_data = obs_flat[24:28]

            print(f"   Base position: {base_pos}")
            print(f"   Target position: {target_pos}")

            # Check finger positions
            if len(finger_data) >= 12:
                finger_positions = finger_data[:12].reshape(4, 3)
                print(f"   Finger positions shape: {finger_positions.shape}")
                print(f"   Finger positions:\n{finger_positions}")

                # Check for invalid finger positions
                finger_nan = np.any(np.isnan(finger_positions))
                finger_inf = np.any(np.isinf(finger_positions))
                print(f"   Finger NaN: {finger_nan}, Finger Inf: {finger_inf}")

                if finger_nan or finger_inf:
                    print("   ⚠️ INVALID FINGER POSITIONS DETECTED!")

                # Check finger position ranges
                finger_min = np.min(finger_positions)
                finger_max = np.max(finger_positions)
                print(f"   Finger position range: [{finger_min:.4f}, {finger_max:.4f}]")

                # Calculate palm position like the environment
                if env.hand is not None:
                    finger_base_links = [0, 4, 8, 12]
                    finger_base_positions = []
                    for link_id in finger_base_links:
                        try:
                            link_state = p.getLinkState(env.hand, link_id)
                            finger_base_positions.append(np.array(link_state[0])[:3])
                        except Exception as e:
                            print(f"   ⚠️ Failed to get link {link_id} state: {e}")
                            finger_base_positions.append(np.array([0, 0, 0]))

                    palm_position = np.mean(finger_base_positions, axis=0)
                    print(f"   Calculated palm position: {palm_position}")

                    palm_nan = np.any(np.isnan(palm_position))
                    palm_inf = np.any(np.isinf(palm_position))
                    print(f"   Palm NaN: {palm_nan}, Palm Inf: {palm_inf}")

                    if palm_nan or palm_inf:
                        print("   ⚠️ INVALID PALM POSITION DETECTED!")

                else:
                    print("   ⚠️ Hand object is None!")
                    palm_position = base_pos  # Fallback

            else:
                print(f"   ⚠️ Insufficient finger data: {len(finger_data)} < 12")
                finger_positions = np.zeros((4, 3))
                palm_position = base_pos

            # Test reward calculation
            reward_obs = {
                'finger_positions': finger_positions,
                'palm_position': palm_position,
                'object_pos': target_pos,
                'binary_contact': tactile_data,
                'episode_step': step,
            }

            print(f"\n🧮 Testing Hull Generation:")

            # Test hull generation manually
            reward_calc = env.reward_calculator

            # Test object hull
            object_hull_points = reward_calc.generate_object_hull_points(target_pos)
            print(f"   Object hull points shape: {object_hull_points.shape}")
            print(f"   Object hull points sample: {object_hull_points[0]}")

            obj_valid, obj_volume, obj_error = reward_calc.validate_hull(object_hull_points, "object")
            print(f"   Object hull - Valid: {obj_valid}, Volume: {obj_volume}, Error: '{obj_error}'")

            # Test hand hull
            hand_hull_points = np.vstack([finger_positions, palm_position.reshape(1, 3)])
            print(f"   Hand hull points shape: {hand_hull_points.shape}")
            print(f"   Hand hull points:\n{hand_hull_points}")

            hand_valid, hand_volume, hand_error = reward_calc.validate_hull(hand_hull_points, "hand")
            print(f"   Hand hull - Valid: {hand_valid}, Volume: {hand_volume}, Error: '{hand_error}'")

            if not hand_valid:
                print(f"   ❌ HAND HULL INVALID: {hand_error}")
            if not obj_valid:
                print(f"   ❌ OBJECT HULL INVALID: {obj_error}")

            # Try full reward calculation
            total_reward, reward_info = reward_calc.calculate_reward(reward_obs)
            print(f"\n🎯 Full Reward Calculation:")
            print(f"   Total reward: {total_reward}")
            print(f"   Overlap volume: {reward_info.get('overlap_volume', 0)}")
            print(f"   Hand hull volume: {reward_info.get('hand_hull_volume', 0)}")
            print(f"   Object hull volume: {reward_info.get('object_hull_volume', 0)}")
            print(f"   Error: '{reward_info.get('error', '')}'")

        except Exception as e:
            print(f"   ❌ Reward calculation failed: {e}")
            import traceback
            traceback.print_exc()

        # Check if episode ended
        if done_val:
            print(f"\n🔄 Episode ended, resetting...")
            obs = env.reset()

        print(f"\n" + "="*50)

    env.close()
    print("\n✅ Debug complete!")

if __name__ == "__main__":
    debug_training_hull_calculation()