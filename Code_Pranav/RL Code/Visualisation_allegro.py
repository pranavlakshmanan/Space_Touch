#!/usr/bin/env python3
"""
Visualization and debugging tool for the Allegro Hand Tactile Environment
"""

import numpy as np
import cv2
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import time

from allegro_hand_env import AllegroHandTactileEnv


class AllegroHandVisualizer:
    """
    Visualizer for debugging and understanding the Allegro hand environment.
    """
    
    def __init__(self, env):
        self.env = env
        self.fig = None
        self.axes = None
        self.images = None
        
    def visualize_observations(self, obs, info=None):
        """
        Display tactile sensor observations in a grid.
        """
        n_sensors = obs.shape[1]
        
        if self.fig is None:
            # Create figure on first call
            self.fig, self.axes = plt.subplots(1, n_sensors, figsize=(3*n_sensors, 3))
            if n_sensors == 1:
                self.axes = [self.axes]
            
            self.images = []
            for i, ax in enumerate(self.axes):
                ax.set_title(f"Sensor {i}")
                ax.axis('off')
                img = ax.imshow(obs[0, i], cmap='gray', vmin=0, vmax=255)
                self.images.append(img)
            
            plt.tight_layout()
        else:
            # Update existing images
            for i, img in enumerate(self.images):
                img.set_data(obs[0, i])
        
        # Add info text if provided
        if info is not None:
            self.fig.suptitle(
                f"Contacts: {info.get('num_contacts', 0):.1f} | "
                f"Distance: {info.get('distance', 0):.3f} | "
                f"Grasping: {info.get('is_grasping', False)}",
                fontsize=12
            )
        
        plt.draw()
        plt.pause(0.001)
    
    def test_action_space(self):
        """
        Test different action components to understand their effects.
        """
        print("Testing Action Space Components...")
        print(f"Action space shape: {self.env.action_space.shape}")
        print(f"Action components:")
        print(f"  - Base linear velocity (x,y,z): actions[0:3]")
        print(f"  - Base angular velocity (x,y,z): actions[3:6]")
        print(f"  - Joint velocities: actions[6:]")
        print()
        
        # Test sequences
        test_sequences = [
            ("Move base forward (x)", lambda: self._create_action([1, 0, 0, 0, 0, 0])),
            ("Move base up (z)", lambda: self._create_action([0, 0, 1, 0, 0, 0])),
            ("Rotate base (yaw)", lambda: self._create_action([0, 0, 0, 0, 0, 1])),
            ("Close all fingers", lambda: self._create_action([0, 0, 0, 0, 0, 0], joint_vel=-1)),
            ("Open all fingers", lambda: self._create_action([0, 0, 0, 0, 0, 0], joint_vel=1)),
            ("Complex motion", lambda: self._create_action([0.5, 0, 0.3, 0, 0.2, 0], joint_vel=0.5)),
        ]
        
        for name, action_fn in test_sequences:
            print(f"\nTesting: {name}")
            obs = self.env.reset()
            
            for _ in range(50):
                action = action_fn()
                obs, reward, done, info = self.env.step(action)
                self.visualize_observations(obs, info[0])
                time.sleep(0.05)
                
                if done[0]:
                    break
    
    def _create_action(self, base_action, joint_vel=0):
        """
        Helper to create action array.
        """
        n_joints = len(self.env.joint_indices)
        action = np.zeros(6 + n_joints)
        action[:6] = base_action
        action[6:] = joint_vel
        return np.expand_dims(action, axis=0)
    
    def interactive_control(self):
        """
        Interactive control using keyboard.
        """
        print("\nInteractive Control Mode")
        print("Controls:")
        print("  W/S: Move forward/backward")
        print("  A/D: Move left/right")
        print("  Q/E: Move up/down")
        print("  I/K: Pitch rotation")
        print("  J/L: Yaw rotation")
        print("  U/O: Roll rotation")
        print("  C: Close fingers")
        print("  V: Open fingers")
        print("  R: Reset environment")
        print("  ESC: Exit")
        
        cv2.namedWindow("Tactile Sensors", cv2.WINDOW_NORMAL)
        
        obs = self.env.reset()
        base_vel = np.zeros(6)
        joint_vel = 0
        
        while True:
            # Get keyboard input
            key = cv2.waitKey(1) & 0xFF
            
            # Reset velocities
            base_vel *= 0.9  # Damping
            joint_vel *= 0.9
            
            # Process keys
            if key == ord('w'): base_vel[0] = 0.5
            elif key == ord('s'): base_vel[0] = -0.5
            elif key == ord('a'): base_vel[1] = 0.5
            elif key == ord('d'): base_vel[1] = -0.5
            elif key == ord('q'): base_vel[2] = 0.5
            elif key == ord('e'): base_vel[2] = -0.5
            elif key == ord('i'): base_vel[3] = 0.5
            elif key == ord('k'): base_vel[3] = -0.5
            elif key == ord('j'): base_vel[4] = 0.5
            elif key == ord('l'): base_vel[4] = -0.5
            elif key == ord('u'): base_vel[5] = 0.5
            elif key == ord('o'): base_vel[5] = -0.5
            elif key == ord('c'): joint_vel = -1.0
            elif key == ord('v'): joint_vel = 1.0
            elif key == ord('r'): 
                obs = self.env.reset()
                print("Environment reset!")
            elif key == 27:  # ESC
                break
            
            # Create action
            action = self._create_action(base_vel, joint_vel)
            
            # Step environment
            obs, reward, done, info = self.env.step(action)
            
            # Visualize tactile images
            sensor_imgs = []
            for i in range(obs.shape[1]):
                img = obs[0, i]
                img = cv2.resize(img, (160, 120))
                # Add labels
                cv2.putText(img, f"S{i}", (5, 15), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, 255, 1)
                sensor_imgs.append(img)
            
            # Concatenate images
            display = np.hstack(sensor_imgs)
            
            # Add info text
            info_text = (f"Reward: {reward[0]:.3f} | "
                        f"Contacts: {info[0]['num_contacts']:.1f} | "
                        f"Grasping: {info[0]['is_grasping']}")
            cv2.putText(display, info_text, (10, display.shape[0]-10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, 255, 1)
            
            cv2.imshow("Tactile Sensors", display)
            
            if done[0]:
                print(f"Episode finished! Success: {info[0]['success']}")
                obs = self.env.reset()
        
        cv2.destroyAllWindows()
        plt.close('all')
    
    def analyze_reward_components(self, n_steps=200):
        """
        Analyze different components of the reward function.
        """
        print("\nAnalyzing Reward Components...")
        
        obs = self.env.reset()
        
        reward_components = {
            'total': [],
            'contact': [],
            'distance': [],
            'stability': [],
            'grasp': [],
            'action': []
        }
        
        # Random policy for analysis
        for step in range(n_steps):
            action = self.env.action_space.sample()
            action = np.expand_dims(action, axis=0)
            
            # Store state before step
            if hasattr(self.env, 'hand') and hasattr(self.env, 'sphere'):
                hand_pos_before, _ = p.getBasePositionAndOrientation(self.env.hand.id)
                sphere_pos_before, _ = p.getBasePositionAndOrientation(self.env.sphere.id)
            
            obs, reward, done, info = self.env.step(action)
            
            # Approximate reward components (you'd need to expose these from the env)
            reward_components['total'].append(reward[0])
            reward_components['contact'].append(info[0]['num_contacts'] / 4.0 * 2.0)
            reward_components['distance'].append(np.exp(-info[0]['distance'] * 5.0))
            
            if done[0]:
                obs = self.env.reset()
        
        # Plot reward components
        plt.figure(figsize=(12, 8))
        
        for i, (name, values) in enumerate(reward_components.items()):
            if len(values) > 0:
                plt.subplot(2, 3, i+1)
                plt.plot(values)
                plt.title(f"{name.capitalize()} Reward")
                plt.xlabel("Step")
                plt.ylabel("Reward")
                plt.grid(True)
        
        plt.tight_layout()
        plt.show()


def main():
    """
    Main function to run different visualization modes.
    """
    import argparse
    
    parser = argparse.ArgumentParser(description="Allegro Hand Environment Visualizer")
    parser.add_argument('--mode', choices=['test', 'interactive', 'analyze'], 
                       default='interactive',
                       help='Visualization mode')
    parser.add_argument('--max-steps', type=int, default=500,
                       help='Maximum steps per episode')
    args = parser.parse_args()
    
    # Create environment
    env = AllegroHandTactileEnv(
        num_envs=1,
        vis=True,
        max_steps=args.max_steps,
        sensor_res=(60, 80)
    )
    
    # Create visualizer
    viz = AllegroHandVisualizer(env)
    
    try:
        if args.mode == 'test':
            viz.test_action_space()
        elif args.mode == 'interactive':
            viz.interactive_control()
        elif args.mode == 'analyze':
            viz.analyze_reward_components()
    finally:
        env.close()


if __name__ == "__main__":
    # If you need to import pybullet for the analyze function
    import pybullet as p
    main()