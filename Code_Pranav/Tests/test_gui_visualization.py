#!/usr/bin/env python3
"""
Simple test script to verify PyBullet GUI displays the Allegro hand and sphere objects.
This should open a 3D visualization window with:
- Allegro hand model
- Colored spheres around the hand
- Camera positioned to view the hand
"""

import time
import pybullet as p
import pybullet_data
import pybulletX as px

# Configuration
URDF_HAND = "/home/pralak/Space_touch/examples/allegro_hand_description/allegro_hand_description_left_digit.urdf"
URDF_SPHERE = "/home/pralak/Space_touch/examples/objects/sphere_small.urdf"

def main():
    print("🤖 Starting PyBullet GUI visualization test...")
    
    # Initialize PyBullet with GUI mode
    print("📺 Opening PyBullet GUI window...")
    px.init(mode=p.GUI)
    
    # Standard PyBullet setup
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    p.setGravity(0, 0, -9.81)
    p.setRealTimeSimulation(0)
    
    # Load ground plane
    print("🏠 Loading ground plane...")
    p.loadURDF("plane.urdf", [0, 0, 0], useFixedBase=True)
    
    # Set camera position to view the hand
    print("📷 Setting camera position...")
    p.resetDebugVisualizerCamera(
        cameraDistance=0.6,
        cameraYaw=60,
        cameraPitch=-30,
        cameraTargetPosition=[0, 0, 0.25]
    )
    
    # Load Allegro hand
    print("✋ Loading Allegro hand model...")
    try:
        hand = p.loadURDF(URDF_HAND, [0, 0, 0.25], useFixedBase=True)
        print(f"✅ Hand loaded successfully! Body ID: {hand}")
    except Exception as e:
        print(f"❌ Failed to load hand: {e}")
        return
    
    # Load colorful spheres around the hand
    print("🔴🟢🔵 Adding colored spheres...")
    sphere_positions = [
        ([0.1, 0.1, 0.3], [1, 0, 0, 1]),   # Red sphere
        ([-0.1, 0.1, 0.3], [0, 1, 0, 1]),  # Green sphere
        ([0, -0.1, 0.3], [0, 0, 1, 1]),    # Blue sphere
        ([0.1, -0.1, 0.4], [1, 1, 0, 1]),  # Yellow sphere
    ]
    
    spheres = []
    for i, (pos, color) in enumerate(sphere_positions):
        try:
            sphere = p.loadURDF(URDF_SPHERE, pos)
            p.changeVisualShape(sphere, -1, rgbaColor=color)
            spheres.append(sphere)
            print(f"  ✅ Sphere {i+1} loaded at {pos} with color {color[:3]}")
        except Exception as e:
            print(f"  ❌ Failed to load sphere {i+1}: {e}")
    
    # Display joint information
    print("\n📋 Hand joint information:")
    joint_count = p.getNumJoints(hand)
    print(f"Total joints: {joint_count}")
    
    for i in range(min(5, joint_count)):  # Show first 5 joints
        joint_info = p.getJointInfo(hand, i)
        joint_name = joint_info[1].decode('utf-8')
        joint_type = joint_info[2]
        print(f"  Joint {i}: {joint_name} (type: {joint_type})")
    
    print("\n🎯 Visualization ready!")
    print("Expected to see:")
    print("  • Allegro hand model (gray/white)")
    print("  • Ground plane")
    print("  • 4 colored spheres around the hand")
    print("  • Camera positioned to view the scene")
    print("\nIf running on a machine with display, the PyBullet GUI window should be visible.")
    print("In headless mode, you'll see OpenGL context warnings (expected).")
    
    # Simple animation - move spheres in a circle
    print("\n🔄 Starting simple animation...")
    t = 0
    for step in range(100):  # Run for 100 steps
        # Simple sine wave motion for spheres
        for i, sphere in enumerate(spheres):
            if sphere is not None:
                base_pos = sphere_positions[i][0]
                new_pos = [
                    base_pos[0] + 0.05 * np.sin(t + i),
                    base_pos[1] + 0.05 * np.cos(t + i),
                    base_pos[2]
                ]
                p.resetBasePositionAndOrientation(sphere, new_pos, [0, 0, 0, 1])
        
        # Step simulation
        p.stepSimulation()
        time.sleep(1.0/240.0)  # 240 Hz
        t += 0.1
        
        if step % 20 == 0:
            print(f"  Animation step {step}/100")
    
    print("\n✅ Test completed successfully!")
    print("If you saw the PyBullet GUI window with the hand and moving spheres, visualization is working!")
    
    # Keep window open for a moment
    time.sleep(2)
    p.disconnect()

if __name__ == "__main__":
    # Import numpy here to avoid dependency issues
    try:
        import numpy as np
    except ImportError:
        print("❌ NumPy not available, using basic math")
        import math
        np = math  # Simple fallback
    
    main()