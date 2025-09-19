#!/usr/bin/env python3
import os
import cv2
import pybullet as p
import pybulletX as px
import tacto

# paths
sphere_urdf  = "/home/pranav/Space_touch/examples/objects/sphere_small.urdf"
hand_urdf    = "/home/pranav/Space_touch/examples/allegro_hand_description/allegro_hand_description_left_digit.urdf"

def main():
    # 1) Load DIGIT background
    bg = cv2.imread("examples/conf/bg_digit_240_320.jpg")

    # 2) Create TACTO Sensor
    sensor = tacto.Sensor(
        width=120,
        height=160,
        background=bg,
        config_path=tacto.get_digit_config_path(),
    )

    # 3) Init PyBullet + camera view
    px.init(mode=p.GUI)
    p.resetDebugVisualizerCamera(
        cameraDistance=0.20,
        cameraYaw=90,
        cameraPitch=-45,
        cameraTargetPosition=[0, 0, 0],
    )

    # 4) Spawn your Allegro hand URDF
    gripper = px.Body(
        urdf_path=hand_urdf,
        base_position=[0, 0, 0.18],
        base_orientation=[0, 0, 0, 1],
        use_fixed_base=True,
    )

    # 5) Attach DIGIT camera to the base link (change link_ids if needed)
    sensor.add_camera(gripper.id, link_ids=[0])
    # print(gripper.id)

    # 6) Spawn a sphere in front of the fingers
    sphere = px.Body(
        urdf_path=sphere_urdf,
        base_position=[0.05, 0, 0.3],
        global_scaling=0.15,
        use_fixed_base=False,
    )
    sensor.add_body(sphere)

    # 7) Step simulation so it drops
    for _ in range(200):
        p.stepSimulation()

    # 8) Render and display
    colors, depths = sensor.render()
    sensor.updateGUI(colors, depths)
    cv2.waitKey(0)

if __name__ == "__main__":
    main()
