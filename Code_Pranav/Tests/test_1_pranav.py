#!/usr/bin/env python3
import cv2
import pybullet as p
import pybulletX as px
import os, tacto


pkg_examples = "/home/pranav/Space_touch/examples/objects/"
sphere_urdf  = os.path.join(pkg_examples, "sphere_small.urdf")

def main():
    # 1) Load the DIGIT background image (already in the repo)
    bg = cv2.imread("examples/conf/bg_digit_240_320.jpg")

    # 2) Create the TACTO Sensor for DIGIT
    sensor = tacto.Sensor(
        width=120,
        height=160,
        background=bg,
        config_path=tacto.get_digit_config_path(),  # default DIGIT config :contentReference[oaicite:0]{index=0}
    )

    # 3) Initialize PyBullet + camera
    px.init(mode=p.GUI)
    p.resetDebugVisualizerCamera(
        cameraDistance=0.12,
        cameraYaw=90,
        cameraPitch=-45,
        cameraTargetPosition=[0, 0, 0],
    )

    # 4) Load the DIGIT URDF and attach a camera
    digit = px.Body(
        urdf_path="meshes/digit.urdf",
        base_position=[0, 0, 0],
        base_orientation=[0, -0.707106, 0, 0.707106],
        use_fixed_base=True,
    )
    sensor.add_camera(digit.id, link_ids=[-1])  # add base link as camera :contentReference[oaicite:1]{index=1}

    # 5) Add a small sphere to make contact
    sphere = px.Body(
        urdf_path=sphere_urdf,
        base_position=[-0.02, 0.0, 0.1],
        global_scaling=0.15,
        use_fixed_base=False,
    )
    sensor.add_body(sphere)

    # 6) Step the simulation so the sphere falls onto the gel
    for _ in range(100):
        p.stepSimulation()

    # 7) Capture one frame of RGB + depth
    colors, depths = sensor.render()  # returns lists of NumPy arrays :contentReference[oaicite:2]{index=2}

    # 8) Display the results
    sensor.updateGUI(colors, depths)    # pops up an OpenCV window
    cv2.waitKey(0)

if __name__ == "__main__":
    main()
