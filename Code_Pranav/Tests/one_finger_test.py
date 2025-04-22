# one_finger_example.py
# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import time
import pybullet as p
import pybullet_data
from tacto.sensor import Sensor, get_digit_config_path

# 1) Launch PyBullet
p.connect(p.GUI)
p.setGravity(0, 0, -9.81)
p.setAdditionalSearchPath(pybullet_data.getDataPath())

# 2) Load your one‑finger Allegro + DIGIT URDF
hand_urdf = "/home/pranav/tacto/examples/allegro_hand_description/allegro_hand_description_left.urdf"
hand_id = p.loadURDF(hand_urdf,
                    basePosition=[0, 0, 0.15],
                    useFixedBase=True)

# 3) Create the TACTO sensor
sensor = Sensor(
    width=160,
    height=120,
    config_path=get_digit_config_path(),
    visualize_gui=True,
    show_depth=True,
    cid=0,
)

# 4) Tell TACTO about your hand so it mirrors the scene
class DummyBody:
    pass

body = DummyBody()
body.urdf_path = hand_urdf
body.id = hand_id
body.global_scaling = 1.0
sensor.add_body(body)

# 5) Attach the DIGIT camera to your finger’s tip link
#    (print joint names to find the right index)
for i in range(-1, p.getNumJoints(hand_id)):
    name = p.getJointInfo(hand_id, i)[12].decode()
    print(f"link {i}: {name}")
# Suppose the tip link is index 9:
TIP_LINK_IDX = 9
sensor.add_camera(obj_id=hand_id, link_ids=TIP_LINK_IDX)

# 6) Spawn a small red sphere
sphere_col = p.createCollisionShape(p.GEOM_SPHERE, radius=0.02)
sphere_vis = p.createVisualShape(p.GEOM_SPHERE, radius=0.02,
                                 rgbaColor=[1, 0, 0, 1])
sphere_id = p.createMultiBody(
    baseMass=0.01,
    baseCollisionShapeIndex=sphere_col,
    baseVisualShapeIndex=sphere_vis,
    basePosition=[0.05, 0, 0.25]
)

# 7) Main loop: step physics and render tactile images
while True:
    p.stepSimulation()
    colors, depths = sensor.render()
    sensor.updateGUI(colors, depths)
    time.sleep(1/240)
