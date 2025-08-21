#!/usr/bin/env python3
import time
import numpy as np
import pybullet as p
import pybullet_data

# ─── CONFIG ────────────────────────────────────────────────────────────────
URDF_HAND   = "/home/pranav/Space_touch/examples/allegro_hand_description/allegro_hand_description_left_digit.urdf"
PLANE_URDF  = "plane.urdf"

SIM_FPS     = 240
DT          = 1.0 / SIM_FPS

# Max torque/force for velocity control
MAX_FORCE   = 5.0

# Sinusoid parameters
AMP         = 0.5    # ±0.5 rad swing
FREQ        = 0.1    # 0.1 Hz → 10 s period
OMEGA       = 2 * np.pi * FREQ  # angular frequency

# Joint prefixes by finger (4 joints each)
FINGER_PREFIXES = {
    "ring":   ["joint_0.0",  "joint_1.0",  "joint_2.0",  "joint_3.0"],
    "middle": ["joint_4.0",  "joint_5.0",  "joint_6.0",  "joint_7.0"],
    "index":  ["joint_8.0",  "joint_9.0",  "joint_10.0", "joint_11.0"],
    "little": ["joint_12.0", "joint_13.0", "joint_14.0", "joint_15.0"],
}

# “Upright” target angles for each finger (to initialize pose)
UPRIGHT_ANGLES = {
    "ring":   [0.8, 0.6, 0.4, 0.2],
    "middle": [0.8, 0.6, 0.4, 0.2],
    "index":  [0.8, 0.6, 0.4, 0.2],
    "little": [0.8, 0.6, 0.4, 0.2],
}

def main():
    # 1) PyBullet init
    p.connect(p.GUI)
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    p.setGravity(0, 0, 0)             # zero-G
    p.setRealTimeSimulation(0)
    p.loadURDF(PLANE_URDF, [0,0,0], useFixedBase=True)

    # 2) Load hand
    hand = p.loadURDF(URDF_HAND, [0,0,0.25], useFixedBase=True)
    num_j = p.getNumJoints(hand)
    info  = {p.getJointInfo(hand,j)[1].decode(): j for j in range(num_j)}

    # 3) Finger joint lists
    finger_joints = {
        f: [info[pref] for pref in prefs]
        for f, prefs in FINGER_PREFIXES.items()
    }

    # 4) Disable non-actuated
    actuated = sum(finger_joints.values(), [])
    for j in range(num_j):
        if j not in actuated:
            p.setJointMotorControl2(hand, j, p.VELOCITY_CONTROL, force=0)

    # 5) Move to “upright” via position control then record home
    all_joints, upright_tgts = [], []
    for f in ["ring","middle","index","little"]:
        all_joints   += finger_joints[f]
        upright_tgts += UPRIGHT_ANGLES[f]

    p.setJointMotorControlArray(
        bodyUniqueId=hand,
        jointIndices=all_joints,
        controlMode=p.POSITION_CONTROL,
        targetPositions=upright_tgts,
        forces=[MAX_FORCE]*len(all_joints)
    )
    for _ in range(SIM_FPS):
        p.stepSimulation()
        time.sleep(DT)

    # (home positions aren’t needed for velocity mode, so we skip q0)

    # 6) Main loop: sinusoidal VELOCITY_CONTROL
    t0 = time.time()
    while True:
        t = time.time() - t0
        # compute velocity command
        vel_cmd = AMP * OMEGA * np.cos(OMEGA * t)

        # every joint gets same sinusoidal velocity
        vels = [vel_cmd] * len(all_joints)

        p.setJointMotorControlArray(
            bodyUniqueId=hand,
            jointIndices=all_joints,
            controlMode=p.VELOCITY_CONTROL,
            targetVelocities=vels,
            forces=[MAX_FORCE]*len(all_joints)
        )

        p.stepSimulation()
        time.sleep(DT)

if __name__ == "__main__":
    main()
