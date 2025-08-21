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

# Position‐control gains & limits
PG_POS      = 0.15
VG_POS      = 1.0
MAX_FORCE   = 5.0

# Sinusoid parameters
AMP         = 0.5    # ±0.5 rad swing
FREQ        = 0.1    # 0.1 Hz → 10 s period

# Joint prefixes by finger (4 joints each)
FINGER_PREFIXES = {
    "ring":   ["joint_0.0",  "joint_1.0",  "joint_2.0",  "joint_3.0"],
    "middle": ["joint_4.0",  "joint_5.0",  "joint_6.0",  "joint_7.0"],
    "index":  ["joint_8.0",  "joint_9.0",  "joint_10.0", "joint_11.0"],
    "little": ["joint_12.0", "joint_13.0", "joint_14.0", "joint_15.0"],
}

# “Upright” target angles for each finger (you can tweak these)
UPRIGHT_ANGLES = {
    "ring":   [0.8, 0.6, 0.4, 0.2],
    "middle": [0.8, 0.6, 0.4, 0.2],
    "index":  [0.8, 0.6, 0.4, 0.2],
    "little": [0.8, 0.6, 0.4, 0.2],
}

def main():
    # ——— 1) Setup PyBullet ———————————————————————————————————————
    p.connect(p.GUI)
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    p.setGravity(0, 0, 0)
    p.setRealTimeSimulation(0)
    p.loadURDF(PLANE_URDF, [0, 0, 0], useFixedBase=True)

    # ——— 2) Load hand & discover joints —————————————————————————
    hand = p.loadURDF(URDF_HAND, [0, 0, 0.25], useFixedBase=True)
    num_j = p.getNumJoints(hand)
    joint_info = {p.getJointInfo(hand, j)[1].decode(): j
                  for j in range(num_j)}
    
    # ——— 3) Build per‐finger joint lists in order ————————————————————
    finger_joints = {}
    for finger, prefixes in FINGER_PREFIXES.items():
        inds = [joint_info[pref] for pref in prefixes]
        finger_joints[finger] = inds

    # ——— 4) Disable all non‐actuated joints ———————————————————————
    actuated = sum(finger_joints.values(), [])
    for j in range(num_j):
        if j not in actuated:
            p.setJointMotorControl2(hand, j,
                                   p.VELOCITY_CONTROL,
                                   force=0)

    # ——— 5) Move each finger to its “upright” pose ——————————————————
    # Flatten actuated & upright lists in same order
    all_joints   = []
    upright_tgts = []
    for finger in ["ring","middle","index","little"]:
        all_joints   += finger_joints[finger]
        upright_tgts += UPRIGHT_ANGLES[finger]

    p.setJointMotorControlArray(
        bodyUniqueId=hand,
        jointIndices=all_joints,
        controlMode=p.POSITION_CONTROL,
        targetPositions=upright_tgts,
        positionGains=[PG_POS]*len(all_joints),
        velocityGains=[VG_POS]*len(all_joints),
        forces=[MAX_FORCE]*len(all_joints)
    )
    # step 1 second for settle
    for _ in range(SIM_FPS):
        p.stepSimulation()
        time.sleep(DT)

    # ——— 6) Record “home” angles q0 for each joint ———————————————
    q0s = [p.getJointState(hand, j)[0] for j in all_joints]

    # ——— 7) Main loop: sinusoidal POSITION_CONTROL on all fingers ————
    t0 = time.time()
    while True:
        t = time.time() - t0
        sine = AMP * np.sin(2*np.pi*FREQ * t)
        # new targets = q0 + sine
        tgts = [q0 + sine for q0 in q0s]

        p.setJointMotorControlArray(
            bodyUniqueId=hand,
            jointIndices=all_joints,
            controlMode=p.POSITION_CONTROL,
            targetPositions=tgts,
            positionGains=[PG_POS]*len(all_joints),
            velocityGains=[VG_POS]*len(all_joints),
            forces=[MAX_FORCE]*len(all_joints)
        )

        p.stepSimulation()
        time.sleep(DT)

if __name__ == "__main__":
    main()
