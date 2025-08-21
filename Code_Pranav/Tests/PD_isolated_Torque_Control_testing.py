#!/usr/bin/env python3
import time
import numpy as np
import pybullet as p
import pybullet_data

# ─── CONFIG ────────────────────────────────────────────────────────────────
URDF_HAND    = "/home/pranav/Space_touch/examples/allegro_hand_description/allegro_hand_description_left_digit.urdf"
PLANE_URDF   = "plane.urdf"
P_GAIN       = 20.0           # so P_GAIN * error ≤ MAX_TAU
D_GAIN       = 5.0
AMP          = 1.0           # ±1 rad swing
FREQ         = 0.1           # 0.1 Hz → 10 s per cycle
MAX_TAU      = 5.0           # clamp torque to ±5 N·m
SIM_FPS      = 240
DT           = 1.0 / SIM_FPS

# Joint name prefixes for the little (pinky) finger
LITTLE_PREFIXES = ["joint_12.0", "joint_13.0", "joint_14.0", "joint_15.0"]

def main():
    # 1) Connect & scene
    p.connect(p.GUI)
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    p.setGravity(0, 0, -9.81)
    p.setRealTimeSimulation(1)  # let PyBullet run in real time
    p.loadURDF(PLANE_URDF, [0, 0, 0], useFixedBase=True)

    # 2) Load the Allegro hand
    hand = p.loadURDF(URDF_HAND, [0, 0, 0.25], useFixedBase=True)

    # 3) Identify the little‐finger joints
    test_joints = []
    for j in range(p.getNumJoints(hand)):
        info = p.getJointInfo(hand, j)
        name = info[1].decode()
        jtype = info[2]
        if jtype == p.JOINT_REVOLUTE and any(name.startswith(pref) for pref in LITTLE_PREFIXES):
            test_joints.append(j)
    print("Testing little-finger joints:", test_joints)

    # 4) Disable all other joints
    for j in range(p.getNumJoints(hand)):
        if j not in test_joints:
            p.setJointMotorControl2(
                bodyIndex=hand,
                jointIndex=j,
                controlMode=p.VELOCITY_CONTROL,
                force=0
            )

    # 5) Record each joint’s initial position
    q0s = [p.getJointState(hand, j)[0] for j in test_joints]

    # 6) Real-time PD torque loop
    t0 = time.time()
    while True:
        t = time.time() - t0
        # sine wave around each joint's home
        sine = AMP * np.sin(2 * np.pi * FREQ * t)
        targets = [q0 + sine for q0 in q0s]

        # read current state
        qs, qds = [], []
        for j in test_joints:
            q, qd = p.getJointState(hand, j)[:2]
            qs.append(q)
            qds.append(qd)

        # compute and clamp torques
        taus = []
        for target, q, qd in zip(targets, qs, qds):
            raw_tau = P_GAIN * (target - q) - D_GAIN * qd
            tau = float(np.clip(raw_tau, -MAX_TAU, MAX_TAU))
            taus.append(tau)

        # apply torques to all little-finger joints
        p.setJointMotorControlArray(
            bodyIndex=hand,
            jointIndices=test_joints,
            controlMode=p.TORQUE_CONTROL,
            forces=taus
        )

        # optional: debug print
        # print("q:", ["%.2f"%q for q in qs], "tau:", ["%.2f"%t for t in taus])

        # sleep to let real-time sim run
        time.sleep(DT)

if __name__ == "__main__":
    main()
