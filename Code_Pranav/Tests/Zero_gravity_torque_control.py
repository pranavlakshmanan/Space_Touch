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

# PD gains & torque limit
P_GAIN      = 20.0    # proportional gain
D_GAIN      = 2.0     # derivative gain
MAX_TAU     = 5.0     # clamp torque to ±5 N·m

# Sine‐wave parameters (for demonstration)
AMP         = 0.5     # ±0.5 rad swing
FREQ        = 0.1     # 0.1 Hz → 10 s period

def main():
    # 1) Connect & world setup in zero‐g
    p.connect(p.GUI)
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    p.setGravity(0, 0, 0)              # ← zero gravity
    p.loadURDF(PLANE_URDF, [0, 0, 0], useFixedBase=True)

    # 2) Load the hand
    hand = p.loadURDF(URDF_HAND, [0, 0, 0.25], useFixedBase=True)

    # 3) Identify all revolute joints to torque‐control
    joint_indices = []
    for j in range(p.getNumJoints(hand)):
        info = p.getJointInfo(hand, j)
        if info[2] == p.JOINT_REVOLUTE:
            joint_indices.append(j)

    # 4) (Optional) Pre‐place objects / apply preload here
    #    e.g. spawn spheres in contact or create soft‐spring constraints.

    # 5) Record each joint’s home position
    q0s = [p.getJointState(hand, j)[0] for j in joint_indices]

    # 6) Main PD‐torque loop
    t_start = time.time()
    while True:
        t = time.time() - t_start
        sine = AMP * np.sin(2 * np.pi * FREQ * t)

        # Read current positions & velocities
        qs, qds = [], []
        for j in joint_indices:
            q, qd = p.getJointState(hand, j)[:2]
            qs.append(q)
            qds.append(qd)

        # Compute torques for each joint
        taus = []
        for q0, q, qd in zip(q0s, qs, qds):
            error = (q0 + sine) - q
            raw_tau = P_GAIN * error - D_GAIN * qd
            tau = float(np.clip(raw_tau, -MAX_TAU, +MAX_TAU))
            taus.append(tau)

        # Apply torques
        p.setJointMotorControlArray(
            bodyUniqueId=hand,
            jointIndices=joint_indices,
            controlMode=p.TORQUE_CONTROL,
            forces=taus
        )

        # Step and wait
        p.stepSimulation()
        time.sleep(DT)

if __name__ == "__main__":
    main()
