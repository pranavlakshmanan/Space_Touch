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

# Sinusoid parameters for fingers
AMP_F       = 0.5     # ±0.5 rad swing
FREQ_F      = 0.1     # 0.1 Hz → 10 s period
OMEGA_F     = 2 * np.pi * FREQ_F

# Sinusoid parameters for base translation
BASE_AMP_T  = 0.2     # ±0.2 m swing along X
FREQ_T      = 0.05    # 0.05 Hz → 20 s period
OMEGA_T     = 2 * np.pi * FREQ_T

# Sinusoid parameters for base rotation
BASE_AMP_R  = 0.3     # ±0.3 rad swing on each axis
FREQ_R      = 0.07    # 0.07 Hz → ~14.3 s period
OMEGA_R     = 2 * np.pi * FREQ_R

# Joint prefixes by finger (4 joints each)
FINGER_PREFIXES = {
    "ring":   ["joint_0.0",  "joint_1.0",  "joint_2.0",  "joint_3.0"],
    "middle": ["joint_4.0",  "joint_5.0",  "joint_6.0",  "joint_7.0"],
    "index":  ["joint_8.0",  "joint_9.0",  "joint_10.0", "joint_11.0"],
    "little": ["joint_12.0", "joint_13.0", "joint_14.0", "joint_15.0"],
}

def main():
    # 1) PyBullet init
    p.connect(p.GUI)
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    p.setGravity(0, 0, 0)
    p.setRealTimeSimulation(0)
    p.loadURDF(PLANE_URDF, [0,0,0], useFixedBase=True)

    # 2) Load hand with a dynamic base
    hand = p.loadURDF(URDF_HAND, [0,0,0.25], useFixedBase=False)
    base_pos0, base_ori0 = p.getBasePositionAndOrientation(hand)
    base_eul0 = p.getEulerFromQuaternion(base_ori0)

    # 3) Discover actuated joints
    num_j = p.getNumJoints(hand)
    info  = {p.getJointInfo(hand,j)[1].decode(): j for j in range(num_j)}
    finger_joints = {
        f: [info[pref] for pref in prefs]
        for f, prefs in FINGER_PREFIXES.items()
    }

    # 4) Disable non-actuated
    actuated = sum(finger_joints.values(), [])
    for j in range(num_j):
        if j not in actuated:
            p.setJointMotorControl2(hand, j, p.VELOCITY_CONTROL, force=0)

    # 5) Initialize pose via position control (optional)
    all_joints = []
    for f in ["ring","middle","index","little"]:
        all_joints += finger_joints[f]
    upright = [0.8,0.6,0.4,0.2] * 4  # 4 fingers
    p.setJointMotorControlArray(
        bodyUniqueId=hand,
        jointIndices=all_joints,
        controlMode=p.POSITION_CONTROL,
        targetPositions=upright,
        forces=[MAX_FORCE]*len(all_joints)
    )
    for _ in range(SIM_FPS):
        p.stepSimulation()
        time.sleep(DT)

    # 6) Main loop: velocity‐driven fingers + base translate & rotate
    t0 = time.time()
    while True:
        t = time.time() - t0

        # — finger velocities
        vel_cmd = AMP_F * OMEGA_F * np.cos(OMEGA_F * t)
        p.setJointMotorControlArray(
            bodyUniqueId=hand,
            jointIndices=all_joints,
            controlMode=p.VELOCITY_CONTROL,
            targetVelocities=[vel_cmd]*len(all_joints),
            forces=[MAX_FORCE]*len(all_joints)
        )

        # — base translation along X
        disp_x = BASE_AMP_T * np.sin(OMEGA_T * t)
        new_pos = [base_pos0[0] + disp_x,
                   base_pos0[1],
                   base_pos0[2]]

        # — base rotation: roll, pitch, yaw
        roll  = base_eul0[0] + BASE_AMP_R * np.sin(OMEGA_R * t)
        pitch = base_eul0[1] + BASE_AMP_R * np.sin(OMEGA_R * t + 2.0)
        yaw   = base_eul0[2] + BASE_AMP_R * np.sin(OMEGA_R * t + 4.0)
        new_ori = p.getQuaternionFromEuler([roll, pitch, yaw])

        # — apply base transform
        p.resetBasePositionAndOrientation(hand, new_pos, new_ori)

        p.stepSimulation()
        time.sleep(DT)

if __name__ == "__main__":
    main()
