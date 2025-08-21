#!/usr/bin/env python3
import time
import numpy as np
import cv2
import pybullet as p
import pybullet_data
import pybulletX as px
import tacto

#  Final Complete hand movement with sensors code - Base using velocity control

# ─── PARAMETERS ────────────────────────────────────────────────────────────
URDF_HAND    = "/home/pranav/Space_touch/examples/allegro_hand_description/allegro_hand_description_left_digit.urdf"
URDF_SPHERE  = "/home/pranav/Space_touch/examples/objects/sphere_small.urdf"
P_GAIN       = 50.0
D_GAIN       = 2.0
AMP_J        = 1.0      # joint swing amplitude (rad)
FREQ_J       = 0.1      # joint swing freq (Hz)
SIM_FREQ     = 240
DROP_TIME    = 2.0      # seconds for spheres to settle
THRESHOLD    = 0.1      # mask threshold

# Sinusoid for base translation
BASE_AMP_T   = 0.2      # ±0.2 m
FREQ_T       = 0.05     # 0.05 Hz
OMEGA_T      = 2 * np.pi * FREQ_T

# Sinusoid for base rotation
BASE_AMP_R   = 0.3      # ±0.3 rad
FREQ_R       = 0.07     # 0.07 Hz
OMEGA_R      = 2 * np.pi * FREQ_R

# Base velocity control gains
BASE_P_GAIN  = 2.0      # Position gain for base control (reduced for smoother motion)
BASE_D_GAIN  = 0.5      # Damping gain for base control (reduced for smoother motion)

# fingertip labels for sensor cameras
TIP_LABELS = {
    "joint_15.0_tip": "little",
    "joint_11.0_tip": "index",
    "joint_7.0_tip":  "middle",
    "joint_3.0_tip":  "ring",
}

def main():
    # 1) Initialize PyBulletX (opens the GUI once)
    px.init()

    # 2) Standard PyBullet setup
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    p.setGravity(0, 0, 0)
    p.setRealTimeSimulation(0)
    p.loadURDF("plane.urdf", [0, 0, 0], useFixedBase=True)
    p.resetDebugVisualizerCamera(0.6, 60, -30, [0, 0, 0.25])

    # 3) Initialize Tacto
    bg = cv2.imread("examples/conf/bg_digit_240_320.jpg")
    sensor = tacto.Sensor(120, 160,
                         background=bg,
                         config_path=tacto.get_digit_config_path())

    # 4) Load hand with a dynamic base
    hand = px.Body(urdf_path=URDF_HAND,
                   base_position=[0, 0, 0.25],
                   base_orientation=[0, 0, 0, 1],
                   use_fixed_base=False)
    base_pos0, base_ori0 = p.getBasePositionAndOrientation(hand.id)
    base_eul0 = p.getEulerFromQuaternion(base_ori0)

    # 5) Collect actuated joints
    joint_inds, joint_names = [], []
    for i in range(hand.num_joints):
        info = hand.get_joint_info(i)
        if info.joint_type in (p.JOINT_REVOLUTE, p.JOINT_PRISMATIC):
            joint_inds.append(i)
            joint_names.append(info.joint_name.decode())
    print("Actuated joints:", joint_names)

    # 6) Attach Tacto cameras & spawn spheres at each tip
    tip_entries = []
    for i in range(hand.num_joints):
        jn = hand.get_joint_info(i).joint_name.decode()
        if jn.endswith("_tip") and jn in TIP_LABELS:
            sensor.add_camera(hand.id, link_ids=[i])
            pos, _ = hand.get_link_state(i)[:2]
            tip_entries.append({
                "joint":    jn,
                "link_idx": i,
                "x":        pos[0],
                "cam_idx":  len(tip_entries)
            })
            sph = px.Body(urdf_path=URDF_SPHERE,
                          base_position=[pos[0], pos[1], pos[2]+0.05],
                          global_scaling=0.15,
                          use_fixed_base=False)
            sensor.add_body(sph)

    # 7) Let spheres settle
    t0 = time.time()
    while time.time() - t0 < DROP_TIME:
        p.stepSimulation()
        time.sleep(1.0/SIM_FREQ)

    # 8) Sort by x → determine camera order & labels
    tip_entries.sort(key=lambda e: e["x"])
    cam_order = [e["cam_idx"] for e in tip_entries]
    labels    = [TIP_LABELS[e["joint"]] for e in tip_entries]

    # 9) Prepare OpenCV windows
    title_c = "Tacto Color (L→R: " + " | ".join(labels) + ")"
    title_d = "Tacto Depth+Mask"
    cv2.namedWindow(title_c, cv2.WINDOW_NORMAL)
    cv2.namedWindow(title_d, cv2.WINDOW_NORMAL)

    # 10) Main control + render loop
    t_start = time.time()
    phases  = np.linspace(0, 2*np.pi, len(joint_inds), endpoint=False)
    while True:
        t = time.time() - t_start

        # a) PD torque-control on joints
        targets = AMP_J * np.sin(2*np.pi*FREQ_J*t + phases)
        q   = np.array([p.getJointState(hand.id, j)[0] for j in joint_inds])
        qd  = np.array([p.getJointState(hand.id, j)[1] for j in joint_inds])
        tau = P_GAIN*(targets - q) - D_GAIN*qd
        p.setJointMotorControlArray(
            bodyUniqueId=hand.id,
            jointIndices=joint_inds,
            controlMode=p.TORQUE_CONTROL,
            forces=tau.tolist()
        )

        # b) Base velocity control for translation and rotation
        # Calculate desired velocities directly from sinusoidal derivatives
        desired_vel_x = BASE_AMP_T * OMEGA_T * np.cos(OMEGA_T * t)
        desired_lin_vel = [desired_vel_x, 0, 0]
        
        desired_ang_vel_roll  = BASE_AMP_R * OMEGA_R * np.cos(OMEGA_R * t)
        desired_ang_vel_pitch = BASE_AMP_R * OMEGA_R * np.cos(OMEGA_R * t + 2.0)
        desired_ang_vel_yaw   = BASE_AMP_R * OMEGA_R * np.cos(OMEGA_R * t + 4.0)
        desired_ang_vel = [desired_ang_vel_roll, desired_ang_vel_pitch, desired_ang_vel_yaw]
        
        # Get current velocities for damping
        current_vel, current_ang_vel = p.getBaseVelocity(hand.id)
        
        # Apply simple velocity control with damping
        vel_error = np.array(desired_lin_vel) - np.array(current_vel)
        ang_vel_error = np.array(desired_ang_vel) - np.array(current_ang_vel)
        
        # Scale down the velocity commands for smoother motion
        target_lin_vel = desired_lin_vel + BASE_D_GAIN * vel_error
        target_ang_vel = desired_ang_vel + BASE_D_GAIN * ang_vel_error
        
        # Limit maximum velocities to prevent glitching
        max_lin_vel = 0.5
        max_ang_vel = 1.0
        target_lin_vel = np.clip(target_lin_vel, -max_lin_vel, max_lin_vel)
        target_ang_vel = np.clip(target_ang_vel, -max_ang_vel, max_ang_vel)
        
        # Apply velocity control to base
        p.resetBaseVelocity(hand.id, 
                           linearVelocity=target_lin_vel.tolist(),
                           angularVelocity=target_ang_vel.tolist())

        p.stepSimulation()

        # c) Tacto render
        colors, depths = sensor.render()

        # d) Color mosaic + labels
        ordered_c = [colors[i] for i in cam_order]
        mosaic_c  = np.concatenate(ordered_c, axis=1)
        w = mosaic_c.shape[1] // len(labels)
        for idx, lab in enumerate(labels):
            cv2.putText(mosaic_c, lab, (idx*w+5, 15),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)
        cv2.imshow(title_c, mosaic_c)

        # e) Depth+mask mosaic + labels
        spans = []
        mask_mosaic = None
        for d in [depths[i] for i in cam_order]:
            norm = (d - d.min()) / (d.max() - d.min() + 1e-6)
            spans.append(norm.max() - norm.min())
            mask = (norm < THRESHOLD).astype(np.uint8)*255
            mask_mosaic = mask if mask_mosaic is None else np.concatenate([mask_mosaic, mask], axis=1)
        for idx, lab in enumerate(labels):
            cv2.putText(mask_mosaic, lab, (idx*w+5, 15),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 1)
        cv2.imshow(title_d, mask_mosaic)

        # f) Print deformation spans
        print("Depth spans:", ["%.3f"%s for s in spans])

        # g) Exit on Esc
        if cv2.waitKey(1) == 27:
            break
        time.sleep(1.0/SIM_FREQ)

    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
