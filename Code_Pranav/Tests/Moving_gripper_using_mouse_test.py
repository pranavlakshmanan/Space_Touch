#!/usr/bin/env python3
import time
import cv2
import pybullet as p
import pybulletX as px
import tacto

# ─── 1) Paths ───────────────────────────────────────────────────────
DIGIT_URDF   = "/home/pranav/Space_touch/examples/allegro_hand_description/allegro_hand_description_left_digit.urdf"
SPHERE_URDF  = "/home/pranav/Space_touch/examples/objects/sphere_small.urdf"
BG_DIGIT     = "examples/conf/bg_digit_240_320.jpg"

def main():
    # ─── 2) Set up Tacto + load Digit background ───────────────────────
    bg = cv2.imread(BG_DIGIT)
    sensor = tacto.Sensor(
        width=120,
        height=160,
        background=bg,
        config_path=tacto.get_digit_config_path()
    )

    # ─── 3) Start PyBullet GUI & real-time mode ────────────────────────
    px.init()                    # connect, set gravity, solver defaults
    p.setRealTimeSimulation(1)   # run at ~240 Hz automatically
    p.resetDebugVisualizerCamera(
        cameraDistance=0.30,
        cameraYaw=60,
        cameraPitch=-30,
        cameraTargetPosition=[0, 0, 0.02]
    )

    # ─── 4) Load the Digit-enabled Allegro URDF (no need to spawn additional gel) ──
    gripper = px.Body(
        urdf_path=DIGIT_URDF,
        base_position=[0, 0, 0.18],
        base_orientation=[0, 0, 0, 1],
        use_fixed_base=True
    )

    # ─── 5) Print out all joint→link so we see exactly what ends in "_tip" ─────────
    print("\n=== Digit-enabled URDF Joint → child link mapping ===")
    for i in range(gripper.num_joints):
        info = gripper.get_joint_info(i)
        jname = info.joint_name.decode()
        lname = info.link_name.decode()
        print(f" idx={i:2d}  joint='{jname:20s}' → child link='{lname:20s}'")
    print("────────────────────────────────────────────────────────\n")

    # ─── 6) Detect exactly those indices ending in "_tip" (same as plain URDF) ─────
    fingertip_indices = []
    for i in range(gripper.num_joints):
        jname = gripper.get_joint_info(i).joint_name.decode()
        if jname.endswith("_tip"):
            fingertip_indices.append(i)

    print("→ Found fingertip indices:", fingertip_indices)
    #  prints: [4, 9, 14, 19]

    # ─── 7) Attach a Tacto camera to each of those tip‐links (DON’T respawn a new gel) ─
    for idx in fingertip_indices:
        sensor.add_camera(gripper.id, link_ids=[idx])
    print(f"→ Attached {len(fingertip_indices)} Tacto cameras (one per fingertip)\n")

    # (Optionally print exactly which camera → which link)
    for cam_i, link_idx in enumerate(fingertip_indices):
        info = gripper.get_joint_info(link_idx)
        print(f"   cam{cam_i} → link '{info.link_name.decode()}'    (joint '{info.joint_name.decode()}')")



    sphere = px.Body(
        urdf_path=SPHERE_URDF,
        base_position=[-0.0, 0.0, 0.3],
        global_scaling=0.15,
        use_fixed_base=False,
           )
    sensor.add_body(sphere)

    # # ─── 8) Drop a sphere onto, say, the “first” fingertip (cam0) ────────────
    # if fingertip_indices:
    #     tip_idx = fingertip_indices[0]
    #     pos, _ = gripper.get_link_state(tip_idx)[:2]   # wrist in world coords
    #     sphere_start = (pos[0], pos[1], pos[2] + 0.03)
    #     sphere = px.Body(
    #         urdf_path=SPHERE_URDF,
    #         base_position=sphere_start,
    #         global_scaling=0.15,
    #         use_fixed_base=False
    #     )
    #     sensor.add_body(sphere)

    # # Step a bit so the sphere falls onto that fingertip
    # for _ in range(200):
    #     p.stepSimulation()


    # ─── 9) Render all camera streams (there will be 4 of them) ────────────
    colors, depths = sensor.render()
    sensor.updateGUI(colors, depths)
    cv2.waitKey(0)

if __name__ == "__main__":
    main()
