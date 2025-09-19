#!/usr/bin/env python3
import time, numpy as np, cv2
import pybullet as p, pybullet_data
import pybulletX as px
import tacto

# ─── PARAMETERS ────────────────────────────────────────────────────────────
URDF_HAND   = "/home/pralak/Space_touch/examples/allegro_hand_description/allegro_hand_description_left_digit.urdf"
URDF_SPHERE = "/home/pralak/Space_touch/examples/objects/sphere_small.urdf"
P_GAIN      = 50.0
D_GAIN      = 2.0
AMP         = 1.0
FREQ        = 0.1
SIM_FREQ    = 240
DROP_TIME   = 2.0     # let spheres settle under gravity
THRESHOLD   = 0.1     # for mask

# Map joint_tip → human finger name
TIP_LABELS = {
    "joint_15.0_tip": "little",   # pinky
    "joint_11.0_tip": "index",
    "joint_7.0_tip":  "middle",
    "joint_3.0_tip":  "ring",
}

def main():
    # 1) Tacto & PyBullet (gravity on)
    bg = cv2.imread("examples/conf/bg_digit_240_320.jpg")
    sensor = tacto.Sensor(120,160, background=bg,
                         config_path=tacto.get_digit_config_path())
    px.init(mode=p.GUI); p.setRealTimeSimulation(0)
    p.setGravity(0,0,-9.81)
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    p.loadURDF("plane.urdf",[0,0,0],useFixedBase=True)
    p.resetDebugVisualizerCamera(0.3,60,-30,[0,0,0.02])

    # 2) Load hand & collect actuated joints
    hand = px.Body(
        urdf_path=URDF_HAND,
        base_position=[0,0,0.25],
        base_orientation=[0,0,0,1],
        use_fixed_base=True
    )
    joint_inds, joint_names = [], []
    for i in range(hand.num_joints):
        info = hand.get_joint_info(i)
        if info.joint_type in (p.JOINT_REVOLUTE, p.JOINT_PRISMATIC):
            joint_inds.append(i)
            joint_names.append(info.joint_name.decode())
    print("Actuated joints:", joint_names)
    print("Joint indices:", joint_inds)

    # 3) Attach cameras & record each tip’s x-position
    tip_entries = []
    for i in range(hand.num_joints):
        jn = hand.get_joint_info(i).joint_name.decode()
        if jn.endswith("_tip") and jn in TIP_LABELS:
            sensor.add_camera(hand.id, link_ids=[i])
            pos,_ = hand.get_link_state(i)[:2]
            tip_entries.append({
                "joint": jn,
                "link_idx": i,
                "x": pos[0],
                "cam_idx": len(tip_entries)
            })
            # spawn sphere
            sph = px.Body(urdf_path=URDF_SPHERE,
                          base_position=[pos[0],pos[1],pos[2]+0.05],
                          global_scaling=0.15,use_fixed_base=False)
            sensor.add_body(sph)

    # 4) Let spheres settle
    t_start = time.time()
    while time.time() - t_start < DROP_TIME:
        p.stepSimulation()
        time.sleep(1.0/SIM_FREQ)

    # 5) Sort tips left→right by x, build order + labels
    tip_entries.sort(key=lambda e: e["x"])
    cam_order = [e["cam_idx"] for e in tip_entries]
    labels    = [TIP_LABELS[e["joint"]] for e in tip_entries]

    # 6) Prepare windows
    title_c = "Tacto Color (L→R: " + " | ".join(labels) + ")"
    title_d = "Tacto Depth+Mask"
    cv2.namedWindow(title_c, cv2.WINDOW_NORMAL)
    cv2.namedWindow(title_d, cv2.WINDOW_NORMAL)

    # 7) Control + render loop
    t0 = time.time()
    while True:
        t = time.time() - t0

        # a) PD torque
        phases  = np.linspace(0,2*np.pi,len(joint_inds),endpoint=False)
        targets = AMP * np.sin(2*np.pi*FREQ*t + phases)
        q   = np.array([p.getJointState(hand.id,j)[0] for j in joint_inds])
        qd  = np.array([p.getJointState(hand.id,j)[1] for j in joint_inds])
        tau = P_GAIN*(targets - q) - D_GAIN*qd
        p.setJointMotorControlArray(
            bodyUniqueId=hand.id,
            jointIndices=joint_inds,
            controlMode=p.TORQUE_CONTROL,
            forces=tau.tolist()
        )
        p.stepSimulation()

        # b) render
        colors, depths = sensor.render()

        # c) color mosaic + labels
        ordered_c = [colors[i] for i in cam_order]
        mosaic_c  = np.concatenate(ordered_c, axis=1)
        w = mosaic_c.shape[1]//len(labels)
        for idx, lab in enumerate(labels):
            cv2.putText(mosaic_c, lab, (idx*w+5,15),
                        cv2.FONT_HERSHEY_SIMPLEX,0.5,(255,255,255),1)
        cv2.imshow(title_c, mosaic_c)

        # d) depth+mask mosaic + labels
        ordered_d = [depths[i] for i in cam_order]
        all_d = np.concatenate([d.flatten() for d in ordered_d])
        mn, mx = all_d.min(), all_d.max()
        spans = []
        mask_mosaic = None
        for d in ordered_d:
            norm = (d - mn)/(mx-mn+1e-6)
            spans.append(norm.max() - norm.min())
            mask = (norm < THRESHOLD).astype(np.uint8)*255
            mask_mosaic = mask if mask_mosaic is None else np.concatenate([mask_mosaic, mask], axis=1)
        # label
        for idx, lab in enumerate(labels):
            cv2.putText(mask_mosaic, lab, (idx*w+5,15),
                        cv2.FONT_HERSHEY_SIMPLEX,0.5,(0,0,0),1)
        cv2.imshow(title_d, mask_mosaic)

        # e) print deformation spans
        print("Depth spans:", ["%.3f"%s for s in spans])

        if cv2.waitKey(1)==27:
            break
        time.sleep(1.0/SIM_FREQ)

if __name__=="__main__":
    main()
