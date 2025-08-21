#!/usr/bin/env python3
import time
import numpy as np
import cv2
import pybullet as p
import pybullet_data
import pybulletX as px
import tacto

# Tendon-based hand movement with velocity control

# ─── PARAMETERS ────────────────────────────────────────────────────────────
URDF_HAND    = "/home/pranav/Space_touch/examples/allegro_hand_description/allegro_hand_description_left_digit.urdf"
URDF_SPHERE  = "/home/pranav/Space_touch/examples/objects/sphere_small.urdf"

# Tendon control parameters
TENDON_GAIN = 2.0      # Base gain for tendon force
ALPHA = 1.0            # Linear interpolation factor (1.0 = full range)
MIN_ANGLE = -1.0       # Minimum joint angle (rad)
MAX_ANGLE = 1.0        # Maximum joint angle (rad)
MAX_VELOCITY = 2.0     # Maximum joint velocity (rad/s)

# Sinusoidal motion parameters for finger actuation
AMP_TENDON = 1.0       # Tendon pull amplitude
FREQ_TENDON = 0.1      # Tendon pull frequency (Hz)

SIM_FREQ = 240
DROP_TIME = 2.0        # seconds for spheres to settle
THRESHOLD = 0.1        # mask threshold

# Base movement parameters (unchanged)
BASE_AMP_T = 0.2       # ±0.2 m
FREQ_T = 0.05          # 0.05 Hz
OMEGA_T = 2 * np.pi * FREQ_T

BASE_AMP_R = 0.3       # ±0.3 rad
FREQ_R = 0.07          # 0.07 Hz
OMEGA_R = 2 * np.pi * FREQ_R

# Fingertip labels for sensor cameras
TIP_LABELS = {
    "joint_15.0_tip": "little",
    "joint_11.0_tip": "index",
    "joint_7.0_tip":  "middle",
    "joint_3.0_tip":  "ring",
}

# Define finger chains (base to tip joint indices)
FINGER_CHAINS = {
    "index": ["joint_8.0", "joint_9.0", "joint_10.0", "joint_11.0"],
    "middle": ["joint_4.0", "joint_5.0", "joint_6.0", "joint_7.0"],
    "ring": ["joint_0.0", "joint_1.0", "joint_2.0", "joint_3.0"],
    "thumb": ["joint_12.0", "joint_13.0", "joint_14.0", "joint_15.0"]
}

class TendonController:
    """Implements tendon-based control with distance-dependent force transmission"""
    
    def __init__(self, hand_id, joint_names, joint_indices):
        self.hand_id = hand_id
        self.joint_names = joint_names
        self.joint_indices = joint_indices
        
        # Create mapping from joint names to indices
        self.name_to_idx = {name: idx for name, idx in zip(joint_names, joint_indices)}
        
        # Organize joints by finger
        self.finger_joints = {}
        for finger, chain in FINGER_CHAINS.items():
            self.finger_joints[finger] = []
            for joint_name in chain:
                if joint_name in self.name_to_idx:
                    self.finger_joints[finger].append(self.name_to_idx[joint_name])
    
    def compute_tendon_velocities(self, tendon_pulls):
        """
        Compute velocity commands for each joint based on tendon pull.
        
        Args:
            tendon_pulls: dict of finger -> pull strength (0 to 1)
        
        Returns:
            velocities: array of velocity commands for all joints
        """
        velocities = np.zeros(len(self.joint_indices))
        
        for finger, pull in tendon_pulls.items():
            if finger not in self.finger_joints:
                continue
                
            joints = self.finger_joints[finger]
            n_joints = len(joints)
            
            for i, joint_idx in enumerate(joints):
                # Distance factor: joints farther from base get less force
                # This mimics how tendons lose force along the finger
                distance_factor = 1.0 - (i / n_joints) * 0.5  # Ranges from 1.0 to 0.5
                
                # Get current joint state
                idx_in_list = self.joint_indices.index(joint_idx)
                current_pos = p.getJointState(self.hand_id, joint_idx)[0]
                
                # Linear interpolation for target position
                target_pos = MIN_ANGLE + ALPHA * pull * (MAX_ANGLE - MIN_ANGLE)
                
                # Position error
                error = target_pos - current_pos
                
                # Velocity command proportional to error and distance factor
                # This creates natural compliance: distal joints move less when proximal joints meet resistance
                velocity = TENDON_GAIN * distance_factor * error
                
                # Clamp velocity to maximum
                velocity = np.clip(velocity, -MAX_VELOCITY, MAX_VELOCITY)
                
                velocities[idx_in_list] = velocity
        
        return velocities

def main():
    # 1) Initialize PyBulletX (opens the GUI once)
    px.init()

    # 2) Standard PyBullet setup
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    p.setGravity(0, 0, 0)  # Zero gravity for space
    p.setRealTimeSimulation(0)
    
    # Remove the default plane - we don't need it for space
    # p.loadURDF("plane.urdf", [0, 0, 0], useFixedBase=True)
    
    # Configure visualization for space-like appearance
    p.resetDebugVisualizerCamera(0.6, 60, -30, [0, 0, 0.25])
    
    # Configure GUI for dark space environment
    p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0)  # Hide GUI panels
    p.configureDebugVisualizer(p.COV_ENABLE_SHADOWS, 1)  # Enable shadows
    p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 1)
    
    # Set dark background color (near black for space)
    p.configureDebugVisualizer(lightPosition=[0.5, 0.5, 1.0])
    
    # Create dramatic directional lighting
    # Main key light from above-right
    p.configureDebugVisualizer(lightPosition=[1.0, 0.5, 2.0])
    
    # Adjust ambient and diffuse lighting for space effect
    # This creates strong directional shadows with dark areas
    p.changeVisualShape(objectUniqueId=-1, linkIndex=-1,
                       rgbaColor=[0.05, 0.05, 0.1, 1.0],  # Very dark blue-black for space
                       specularColor=[0.8, 0.8, 0.9])

    # 3) Create stars in the background (optional - adds visual interest)
    # Add some distant sphere "stars" for space effect
    import random
    for _ in range(50):
        star_pos = [
            random.uniform(-5, 5),
            random.uniform(-5, 5),
            random.uniform(-2, 3)
        ]
        star = p.createVisualShape(
            shapeType=p.GEOM_SPHERE,
            radius=0.01,
            rgbaColor=[1, 1, 1, 1],  # White stars
            specularColor=[1, 1, 1]
        )
        p.createMultiBody(
            baseVisualShapeIndex=star,
            basePosition=star_pos,
            useMaximalCoordinates=False
        )
    
    # Initialize Tacto
    bg = cv2.imread("examples/conf/bg_digit_240_320.jpg")
    sensor = tacto.Sensor(120, 160,
                         background=bg,
                         config_path=tacto.get_digit_config_path())

    # 4) Load hand with a dynamic base
    hand = px.Body(urdf_path=URDF_HAND,
                   base_position=[0, 0, 0.25],
                   base_orientation=[0, 0, 0, 1],
                   use_fixed_base=False)
    
    # Make hand materials more metallic/robotic for space
    for i in range(hand.num_joints + 1):  # Include base link
        p.changeVisualShape(hand.id, i - 1,
                          rgbaColor=[0.7, 0.75, 0.8, 1.0],  # Metallic silver-grey
                          specularColor=[0.9, 0.9, 0.95])
    
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

    # 6) Initialize tendon controller
    tendon_controller = TendonController(hand.id, joint_names, joint_inds)

    # 7) Attach Tacto cameras & spawn spheres at each tip
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
            
            # Make spheres glow with different colors for space effect
            colors = [
                [1.0, 0.3, 0.3, 1.0],  # Red
                [0.3, 1.0, 0.3, 1.0],  # Green  
                [0.3, 0.3, 1.0, 1.0],  # Blue
                [1.0, 1.0, 0.3, 1.0],  # Yellow
            ]
            sphere_color = colors[len(tip_entries) % len(colors)]
            p.changeVisualShape(sph.id, -1,
                              rgbaColor=sphere_color,
                              specularColor=[1.0, 1.0, 1.0])
            
            sensor.add_body(sph)

    # 8) Let spheres settle
    t0 = time.time()
    while time.time() - t0 < DROP_TIME:
        p.stepSimulation()
        time.sleep(1.0/SIM_FREQ)

    # 9) Sort by x → determine camera order & labels
    tip_entries.sort(key=lambda e: e["x"])
    cam_order = [e["cam_idx"] for e in tip_entries]
    labels    = [TIP_LABELS[e["joint"]] for e in tip_entries]

    # 10) Prepare OpenCV windows
    title_c = "Tacto Color (L→R: " + " | ".join(labels) + ")"
    title_d = "Tacto Depth+Mask"
    cv2.namedWindow(title_c, cv2.WINDOW_NORMAL)
    cv2.namedWindow(title_d, cv2.WINDOW_NORMAL)

    # 11) Main control + render loop
    t_start = time.time()
    # Phase offsets for different fingers to create varied motion
    finger_phases = {
        "index": 0,
        "middle": np.pi/3,
        "ring": 2*np.pi/3,
        "thumb": np.pi
    }
    
    while True:
        t = time.time() - t_start

        # a) Compute tendon pulls for each finger (sinusoidal pattern)
        tendon_pulls = {}
        for finger, phase in finger_phases.items():
            # Generate pull strength between 0 and 1
            pull = 0.5 + 0.5 * AMP_TENDON * np.sin(2*np.pi*FREQ_TENDON*t + phase)
            tendon_pulls[finger] = np.clip(pull, 0, 1)
        
        # b) Get velocity commands from tendon controller
        velocities = tendon_controller.compute_tendon_velocities(tendon_pulls)
        
        # c) Apply velocity control to joints
        p.setJointMotorControlArray(
            bodyUniqueId=hand.id,
            jointIndices=joint_inds,
            controlMode=p.VELOCITY_CONTROL,
            targetVelocities=velocities.tolist(),
            forces=[5.0] * len(joint_inds)  # Max force for each joint
        )

        # d) Base translate + rotate (unchanged from original)
        disp_x = BASE_AMP_T * np.sin(OMEGA_T * t)
        new_pos = [base_pos0[0] + disp_x,
                   base_pos0[1],
                   base_pos0[2]]
        roll  = base_eul0[0] + BASE_AMP_R * np.sin(OMEGA_R * t)
        pitch = base_eul0[1] + BASE_AMP_R * np.sin(OMEGA_R * t + 2.0)
        yaw   = base_eul0[2] + BASE_AMP_R * np.sin(OMEGA_R * t + 4.0)
        new_ori = p.getQuaternionFromEuler([roll, pitch, yaw])
        p.resetBasePositionAndOrientation(hand.id, new_pos, new_ori)

        p.stepSimulation()

        # e) Tacto render
        colors, depths = sensor.render()

        # f) Color mosaic + labels
        ordered_c = [colors[i] for i in cam_order]
        mosaic_c  = np.concatenate(ordered_c, axis=1)
        w = mosaic_c.shape[1] // len(labels)
        for idx, lab in enumerate(labels):
            cv2.putText(mosaic_c, lab, (idx*w+5, 15),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)
        cv2.imshow(title_c, mosaic_c)

        # g) Depth+mask mosaic + labels
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

        # h) Print deformation spans and tendon pulls
        print("Depth spans:", ["%.3f"%s for s in spans])
        print("Tendon pulls:", {f: "%.2f"%p for f, p in tendon_pulls.items()})

        # i) Exit on Esc
        if cv2.waitKey(1) == 27:
            break
        time.sleep(1.0/SIM_FREQ)

    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()