#!/usr/bin/env python3
import time
import numpy as np
import cv2
import pybullet as p
import pybullet_data
import pybulletX as px
import tacto

# Tendon-based hand movement with velocity control and reference axis

# ─── PARAMETERS ────────────────────────────────────────────────────────────
URDF_HAND    = "/home/pranav/Space_touch/examples/allegro_hand_description/allegro_hand_description_left_digit_fixed.urdf"
URDF_SPHERE  = "/home/pranav/Space_touch/examples/objects/sphere_small.urdf"

# Velocity control parameters (replacing torque parameters)
VELOCITY_GAIN = 2.0         # Base gain for tendon force to velocity conversion
MAX_JOINT_VELOCITY = 3.0    # Maximum joint velocity (rad/s)
VELOCITY_SMOOTHING = 0.8    # Smoothing factor for velocity commands

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

# Reference axis points for each finger (parallel to finger, within hand)
# These points define a line parallel to each finger that serves as the tendon routing
FINGER_REFERENCE_AXES = {
    "index": {"start": np.array([0.0, -0.02, 0.0]), "end": np.array([0.0, -0.02, 0.08])},
    "middle": {"start": np.array([0.0, -0.01, 0.0]), "end": np.array([0.0, -0.01, 0.08])},
    "ring": {"start": np.array([0.0, 0.01, 0.0]), "end": np.array([0.0, 0.01, 0.08])},
    "thumb": {"start": np.array([-0.02, 0.0, 0.0]), "end": np.array([-0.02, 0.0, 0.06])}
}

class TendonVelocityController:
    """Implements tendon-based control with reference axis and velocity-based actuation"""
    
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
        
        # Pre-compute reference axis data and velocity scaling for each joint
        self.joint_velocity_scales = {}
        self._compute_velocity_scales()
        
        # Store previous velocity commands for smoothing
        self.prev_velocities = np.zeros(len(self.joint_indices))
    
    def _compute_velocity_scales(self):
        """Pre-compute velocity scaling factors for each joint relative to its finger's reference axis"""
        for finger, joint_indices in self.finger_joints.items():
            if finger not in FINGER_REFERENCE_AXES:
                continue
                
            axis_data = FINGER_REFERENCE_AXES[finger]
            axis_start = axis_data["start"]
            axis_direction = (axis_data["end"] - axis_data["start"])
            axis_direction = axis_direction / np.linalg.norm(axis_direction)
            
            self.joint_velocity_scales[finger] = []
            
            for i, joint_idx in enumerate(joint_indices):
                # Get joint position in hand frame (approximate)
                joint_info = p.getJointInfo(self.hand_id, joint_idx)
                joint_pos = np.array(joint_info[14])  # joint frame position relative to parent
                
                # Compute velocity scaling based on joint position in chain
                # Distal joints move faster for the same tendon displacement
                velocity_scale = self._compute_velocity_scale_for_joint(joint_pos, axis_start, axis_direction, i)
                self.joint_velocity_scales[finger].append(velocity_scale)
    
    def _compute_velocity_scale_for_joint(self, joint_pos, axis_start, axis_direction, joint_index):
        """
        Compute velocity scaling factor for a joint.
        Distal joints typically have higher velocity scaling.
        """
        # Base velocity scale increases along the chain (distal joints move faster)
        base_scale = 1.0 + (joint_index * 0.3)  # Each joint 30% faster than previous
        
        # Distance-based scaling (farther from axis = higher scaling)
        point_vec = joint_pos - axis_start
        projection_length = np.dot(point_vec, axis_direction)
        perpendicular_distance = np.linalg.norm(point_vec - projection_length * axis_direction)
        
        # Scale by perpendicular distance with minimum value
        distance_scale = max(perpendicular_distance * 10.0, 0.5)  # Scale by distance from axis
        
        return base_scale * distance_scale
    
    def compute_tendon_velocities(self, tendon_forces):
        """
        Compute velocity commands for each joint based on tendon forces.
        Higher tendon forces result in faster joint velocities.
        
        Args:
            tendon_forces: dict of finger -> normalized force (0 to 1)
        
        Returns:
            velocities: array of velocity commands for all joints
        """
        velocities = np.zeros(len(self.joint_indices))
        
        for finger, normalized_force in tendon_forces.items():
            if finger not in self.finger_joints or finger not in self.joint_velocity_scales:
                continue
            
            # Convert normalized force (0-1) to desired velocity
            target_velocity = normalized_force * MAX_JOINT_VELOCITY * VELOCITY_GAIN
            
            joints = self.finger_joints[finger]
            velocity_scales = self.joint_velocity_scales[finger]
            
            for i, joint_idx in enumerate(joints):
                if i >= len(velocity_scales):
                    continue
                
                # Scale velocity based on joint's position in finger chain
                velocity_scale = velocity_scales[i]
                joint_velocity = target_velocity * velocity_scale
                
                # Clamp to maximum velocity
                joint_velocity = np.clip(joint_velocity, -MAX_JOINT_VELOCITY, MAX_JOINT_VELOCITY)
                
                # Find index in joint list
                idx_in_list = self.joint_indices.index(joint_idx)
                velocities[idx_in_list] = joint_velocity
        
        # Apply velocity smoothing to reduce jitter
        smoothed_velocities = (VELOCITY_SMOOTHING * self.prev_velocities + 
                              (1.0 - VELOCITY_SMOOTHING) * velocities)
        self.prev_velocities = smoothed_velocities.copy()
        
        return smoothed_velocities
    
    def get_interpolated_forces(self, raw_forces, input_range=None):
        """
        Interpolate all finger forces to range [0, 1] for consistency.
        
        Args:
            raw_forces: dict of finger -> raw force value
            input_range: tuple (min_val, max_val) for input range, or None for auto-detection
        
        Returns:
            interpolated_forces: dict of finger -> interpolated force (0 to 1)
        """
        interpolated = {}
        
        # Auto-detect input range if not provided
        if input_range is None:
            all_values = [v for v in raw_forces.values() if v is not None]
            if all_values:
                input_min, input_max = min(all_values), max(all_values)
            else:
                input_min, input_max = 0.0, 1.0
        else:
            input_min, input_max = input_range
        
        # Avoid division by zero
        input_range_size = max(input_max - input_min, 1e-8)
        
        for finger in ["index", "middle", "ring", "thumb"]:
            if finger in raw_forces and raw_forces[finger] is not None:
                raw_val = raw_forces[finger]
                
                # Linear interpolation from input range to [0, 1]
                normalized = (raw_val - input_min) / input_range_size
                
                # Clamp to [0, 1] to handle values outside input range
                interpolated[finger] = np.clip(normalized, 0.0, 1.0)
            else:
                interpolated[finger] = 0.0
        
        return interpolated
    
    def get_interpolated_forces_sigmoid(self, raw_forces, center=0.0, steepness=1.0):
        """
        Alternative: Use sigmoid activation for smooth interpolation to [0, 1].
        Good for neural network outputs that might be unbounded.
        
        Args:
            raw_forces: dict of finger -> raw force value
            center: center point of sigmoid (0.0 = centered)
            steepness: steepness of sigmoid curve (1.0 = standard)
        
        Returns:
            interpolated_forces: dict of finger -> interpolated force (0 to 1)
        """
        interpolated = {}
        
        for finger in ["index", "middle", "ring", "thumb"]:
            if finger in raw_forces and raw_forces[finger] is not None:
                raw_val = raw_forces[finger]
                
                # Sigmoid: 1 / (1 + exp(-(steepness * (x - center))))
                sigmoid_val = 1.0 / (1.0 + np.exp(-steepness * (raw_val - center)))
                interpolated[finger] = sigmoid_val
            else:
                interpolated[finger] = 0.0
        
        return interpolated
    
    def get_interpolated_forces_tanh(self, raw_forces):
        """
        Alternative: Use tanh activation for interpolation to [0, 1].
        Maps (-inf, inf) to [0, 1] with smooth saturation.
        
        Args:
            raw_forces: dict of finger -> raw force value
        
        Returns:
            interpolated_forces: dict of finger -> interpolated force (0 to 1)
        """
        interpolated = {}
        
        for finger in ["index", "middle", "ring", "thumb"]:
            if finger in raw_forces and raw_forces[finger] is not None:
                raw_val = raw_forces[finger]
                
                # tanh maps (-inf, inf) to (-1, 1), then shift to (0, 1)
                tanh_val = (np.tanh(raw_val) + 1.0) / 2.0
                interpolated[finger] = tanh_val
            else:
                interpolated[finger] = 0.0
        
        return interpolated

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

    # 6) Initialize velocity-based tendon controller
    tendon_controller = TendonVelocityController(hand.id, joint_names, joint_inds)

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

        # a) Compute raw tendon forces for each finger (sinusoidal pattern)
        raw_tendon_forces = {}
        for finger, phase in finger_phases.items():
            # Generate force strength between 0 and 1
            force = 0.5 + 0.5 * AMP_TENDON * np.sin(2*np.pi*FREQ_TENDON*t + phase)
            raw_tendon_forces[finger] = force
        
        # b) Interpolate forces to ensure they're all in [0, 1] range
        # Choose one of these interpolation methods:
        
        # Method 1: Linear interpolation with known input range
        # tendon_forces = tendon_controller.get_interpolated_forces(raw_tendon_forces, input_range=(0.0, 1.0))
        
        # Method 2: Auto-detect range and interpolate
        tendon_forces = tendon_controller.get_interpolated_forces(raw_tendon_forces)
        
        # Method 3: Sigmoid interpolation (good for NN outputs)
        # tendon_forces = tendon_controller.get_interpolated_forces_sigmoid(raw_tendon_forces, center=0.5, steepness=2.0)
        
        # Method 4: Tanh interpolation (maps any range to [0,1])
        # tendon_forces = tendon_controller.get_interpolated_forces_tanh(raw_tendon_forces)
        
        # c) Get velocity commands from tendon controller
        velocities = tendon_controller.compute_tendon_velocities(tendon_forces)
        
        # d) Apply velocity control to joints
        p.setJointMotorControlArray(
            bodyUniqueId=hand.id,
            jointIndices=joint_inds,
            controlMode=p.VELOCITY_CONTROL,
            targetVelocities=velocities.tolist(),
            forces=[100.0] * len(joint_inds)  # Force limit for velocity control
        )

        # e) Base translate + rotate (unchanged from original)
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

        # f) Tacto render
        colors, depths = sensor.render()

        # g) Color mosaic + labels
        ordered_c = [colors[i] for i in cam_order]
        mosaic_c  = np.concatenate(ordered_c, axis=1)
        w = mosaic_c.shape[1] // len(labels)
        for idx, lab in enumerate(labels):
            cv2.putText(mosaic_c, lab, (idx*w+5, 15),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)
        cv2.imshow(title_c, mosaic_c)

        # h) Depth+mask mosaic + labels
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

        # i) Print deformation spans and tendon forces
        print("Depth spans:", ["%.3f"%s for s in spans])
        print("Tendon forces:", {f: "%.2f"%p for f, p in tendon_forces.items()})
        
        # Debug: Show computed velocities for first few joints
        if len(velocities) > 4:
            print("Sample velocities:", ["%.2f"%velocities[i] for i in range(4)])

        # j) Exit on Esc
        if cv2.waitKey(1) == 27:
            break
        time.sleep(1.0/SIM_FREQ)

    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()