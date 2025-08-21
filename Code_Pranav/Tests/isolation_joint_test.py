#!/usr/bin/env python3
import time
import pybullet as p
import pybullet_data

# ——— CONFIG ———
URDF_PATH = "/home/pranav/Space_touch/examples/allegro_hand_description/allegro_hand_description_left_digit.urdf"  # adjust if needed
TARGET_POS = 0.5     # radians
TEST_FORCE  = 5.0    # N·m
TEST_STEPS  = 240    # 1 second @240 Hz

def main():
    # 1) connect & scene
    p.connect(p.GUI)
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    p.resetSimulation()
    p.setGravity(0, 0, 0)
    plane = p.loadURDF("plane.urdf")
    
    # 2) load hand
    start_pos = [0,0,0.2]
    hand_id = p.loadURDF(URDF_PATH, start_pos, useFixedBase=True)

    
    # teting URDF finger limits and findout which joints have effort limits and angle limits 
    # for j in (0, 5, 10):
    #     info      = p.getJointInfo(hand_id, j)
    #     name      = info[1].decode()
    #     lower_lim = info[8]
    #     upper_lim = info[9]
    #     effort    = info[10]
    #     print(f"joint {j:2d} ({name}): limits = [{lower_lim:.3f}, {upper_lim:.3f}],  effort={effort}")



# Joints that have issues - 
# joint  0 (joint_8.0): limits = [-0.470, 0.470],  effort=0.0
# joint  5 (joint_4.0): limits = [-0.470, 0.470],  effort=0.0
# joint 10 (joint_0.0): limits = [-0.470, 0.470],  effort=0.0
#Hence will be modifyung the URDF file to set the effort limits ie joints 8,4,0 . 

    
    # 3) gather revolute joints
    revolute_joints = []
    print("\nAvailable joints:")
    for j in range(p.getNumJoints(hand_id)):
        info = p.getJointInfo(hand_id, j)
        name, jtype = info[1].decode(), info[2]
        print(f"  idx={j:2d}  name={name:20s}  type={jtype}")
        # type 0 = REVOLUTE, 1 = PRISMATIC, 4 = FIXED
        if jtype == p.JOINT_REVOLUTE:
            revolute_joints.append(j)
    
    print(f"\nTesting {len(revolute_joints)} revolute joints one by one...\n")
    time.sleep(1)
    
    # 4) helper to disable all motors
    def disable_all():
        p.setJointMotorControlArray(
            hand_id,
            jointIndices=list(range(p.getNumJoints(hand_id))),
            controlMode=p.VELOCITY_CONTROL,
            forces=[0]*p.getNumJoints(hand_id)
        )

    # 5) loop through each joint
    results = {}
    for j in revolute_joints:
        print(f"→ Testing joint idx={j}")
        # reset all joint states to zero
        for jj in revolute_joints:
            p.resetJointState(hand_id, jj, targetValue=0.0, targetVelocity=0.0)
        disable_all()
        
        # command this one in POSITION_CONTROL
        p.setJointMotorControl2(
            bodyUniqueId=hand_id,
            jointIndex=j,
            controlMode=p.POSITION_CONTROL,
            targetPosition=TARGET_POS,
            positionGain=0.2,
            velocityGain=1.0,
            force=TEST_FORCE
        )
        
        # step and wait
        for _ in range(TEST_STEPS):
            p.stepSimulation()
            time.sleep(1/240.)
        
        # read out final angle
        angle = p.getJointState(hand_id, j)[0]
        results[j] = angle
        print(f"  -> reached {angle:.3f} rad\n")
        time.sleep(0.5)
    
    # 6) summary
    print("=== Test complete ===")
    for j, ang in results.items():
        ok = "OK" if abs(ang - TARGET_POS) > 1e-2 else "FAILED"
        print(f"  joint {j:2d}: final={ang:.3f}  [{ok}]")
    
    input("\nPress Enter to exit…")
    p.disconnect()

if __name__=="__main__":
    main()
