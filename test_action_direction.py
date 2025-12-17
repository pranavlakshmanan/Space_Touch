#!/usr/bin/env python3
import numpy as np
import sys
sys.path.append("/home/ubuntu/workspace/Space_Touch/Code_Pranav/RL Code")
from v7_6_sc1 import V76Environment

env = V76Environment(vis=False, max_steps=100)
env.reset()

print(f"Target position: {env.target_pos}")
print(f"Initial hand base will be offset from target by spawn offset")
print()

# Test each axis
for axis, name in [(0, 'X'), (1, 'Y'), (2, 'Z')]:
    env.reset()
    action = np.array([[0.0]*10])
    
    # Get initial position
    env.step(action)
    ri = env.latest_info
    initial_dist = ri.get('palm_to_target_distance', 0)
    
    # Move positive on this axis
    action[0][axis] = 1.0
    for _ in range(5):
        env.step(action)
    ri = env.latest_info
    positive_dist = ri.get('palm_to_target_distance', 0)
    
    # Reset and move negative
    env.reset()
    env.step(np.array([[0.0]*10]))
    action[0][axis] = -1.0
    for _ in range(5):
        env.step(action)
    ri = env.latest_info
    negative_dist = ri.get('palm_to_target_distance', 0)
    
    print(f"Axis {name}: +1 action → dist={positive_dist*100:.1f}cm, -1 action → dist={negative_dist*100:.1f}cm")

env.close()
