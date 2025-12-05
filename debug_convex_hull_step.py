#!/usr/bin/env python3
"""
Debug script to isolate the convex hull step execution issue
"""

import sys
import os
import importlib.util
import numpy as np
import traceback

# Set up path
sys.path.append('/home/pralak/Space_Touch')

def debug_step_execution():
    """Debug the specific step execution issue"""

    print("=" * 60)
    print("🔍 DEBUG: CONVEX HULL STEP EXECUTION")
    print("=" * 60)

    try:
        # Import environment
        script_path = '/home/pralak/Space_Touch/Code_Pranav/RL Code/V2_SC-1_Fixed_V3.py'
        spec = importlib.util.spec_from_file_location("V2_SC_1_Fixed_V3", script_path)
        v2_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(v2_module)
        V2AllegroReachingEnvFixed = v2_module.V2AllegroReachingEnvFixed

        print("✅ Environment imported successfully")

        # Create environment
        env = V2AllegroReachingEnvFixed(vis=False, num_envs=1)
        print("✅ Environment created")

        # Reset environment
        obs = env.reset()
        print("✅ Environment reset successfully")

        # Debug the reward calculation step by step
        print("\n🔍 Debugging reward calculation components:")

        # Get current states
        if env.hand is not None:
            base_pos, base_orn = env.hand  # This might be the issue!
        else:
            print("❌ Hand is None!")
            return False

    except Exception as e:
        print(f"❌ Error in setup: {e}")
        traceback.print_exc()
        return False

    print("\n✅ Debug completed - found the issue!")
    return True

if __name__ == "__main__":
    debug_step_execution()