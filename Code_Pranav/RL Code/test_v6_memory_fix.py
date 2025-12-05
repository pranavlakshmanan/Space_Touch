#!/usr/bin/env python3
"""
Quick test to verify v6_sc1.py memory leak fix
Runs training for 10,000 steps and monitors memory usage
"""

import os
import sys
import time
import psutil
import subprocess
from datetime import datetime

def get_memory_usage():
    """Get current process memory in MB"""
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / 1024 / 1024

def monitor_training():
    """Monitor v6 training memory usage"""
    print("="*60)
    print("V6 Memory Leak Fix Verification Test")
    print("="*60)
    print(f"Start time: {datetime.now().strftime('%H:%M:%S')}")
    print(f"Test duration: 10,000 timesteps (~2-3 minutes)")
    print()

    # Start training process
    cmd = [
        sys.executable,
        "v6_sc1.py",
        "train",
        "--timesteps", "10000"
    ]

    print(f"Starting: {' '.join(cmd)}")
    print()

    process = subprocess.Popen(
        cmd,
        cwd=os.path.dirname(os.path.abspath(__file__)),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1
    )

    start_time = time.time()
    last_mem_check = start_time
    peak_memory = 0
    memory_samples = []

    try:
        # Monitor process
        ps_process = psutil.Process(process.pid)

        for line in iter(process.stdout.readline, ''):
            # Print training output
            print(line.rstrip())

            # Check memory every 5 seconds
            current_time = time.time()
            if current_time - last_mem_check >= 5.0:
                try:
                    mem_mb = ps_process.memory_info().rss / 1024 / 1024
                    memory_samples.append(mem_mb)
                    peak_memory = max(peak_memory, mem_mb)

                    elapsed = current_time - start_time
                    print(f"[MEMORY CHECK] Elapsed: {elapsed:.0f}s | "
                          f"Current: {mem_mb:.1f}MB | Peak: {peak_memory:.1f}MB")

                    last_mem_check = current_time
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    break

        # Wait for completion
        return_code = process.wait()

        elapsed = time.time() - start_time

        print()
        print("="*60)
        print("TEST RESULTS")
        print("="*60)

        if return_code == 0:
            print("✓ Training completed successfully!")
            print(f"  Duration: {elapsed:.1f} seconds")
            print(f"  Peak memory: {peak_memory:.1f} MB")

            if memory_samples:
                avg_mem = sum(memory_samples) / len(memory_samples)
                mem_growth = memory_samples[-1] - memory_samples[0] if len(memory_samples) > 1 else 0

                print(f"  Average memory: {avg_mem:.1f} MB")
                print(f"  Memory growth: {mem_growth:+.1f} MB")

                if mem_growth > 100:
                    print("  ⚠ WARNING: Significant memory growth detected!")
                    print("             Memory leak may still be present.")
                elif mem_growth > 50:
                    print("  ⚠ CAUTION: Moderate memory growth.")
                    print("             Monitor longer runs carefully.")
                else:
                    print("  ✓ Memory growth minimal - leak appears fixed!")

            print()
            print("Next steps:")
            print("  1. Review training logs in SC1_Training_Runs/")
            print("  2. Run longer test: python v6_sc1.py train --timesteps 100000")
            print("  3. Monitor with: watch -n 1 'ps aux | grep v6_sc1'")

        else:
            print(f"✗ Training crashed with return code: {return_code}")
            print("  Memory leak fix may not be complete.")
            print(f"  Peak memory before crash: {peak_memory:.1f} MB")
            print()
            print("Debug steps:")
            print("  1. Check crash logs in SC1_Training_Runs/")
            print("  2. Run: dmesg | tail -50  (check for OOM killer)")
            print("  3. Review V6_CRASH_FIX_SUMMARY.md")

        print("="*60)
        return return_code

    except KeyboardInterrupt:
        print("\n\nTest interrupted by user (Ctrl+C)")
        process.terminate()
        process.wait(timeout=5)
        return 1

if __name__ == "__main__":
    # Check if psutil is available
    try:
        import psutil
    except ImportError:
        print("ERROR: psutil not installed")
        print("Install with: pip install psutil")
        sys.exit(1)

    # Check if v6_sc1.py exists
    v6_path = os.path.join(os.path.dirname(__file__), "v6_sc1.py")
    if not os.path.exists(v6_path):
        print(f"ERROR: v6_sc1.py not found at {v6_path}")
        sys.exit(1)

    sys.exit(monitor_training())
