#!/usr/bin/env python3
"""
Simple training monitor - check WandB metrics without CLI
"""

import wandb
import time

def monitor_current_run():
    """Monitor current training run"""

    try:
        # Connect to wandb
        api = wandb.Api()

        # Get latest run from your project
        runs = api.runs("space-touch-convex-hull-3phase", order="-created_at")

        if not runs:
            print("❌ No runs found in project")
            return

        latest_run = runs[0]
        print(f"🎯 Monitoring: {latest_run.name}")
        print(f"🔗 Dashboard: {latest_run.url}")
        print(f"📊 State: {latest_run.state}")
        print(f"⏱️  Runtime: {latest_run.summary.get('_runtime', 0):.0f}s")

        # Key metrics
        metrics = latest_run.summary
        timestep = metrics.get('train/timestep', 0)
        overlap_vol = metrics.get('hull_cm3/overlap_volume', 0)
        phase = metrics.get('curriculum/phase', 1)
        reward = metrics.get('episode/reward_mean', 0)

        print(f"\n📈 Current Metrics:")
        print(f"   Timesteps: {timestep:,}")
        print(f"   Phase: {phase}")
        print(f"   Overlap Volume: {overlap_vol:.4f} cm³")
        print(f"   Mean Reward: {reward:.2f}")

        # Success indicators
        if overlap_vol > 0:
            print(f"   🎉 SUCCESS: Non-zero overlap detected!")
        else:
            print(f"   ⏳ Waiting for first overlap detection...")

        return latest_run.url

    except Exception as e:
        print(f"❌ Error accessing WandB: {e}")
        print(f"💡 Try opening: https://wandb.ai/{api.default_entity}/space-touch-convex-hull-3phase")
        return None

if __name__ == "__main__":
    url = monitor_current_run()
    if url:
        print(f"\n🌐 Open this URL to see full dashboard:")
        print(f"   {url}")