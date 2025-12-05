# V7 SC-1 Soft-Capture Training - Improved Curriculum

## Overview

V7 improves upon V6 with a cleaner, more pedagogical 3+1 phase curriculum optimized for **200K timesteps on CPU**.

**Key Philosophy**: **Teach one skill at a time, remove training wheels when ready**.

---

## V7 vs V6 Comparison

| Aspect | V6 | V7 |
|--------|----|----|
| **Total Steps** | 500K-1M | **200K** (CPU optimized) |
| **Phases** | 4 phases, overlapping objectives | **3+1 phases, isolated skills** |
| **Phase 1** | Proximity + Overlap (mixed) | **Pure approach** (proximity dominant) |
| **Phase 2** | Proximity + More Overlap | **Pure envelopment** (NO proximity) |
| **Phase 3** | Maximize Overlap | **Precision balance** (overlap + clearance) |
| **Skill Focus** | Gradual weight adjustment | **Sharp transitions, force independence** |
| **Learning Rate** | 3e-4 | **3e-4** (stable) |

---

## 4-Phase Curriculum (200K Total)

### Phase 0: Bootstrap (0-30K steps, 15%)
**Goal**: Learn basic movement controls

**Rewards**:
- `proximity_weight`: 50.0 (strong approach signal)
- `overlap_weight`: 20.0 (introduce concept)
- `contact_penalty`: -1.0 (gentle)
- `quality_weight`: 1.0 (minimal)

**Starting Position**: 5cm from target (ultra-close)

**Success Criteria**:
- Mean distance < 15cm
- Mean overlap > 10 cm³
- Min 20K steps, Force advance at 30K

**What Agent Learns**: "I can move the hand. Getting close gives rewards."

---

### Phase 1: Approach (30K-90K steps, 30%)
**Goal**: Master navigation to target

**Rewards**:
- `proximity_weight`: 50.0 (**PRIMARY** objective)
- `overlap_weight`: 20.0 (**SECONDARY** - awareness only)
- `contact_penalty`: -2.0 (moderate)
- `quality_weight`: 1.0 (minimal)

**Starting Position**: Random around target

**Success Criteria**:
- Mean distance < 15cm
- Mean overlap > 10 cm³
- Min 50K steps, Force advance at 60K

**What Agent Learns**: "I need to navigate close to the target consistently."

---

### Phase 2: Envelopment (90K-160K steps, 35%)
**Goal**: Learn finger coordination for overlap

**Rewards**:
- `proximity_weight`: **0.0** (**REMOVED** - force pure overlap learning)
- `overlap_weight`: 500.0 (**SOLE** primary objective)
- `contact_penalty`: -5.0 (moderate)
- `quality_weight`: 3.0 (reward finger spread)

**Key Innovation**: Distance termination at 25cm prevents wandering

**Success Criteria**:
- Mean overlap > 100 cm³
- Mean distance < 20cm (termination enforces < 25cm)
- Min 60K steps, Force advance at 70K

**What Agent Learns**: "Proximity doesn't help anymore. I must coordinate fingers to maximize overlap."

---

### Phase 3: Precision (160K-200K steps, 20%)
**Goal**: Balance overlap + no contact

**Rewards**:
- `proximity_weight`: **0.0** (still removed)
- `overlap_weight`: 300.0 (high)
- `clearance_weight`: 200.0 (**NEW** - reward optimal distance)
- `contact_penalty`: -20.0 (**HARSH** - strong deterrent)
- `quality_weight`: 5.0 (full weight)

**Target Clearance**: 2cm from object surface

**Success Criteria**:
- Mean overlap > 150 cm³
- Consecutive no-contact steps > 50
- Train to completion (200K)

**What Agent Learns**: "I must maintain high overlap while avoiding contact."

---

## Training Configuration

### Hyperparameters
```python
learning_rate: 3e-4      # Stable for PPO
n_steps: 2048            # PPO rollout buffer
batch_size: 64           # Training batch
n_epochs: 10             # PPO updates per batch
gamma: 0.99              # Discount factor
gae_lambda: 0.95         # GAE parameter
clip_range: 0.2          # PPO clip
ent_coef: 0.001          # Low entropy (reduced exploration)
```

### Environment
```python
Action space: 10D
  - 6 DOF: Base position (xyz) + rotation (rpy)
  - 4 DOF: Tendon commands (one per finger)

Observation space: 28D
  - Base position (3), target position (3)
  - Base velocity (3), angular velocity (3)
  - Finger positions (12)
  - Tactile contacts (4)

Simulation: 240Hz
Control: Direct position control (hand-finger sync fix)
Episode length: 500 steps max, or distance > 25cm
```

### Memory Optimizations (from V6)
- ✅ Hull computation: 10Hz (24 step intervals)
- ✅ PyBullet reset: Every 10 episodes
- ✅ Callback sampling: Every 100 steps
- ✅ WandB logging: Every 480 steps (aligned with hull computation)
- ✅ No step_counter reset bug

---

## Usage

### Training

```bash
cd "Code_Pranav/RL Code"

# Train new model (200K steps, default)
python v7_sc1.py train

# Custom timesteps
python v7_sc1.py train --timesteps 300000

# Resume from checkpoint
python v7_sc1.py train --resume path/to/checkpoint.zip

# With visualization (slower)
python v7_sc1.py train --vis
```

### Testing

```bash
# Test trained model
python v7_sc1.py test path/to/model.zip --episodes 10

# With visualization
python v7_sc1.py test path/to/model.zip --episodes 5 --vis
```

### Monitor Training

**WandB Dashboard**: https://wandb.ai/your-username/sc1-v7-curriculum

**Key Metrics to Watch**:
1. `state/distance` - Should **decrease** in Phase 0-1
2. `hull/overlap_cm3` - Should **increase** in Phase 2-3
3. `reward/proximity` - High in Phase 0-1, **zero in Phase 2-3**
4. `reward/overlap` - Low in Phase 0-1, **dominant in Phase 2-3**
5. `state/phase` - Should advance: 0→1 at ~30K, 1→2 at ~90K, 2→3 at ~160K

---

## Expected Training Timeline

**On CPU (8 it/s)**:
- Phase 0 (30K): ~1 hour
- Phase 1 (60K): ~2 hours
- Phase 2 (70K): ~2.5 hours
- Phase 3 (40K): ~1.5 hours
- **Total**: ~7 hours for full 200K training

**Phase Transitions**:
- **Step ~30K**: Phase 0→1 (learned basic approach)
- **Step ~90K**: Phase 1→2 (mastered navigation, now pure overlap)
- **Step ~160K**: Phase 2→3 (learned envelopment, now add precision)

---

## Troubleshooting

### If Phase Transitions Don't Happen:

**Phase 0→1 stuck**:
- Check `state/distance` - should be < 15cm average
- Check `hull/overlap_cm3` - should be > 10 cm³ average
- May need to lower thresholds if agent struggling

**Phase 1→2 stuck**:
- Same checks as Phase 0→1
- Will force advance at 60K steps (90K total)

**Phase 2→3 stuck**:
- Check `hull/overlap_cm3` - should be > 100 cm³ average
- Phase 2 is hardest - may need more time
- Will force advance at 70K steps (160K total)

### If Training Crashes:

**Memory issues**:
- All V6 fixes included, should not crash
- If crash at ~4600 steps: Check PyBullet reset is working

**Slow training**:
- 8 it/s on CPU is expected
- For faster training: Use GPU (change device='cpu' to device='cuda')

---

## Files

**Core**:
- `v7_sc1.py` - Main training script
- `v7_reward.py` - Reward calculator

**Generated During Training**:
- `SC1_Training_Runs/V7_SC1_TIMESTAMP/` - Run directory
  - `checkpoints/` - Saved every 50K steps
  - `progress.csv` - Training metrics
  - `tensorboard/` - TensorBoard logs
- `wandb/` - WandB logs

---

## Next Steps After Training

1. **Test the model**:
   ```bash
   python v7_sc1.py test SC1_Training_Runs/V7_SC1_TIMESTAMP/final_model.zip --episodes 10
   ```

2. **Analyze results**:
   - Check test plots in `v7_test_results_TIMESTAMP/`
   - Look at success rates across different scenarios
   - Compare overlap volumes to training

3. **If results good**: Deploy to real hardware!

4. **If results poor**:
   - Analyze which phase failed
   - Adjust phase thresholds or weights
   - Retrain with modified curriculum

---

## Improvements Over V6

1. ✅ **Faster training**: 200K vs 500K-1M steps
2. ✅ **Clearer learning**: One skill per phase
3. ✅ **Better phase 2**: Removes proximity to force pure overlap learning
4. ✅ **Optimized for CPU**: Realistic training time (7 hours)
5. ✅ **All V6 bug fixes**: Memory, timing, synchronization
6. ✅ **Better monitoring**: Phase progress clearly visible in WandB

---

**V7 Status**: ✅ Ready for training
**Recommended First Run**: 200K steps, monitor phase transitions

Good luck! 🚀
