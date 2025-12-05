# V6 SC-1 Tuning Summary

## 🎯 Problem Identified

After analyzing the initial training run (steps 0-10K), we found:

1. **Agent not learning to approach target**
   - Distance oscillating between 0.6-1.4m with no convergence trend
   - Random exploration behavior, not directed policy

2. **Proximity reward too weak**
   - Exponential decay `exp(-5.0 * distance)` too steep
   - At 1.0m: reward ≈ 0.07 (almost zero signal)
   - Couldn't compete with random exploration noise

3. **Quality reward dominating**
   - Quality reward (3.6-5.0) was larger than proximity (0.5-3.5)
   - Agent learned "spread fingers" but not "approach target"

4. **Time-based curriculum transitions**
   - Phase 1→2 transition at 300K steps regardless of learning
   - Agent advanced to Phase 2 without mastering Phase 1

---

## ✅ Tuning Changes Applied

### 1. Reward Parameters (`v6_reward.py`)

#### Proximity Reward - Slower Decay
```python
# BEFORE:
proximity = cfg['distance_weight'] * np.exp(-5.0 * distance)
# At 1.0m: 10 * exp(-5.0) = 10 * 0.0067 = 0.07

# AFTER:
proximity = cfg['distance_weight'] * np.exp(-2.0 * distance)
# At 1.0m: 50 * exp(-2.0) = 50 * 0.135 = 6.77
```

**Impact:** Much stronger gradient at medium-far distances (0.5-1.5m)

#### Distance Weight - Increased
```python
Phase 0: 10.0 → 50.0  (5x stronger)
Phase 1: 10.0 → 35.0  (3.5x stronger)
Phase 2:  2.0 →  5.0  (2.5x stronger)
```

**Proximity Rewards at Different Distances:**

| Distance | Phase 0 (Old) | Phase 0 (New) | Phase 1 (Old) | Phase 1 (New) |
|----------|---------------|---------------|---------------|---------------|
| 0.05m    | 7.79          | **45.24**     | 7.79          | **31.67**     |
| 0.10m    | 6.07          | **40.94**     | 6.07          | **28.66**     |
| 0.20m    | 3.68          | **33.52**     | 3.68          | **23.47**     |
| 0.50m    | 0.82          | **18.39**     | 0.82          | **12.87**     |
| 1.00m    | 0.07          | **6.77**      | 0.07          | **4.74**      |

#### Quality Weight - Phase-Dependent
```python
Phase 0: quality_weight = 1.0  (reduced from implicit 5.0)
Phase 1: quality_weight = 2.0  (reduced)
Phase 2: quality_weight = 3.0  (moderate)
Phase 3: quality_weight = 5.0  (full)
```

**Impact:** Proximity dominates early learning, quality increases later

#### Overlap Weight - Increased
```python
Phase 0: 100.0 → 200.0
Phase 1: 100.0 → 200.0
Phase 2: 500.0 (unchanged)
```

---

### 2. Phase 0 Ultra-Close Learning (NEW)

Added a new initial phase to bootstrap learning:

```python
Phase 0: {
    'distance_weight': 50.0,        # Very strong proximity
    'overlap_weight': 200.0,        # Strong overlap signal
    'contact_penalty': -1.0,        # Gentle - allow exploration
    'quality_weight': 1.0,          # Minimal - focus on approach
    'success_overlap': 0.000003,    # 3 cm³ to advance
    'success_distance': 0.12,       # Must stay within 12cm
}
```

**Starting Position:**
- Phase 0: 5cm from target (ultra-close)
- Phase 1: 8cm from target (close)
- Phase 2+: 10cm from target (standard)

---

### 3. Performance-Based Curriculum (`v6_sc1.py`)

Replaced time-based transitions with performance-based criteria:

#### Phase 0 → Phase 1 Criteria:
```python
min_steps: 50,000          # At least 50K steps
max_steps: 200,000         # Force advance if stuck
mean_distance: < 0.12m     # Average distance < 12cm
mean_overlap: > 3 cm³      # Average overlap > 3 cm³
window_size: 500           # Average over last 500 steps
```

#### Phase 1 → Phase 2 Criteria:
```python
min_steps: 100,000
max_steps: 400,000
mean_distance: < 0.15m     # Average distance < 15cm
mean_overlap: > 5 cm³      # Average overlap > 5 cm³
window_size: 1000
```

#### Phase 2 → Phase 3 Criteria:
```python
min_steps: 150,000
max_steps: 500,000
mean_distance: < 0.12m     # Stay close
mean_overlap: > 15 cm³     # Average overlap > 15 cm³
window_size: 1000
```

**Console Output on Transition:**
```
[Step 87,500] Phase 0 performance goals met:
  Mean distance: 0.1150m (target: <0.12)
  Mean overlap: 3.2400 cm³ (target: >3.0000 cm³)
Phase 0 → Phase 1
```

---

## 📊 Expected Behavior After Tuning

### First 10K Steps (Phase 0):
- ✅ Distance should **decrease** from ~0.05m to ~0.08m (hand exploring but staying close)
- ✅ Proximity reward should be **consistently high** (30-45)
- ✅ Small overlap events should start appearing (0.5-3 cm³)
- ✅ Quality reward low (1-2) - agent focused on approach

### Steps 50K-100K (Late Phase 0 / Early Phase 1):
- ✅ Mean distance stabilizing around 0.10-0.12m
- ✅ Overlap events more frequent (2-5 cm³)
- ✅ Phase 0→1 transition when criteria met

### Steps 100K-300K (Phase 1):
- ✅ Mean distance decreasing toward 0.12m
- ✅ Overlap increasing toward 5-10 cm³
- ✅ Quality reward increasing (agent learning finger positioning)

---

## 🔍 Monitoring During Training

### Key Metrics to Watch:

1. **`state/distance`** - Should show **downward trend** in first 50K steps
2. **`reward/proximity`** - Should stay **high (>10)** throughout Phase 0
3. **`stats/mean_overlap_cm3`** - Should show **upward trend**
4. **`state/phase`** - Should advance when performance criteria met
5. **`reward/quality`** - Should be **low initially**, increase in later phases

### Red Flags:
- ❌ Distance not decreasing after 50K steps
- ❌ Proximity reward dropping to <5 consistently
- ❌ Phase transition before mean criteria met
- ❌ Overlap staying at zero for >20K steps

---

## 🚀 Training Command

```bash
cd "Code_Pranav/RL Code"

# Start fresh training with tuned parameters
python v6_sc1.py train --timesteps 1000000

# Or resume if you have a checkpoint
python v6_sc1.py train --timesteps 1000000 --resume path/to/checkpoint.zip
```

**Recommended:** Let it run for at least 100K steps before evaluating. Phase 0→1 transition should happen around 50-150K steps if tuning is working.

---

## 📈 Comparison: Before vs After Tuning

| Metric | Before Tuning | After Tuning |
|--------|---------------|--------------|
| **Proximity @ 0.5m** | 0.82 | 18.39 (22x stronger!) |
| **Proximity @ 1.0m** | 0.07 | 6.77 (97x stronger!) |
| **Quality weight (Phase 0)** | 5.0 | 1.0 (5x reduced) |
| **Starting distance** | 8cm | 5cm (closer) |
| **Phase transitions** | Time-based (blind) | Performance-based (adaptive) |
| **Phase 0 exists?** | No | Yes (ultra-close learning) |

---

## ✅ Verification Results

```
✓ V6RewardCalculator initialized
  Starting phase: 0

Proximity reward at different distances (Phase 0, weight=50):
  0.05m → 45.24
  0.10m → 40.94
  0.20m → 33.52
  0.50m → 18.39
  1.00m → 6.77

✓ V6CurriculumCallback initialized
  Starting phase: 0
  Phase 0 thresholds: {
    'min_steps': 50000,
    'max_steps': 200000,
    'mean_distance': 0.12,
    'mean_overlap': 3e-06,
    'window_size': 500
  }

✓ All tuning verified and ready for training!
```

---

## 🎓 Lessons Learned

1. **Exponential decay rates matter**: `-5.0` was too steep, `-2.0` provides better gradient
2. **Reward balance is critical**: Quality reward was inadvertently dominating
3. **Curriculum needs performance gates**: Time-based transitions advance too early
4. **Bootstrap with easy scenarios**: Phase 0 at 5cm gives agent easy wins to learn from

---

**All tuning complete and verified. Ready for training!**
