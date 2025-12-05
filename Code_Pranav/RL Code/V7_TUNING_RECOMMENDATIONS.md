# V7 Training Analysis & Tuning Recommendations

## Training Run Analysis (40 Epochs / ~92K Steps)

### What Worked ✓
1. **Phase transitions occurring** - Phase 1→2 transition around step 90K
2. **Proximity reward zeroing** - Correctly drops in Phase 2
3. **Curriculum structure** - Phase system executing as designed
4. **No crashes** - All V6 memory fixes working

### Issues Identified ❌

#### 1. **Phase 2 Learning Failure** (Critical)
**Observation:**
- `hull/overlap_cm3` remains noisy (50-250 cm³)
- No upward trend despite pure overlap focus (Phase 2: 90K+ steps)
- Expected: Steady increase to 100+ cm³ average

**Root Cause:**
- Overlap reward signal too weak relative to noise
- Possible sparse reward problem (overlap happens infrequently)
- Hand may be "giving up" after Phase 1 proximity removed

#### 2. **High Contact Frequency** (Major)
**Observation:**
- Frequent spikes in `state/contacts` (0-2 contacts)
- Contact penalties not deterring collisions
- Should see declining trend, especially in Phase 3

**Root Cause:**
- Contact penalty too weak (-5.0 in Phase 2)
- No progressive penalty increase

#### 3. **Distance Instability** (Moderate)
**Observation:**
- `state/distance` oscillates wildly (0.1-0.7m)
- Should converge to stable <0.15m in Phase 2
- 25cm termination not preventing wandering enough

**Root Cause:**
- Agent "forgetting" proximity in Phase 2 (as designed, but too harsh)
- No distance baseline reward after Phase 1

#### 4. **Hand Volume Variance** (Moderate)
**Observation:**
- `hull/hand_volume_cm3` varies 400-1400 cm³
- Suggests fingers collapsing or overextending
- Quality reward not shaping finger configuration

**Root Cause:**
- Quality reward weight too low (3.0 in Phase 2)
- Finger spread metric may not capture finger collapse

---

## Tuning Recommendations

### Option A: Conservative Tuning (Recommended First)
**Goal:** Fix Phase 2 learning without major architectural changes

#### 1. Strengthen Overlap Reward Signal
```python
2: {  # Phase 2: ENVELOPMENT
    'overlap_weight': 500.0 → 800.0,  # +60% increase
    'overlap_threshold': 0.00001,      # Keep same (10 cm³)
}
```

#### 2. Add Distance Baseline Reward (NEW)
```python
2: {  # Phase 2
    'proximity_weight': 0.0 → 10.0,   # Small baseline to prevent wandering
    'proximity_scale': 0.5,            # Exponential scale: exp(-0.5 * distance)
}
```
**Rationale:** Pure zero proximity too harsh. Keep agent "aware" of distance.

#### 3. Progressive Contact Penalties
```python
1: {'contact_penalty': -2.0 → -3.0},   # Phase 1: Moderate
2: {'contact_penalty': -5.0 → -10.0},  # Phase 2: Stronger
3: {'contact_penalty': -20.0},          # Phase 3: Keep harsh
```

#### 4. Boost Quality Reward
```python
2: {
    'quality_weight': 3.0 → 8.0,        # Emphasize finger spread
    'quality_bonus_threshold': 0.04,    # Bonus if spread > 4cm
}
```

#### 5. Early Termination for Poor Performance
```python
# In _check_termination():
if self.reward_calc.current_phase >= 2:
    if distance > 0.20:  # 20cm (was 25cm)
        done = True
```

#### 6. Extend Phase 2 Duration
```python
# In V7CurriculumCallback phase_thresholds:
2: {
    'min_steps': 60000 → 80000,   # Force more Phase 2 time
    'max_steps': 70000 → 100000,  # Delay Phase 3 transition
}
```

---

### Option B: Aggressive Tuning (If Option A Fails)
**Goal:** Restructure Phase 2 with denser rewards

#### 1. Add Shaped Overlap Reward
```python
def _calculate_overlap_reward(self, overlap_cm3):
    """Multi-tier reward for progressive learning"""
    if overlap_cm3 < 10:
        return 0
    elif overlap_cm3 < 50:
        return 50 * (overlap_cm3 / 50)  # Linear 0-50
    elif overlap_cm3 < 100:
        return 50 + 100 * ((overlap_cm3 - 50) / 50)  # Linear 50-150
    else:
        return 150 + 200 * np.tanh((overlap_cm3 - 100) / 100)  # Saturating
```

#### 2. Add "Near-Grasp" Bonus
```python
# Reward when fingers are positioned around object (even without overlap)
finger_angles = calculate_finger_angles_around_object(finger_positions, object_pos)
if finger_angles > 180:  # Fingers span >180° around object
    near_grasp_bonus = 20.0
```

#### 3. Increase Hull Computation Frequency
```python
self.hull_compute_freq = 24 → 12  # 10Hz → 20Hz
```
**Rationale:** More frequent feedback for faster learning

#### 4. Add Phase 1.5 (Intermediate Phase)
```python
# Insert between Phase 1 and 2:
1.5: {  # Phase 1.5: Proximity + Overlap Balance (90K-120K)
    'proximity_weight': 25.0,   # Half of Phase 1
    'overlap_weight': 200.0,    # 4x Phase 1
    'contact_penalty': -5.0,
}
```

---

### Option C: Hyperparameter Tuning (PPO)
**Goal:** Improve exploration and learning stability

#### 1. Increase Entropy for Exploration
```python
ent_coef: 0.001 → 0.005  # Encourage more exploration in Phase 2
```

#### 2. Reduce Learning Rate in Phase 2
```python
# Add adaptive learning rate in model:
if phase >= 2:
    learning_rate = 3e-4 → 1e-4  # More stable learning
```

#### 3. Increase Rollout Buffer
```python
n_steps: 2048 → 4096  # More experience before update
```

---

## Implementation Priority

### Immediate Changes (V7.1 - Quick Win)
1. ✅ Increase overlap weight: 500 → 800
2. ✅ Add small proximity baseline in Phase 2: 0 → 10
3. ✅ Strengthen contact penalties: -5 → -10 (Phase 2)
4. ✅ Boost quality weight: 3 → 8
5. ✅ Extend Phase 2: 70K → 100K max steps
6. ✅ Tighter distance termination: 25cm → 20cm

### Secondary Changes (V7.2 - If Needed)
1. Implement shaped overlap reward (Option B.1)
2. Add near-grasp bonus (Option B.2)
3. Increase hull computation: 10Hz → 20Hz
4. Increase entropy: 0.001 → 0.005

### Long-term Changes (V8 - Major Refactor)
1. Add Phase 1.5 (gradual transition)
2. Implement adaptive learning rate
3. Consider reward normalization
4. Add curriculum auto-tuning based on performance

---

## Testing Protocol

### V7.1 Validation Run
```bash
# Short validation run (50K steps, ~1 hour)
python v7_sc1.py train --timesteps 50000 --vis
```

**Success Criteria:**
- [ ] `hull/overlap_cm3` mean increases in Phase 2 (target: >80 cm³ by 50K)
- [ ] `state/distance` stabilizes <0.18m in Phase 2
- [ ] `state/contacts` frequency decreases over time
- [ ] `hull/hand_volume_cm3` variance <400 cm³

### Full Training Run (If Validation Passes)
```bash
# Full 200K training
python v7_sc1.py train --timesteps 200000
```

**Success Criteria:**
- [ ] Phase 2→3 transition occurs (160K steps)
- [ ] Final mean overlap >120 cm³
- [ ] Contact rate <10% in Phase 3
- [ ] Success rate >30% on test scenarios

---

## Monitoring Checklist

During training, watch for:

**Phase 1 (0-90K):**
- ✓ Distance decreasing
- ✓ Overlap starting to appear (>10 cm³)
- ✓ Proximity reward high (~30-40)

**Phase 2 (90K-160K):**
- ❌ **CRITICAL:** Overlap should INCREASE steadily
- ✓ Distance should stay <0.20m
- ✓ Hand volume should stabilize (800±200 cm³)

**Phase 3 (160K-200K):**
- ✓ Overlap maintaining >100 cm³
- ✓ Contacts dropping to near-zero
- ✓ Clearance reward positive

---

## Next Steps

1. **Implement V7.1** with Option A changes
2. **Run 50K validation** to verify improvements
3. **If successful:** Full 200K training run
4. **If unsuccessful:** Implement Option B changes (V7.2)
5. **Document results** for future reference

---

## Questions to Investigate

1. **Is the hand getting stuck in local minima?**
   - Check `reward/overlap` for plateaus
   - Inspect hand position diversity (add position logging)

2. **Are fingers actually moving in Phase 2?**
   - Log tendon commands: `action[6:10]`
   - Check finger position variance

3. **Is overlap calculation correct?**
   - Add debug visualization of hand/object hulls
   - Verify bbox overlap matches true overlap

4. **Should we use true ConvexHull instead of bbox?**
   - Test scipy.spatial.ConvexHull for accuracy
   - Measure performance impact

---

**Status:** Ready to implement V7.1
**Confidence:** High (70%) that Option A will improve Phase 2 learning
**Risk:** Low (all changes are reversible, no architectural changes)
