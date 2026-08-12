# v3 Workflow: Baseline Replication

**Goal:** Find architecture that matches baseline's 0.2595 validation score  
**Why:** Verify our training pipeline works before trying improvements  
**Started:** Jan 8, 2026

---

## The Problem

After 2 failed attempts (v1, v2), we realized:
- ❌ v1: Overfitting (0.1788)
- ❌ v2: Wrong loss (0.0908)
- ❓ **We don't know if we can even replicate baseline!**

The baseline is a "Vanilla GRU" but we don't know:
- Hidden size?
- Number of layers?
- Dropout?
- Training hyperparameters?

## New Strategy

**Systematically try different architectures** until we match ~0.2595:

| Try | Architecture | Command | Expected Time |
|-----|-------------|---------|---------------|
| 1 | GRU(32h, 1L) | `python src/training/train_v3.py --hidden 32 --layers 1` | ~5-7 hours |
| 2 | GRU(64h, 1L) | `python src/training/train_v3.py --hidden 64 --layers 1` | ~5-7 hours |
| 3 | GRU(128h, 1L) | `python src/training/train_v3.py --hidden 128 --layers 1` | ~6-8 hours |
| 4 | GRU(64h, 2L) | `python src/training/train_v3.py --hidden 64 --layers 2` | ~6-8 hours |
| 5 | GRU(128h, 2L) | `python src/training/train_v3.py --hidden 128 --layers 2` | ~7-9 hours |

**Stop when:** We get validation WPCC ≥ 0.25 (close to baseline 0.2595)

---

## Workflow for Each Try

### 1. Train
```bash
cd /Users/thomasmyles/dev/wunder-predictorium
source venv/bin/activate

# Example: Try 1 (32 hidden, 1 layer)
python src/training/train_v3.py --hidden 32 --layers 1

# Or Try 2 (64 hidden, 1 layer)
python src/training/train_v3.py --hidden 64 --layers 1
```

**What you'll see:**
```
Training v3: Baseline Replication
============================================================
Architecture: GRU(hidden=32, layers=1, dropout=0.0)
Loss: Standard MSE

Epoch 1/10
------------------------------------------------------------
Training:   0%|          | 0/37692 [00:00<?, ?it/s]
```

### 2. Validate
After training completes:
```bash
# Validate (using same config as training)
python src/evaluation/validate_v3.py --hidden 32 --layers 1
```

**What you'll see:**
```
VALIDATION RESULTS
============================================================
Mean Weighted Pearson: 0.XXXX
  t0: 0.XXXX
  t1: 0.XXXX
============================================================

Baseline (v0) validation: 0.2595
v3 (h=32, l=1) validation: 0.XXXX
Change: +/-0.XXXX (+/-XX.XX%)
```

### 3. Decision Tree

**If WPCC ≥ 0.25:** ✅ SUCCESS!
- We found a working architecture
- Document which config worked
- Now we can confidently try improvements

**If 0.20 ≤ WPCC < 0.25:** ⚠️ CLOSE
- Architecture might be right
- Try tweaking: more epochs, different LR, etc.

**If WPCC < 0.20:** ❌ WRONG ARCHITECTURE
- Try next configuration
- Document the result

---

## Configuration Options

All available arguments:
```bash
python src/training/train_v3.py \
  --hidden 64 \        # Hidden size: 32, 64, 128
  --layers 1 \         # Number of layers: 1 or 2
  --dropout 0.0 \      # Dropout: 0.0, 0.1, 0.2
  --epochs 10 \        # Number of epochs
  --lr 0.001 \         # Learning rate
  --batch_size 256     # Batch size
```

---

## Progress Tracker

### Try 1: GRU(32h, 1L)
- [ ] Train (5-7 hours)
- [ ] Validate
- **Result:** ___ WPCC
- **Status:** ___

### Try 2: GRU(64h, 1L)
- [ ] Train (5-7 hours)
- [ ] Validate
- **Result:** ___ WPCC
- **Status:** ___

### Try 3: GRU(128h, 1L)
- [ ] Train (6-8 hours)
- [ ] Validate
- **Result:** ___ WPCC
- **Status:** ___

### Try 4: GRU(64h, 2L)
- [ ] Train (6-8 hours)
- [ ] Validate
- **Result:** ___ WPCC
- **Status:** ___

### Try 5: GRU(128h, 2L)
- [ ] Train (7-9 hours)
- [ ] Validate
- **Result:** ___ WPCC
- **Status:** ___

---

## What We're Keeping Consistent

✅ **Same across all tries:**
- Loss function: Standard MSE (not weighted!)
- Optimizer: AdamW (lr=0.001, weight_decay=0.01)
- Scheduler: ReduceLROnPlateau
- Batch size: 256
- Epochs: 10
- Data: Same train/valid split
- Features: All 32 raw features

❓ **What we're varying:**
- GRU hidden size
- Number of GRU layers

---

## Expected Outcomes

### Best Case
- One of the first 2-3 tries matches baseline (~0.25+)
- We know that architecture works
- We can move forward with confidence

### Worst Case
- None of the tries match baseline
- Then we know the issue is NOT just architecture
- Might be: data preprocessing, training procedure, or something else

Either way, we learn something valuable!

---

## After Finding a Match

Once we match baseline:

1. **Document it** in daily log
2. **Save the model** as our new baseline replica
3. **Then try improvements:**
   - More training data
   - Feature engineering
   - Better hyperparameter tuning
   - Ensemble methods

But we do this FROM A KNOWN WORKING BASELINE!

---

## Quick Commands (Copy-Paste)

### Start Try 1:
```bash
cd /Users/thomasmyles/dev/wunder-predictorium && source venv/bin/activate && python src/training/train_v3.py --hidden 32 --layers 1
```

### Validate Try 1:
```bash
cd /Users/thomasmyles/dev/wunder-predictorium && source venv/bin/activate && python src/evaluation/validate_v3.py --hidden 32 --layers 1
```

### Start Try 2:
```bash
cd /Users/thomasmyles/dev/wunder-predictorium && source venv/bin/activate && python src/training/train_v3.py --hidden 64 --layers 1
```

### Validate Try 2:
```bash
cd /Users/thomasmyles/dev/wunder-predictorium && source venv/bin/activate && python src/evaluation/validate_v3.py --hidden 64 --layers 1
```

---

## Files Created

```
src/models/baseline_replica.py       # Configurable GRU model
src/training/train_v3.py            # Training with argparse
src/evaluation/validate_v3.py       # Validation with argparse
models/v3_h32_l1.pt                 # Saved model (after training)
models/v3_h64_l1.pt                 # Different configs...
```

---

## Key Differences from v1/v2

| Aspect | v1/v2 | v3 |
|--------|-------|-----|
| Goal | "Improve" baseline | Replicate baseline first |
| Approach | One shot guess | Systematic search |
| Loss | MSE (v1), Weighted (v2) | Standard MSE only |
| Architecture | Fixed guess | Multiple tries |
| Philosophy | Assume we're right | Verify we can match baseline |

**This is how we should have started!** 🎯

