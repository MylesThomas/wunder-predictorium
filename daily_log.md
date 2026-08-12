# Daily Progress Log - Wunder Challenge

---

# Day 1 - Saturday, January 3, 2026
**Focus:** Run and understand the baseline

## Tasks
- [x] Run baseline solution (example_solution/solution.py)
- [x] Document baseline Weighted Pearson score
- [x] Study GRU architecture in baseline code
- [x] Understand ONNX export process

---

## How to Run Baseline

```bash
# From wnn_predictorium_starterpack directory (with venv activated)
cd /Users/thomasmyles/dev/wunder-predictorium/wnn_predictorium_starterpack
source ../venv/bin/activate
PYTHONPATH=/Users/thomasmyles/dev/wunder-predictorium/wnn_predictorium_starterpack:$PYTHONPATH python example_solution/solution.py
```

**What it does:**
- Loads validation dataset (datasets/valid.parquet) - 1,444 sequences
- Loads pre-trained baseline.onnx model (Vanilla GRU)
- Runs inference step-by-step through all sequences
- Calculates Weighted Pearson correlation score

**Note:** The solution uses a 100-step context window for predictions

---

## Results ✅

### Baseline Performance
- **Mean Weighted Pearson Correlation: 0.2595**
- **t0 correlation: 0.3884**
- **t1 correlation: 0.1306**
- **Inference time: 7 min 11 sec** (for 1,444,000 rows / 1,301,044 predictions)
- **Throughput: ~3,347 samples/sec**

**Key Observations:**
- Model performs much better on t0 (price) than t1 (volume)
- t0 score of 0.388 is decent for baseline
- t1 score of 0.130 has significant room for improvement
- Overall score of 0.259 is our benchmark to beat

---

## Architecture Analysis

### Baseline GRU Model Structure

**Input Processing:**
```python
Input Shape: (batch=1, sequence_length=100, features=32)
```
- Uses a **sliding window of 100 steps** from sequence history
- Takes last 100 observations (steps t-99 to t)
- Maintains `sequence_history` list that accumulates all steps

**Model Architecture (from solution.py):**
```
32 features → [GRU Layers] → 2 outputs (t0, t1)
```
- Model is pre-trained and exported to ONNX (baseline.onnx)
- Comment says "VanillaLSTM" but it's actually GRU
- Output shape: (1, 2) - returns predictions for both targets

**State Management:**
```python
# Resets on new sequence
if self.current_seq_ix != data_point.seq_ix:
    self.current_seq_ix = data_point.seq_ix
    self.sequence_history = []  # Clear history
```
- **Critical:** Sequence history resets for each new `seq_ix`
- Accumulates all 1000 steps but only uses last 100 for prediction
- No warm-up period needed (predictions start at step 99)

**ONNX Runtime Configuration:**
```python
sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
providers=['CPUExecutionProvider']
```
- Full graph optimization enabled
- CPU-only inference (no GPU)
- Uses all available cores

**Prediction Logic:**
- Returns `None` when `need_prediction=False` (steps 0-98)
- Returns `np.array([t0_pred, t1_pred])` when `need_prediction=True` (steps 99-999)
- Handles both 2D and 3D output shapes from ONNX

### Key Design Decisions

1. **100-step window:** Captures ~10% of sequence context
2. **Stateless inference:** No hidden state passed between predictions
3. **Sliding window:** Inefficient (reprocesses overlapping data) but simple
4. **Single model:** Predicts both t0 and t1 together (multi-task learning)

### Potential Improvements

- ✅ Increase window size (100 → 150 or 200)
- ✅ Use stateful RNN (pass hidden state between steps)
- ✅ Separate models for t0 and t1
- ✅ Batch predictions for speed
- ✅ Add attention mechanism
- ✅ Increase model capacity (more layers/units)

---

## ONNX Export Analysis

### Model File Details
```
File: baseline.onnx
Size: 273 KB
Format: ONNX (Open Neural Network Exchange)
```

### ONNX Model Specifications

**Input:**
```
Name: "input"
Shape: [batch_size, 100, 32]
Type: tensor(float32)
```
- Dynamic batch size (can process multiple sequences at once)
- Fixed sequence length: 100 steps
- Fixed features: 32

**Output:**
```
Name: "output"  
Shape: [batch_size, 2]
Type: tensor(float32)
```
- Returns 2 values: [t0_prediction, t1_prediction]
- Last timestep output only (sequence-to-value)

### Why ONNX?

**Advantages:**
1. **Speed:** Optimized inference (~3,347 samples/sec)
2. **Portability:** Works across frameworks (PyTorch → ONNX → Runtime)
3. **Size:** 273 KB is tiny (easy to submit)
4. **Optimization:** Graph-level optimizations (operator fusion, constant folding)
5. **Competition requirement:** Fast inference needed

**Typical Export Process (PyTorch):**
```python
# 1. Train model in PyTorch
model = GRUModel(input_size=32, hidden_size=128, output_size=2)
model.train()  # training loop...

# 2. Export to ONNX
model.eval()
dummy_input = torch.randn(1, 100, 32)
torch.onnx.export(
    model,
    dummy_input,
    "baseline.onnx",
    input_names=['input'],
    output_names=['output'],
    dynamic_axes={'input': {0: 'batch_size'}}
)

# 3. Load in ONNX Runtime for inference
session = ort.InferenceSession("baseline.onnx")
```

### Runtime Configuration

**From solution.py:**
```python
sess_options = ort.SessionOptions()
sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
```

**Optimization levels:**
- `ORT_DISABLE_ALL`: No optimizations
- `ORT_ENABLE_BASIC`: Basic optimizations (constant folding)
- `ORT_ENABLE_EXTENDED`: Extended optimizations (operator fusion)
- `ORT_ENABLE_ALL`: All optimizations ✅ (used in baseline)

**Provider:** `CPUExecutionProvider`
- Uses CPU only (no GPU requirement)
- Good for submission (runs anywhere)
- Could use `CUDAExecutionProvider` for GPU if available

---

## Day 1 Summary ✅

**Completed:**
- ✅ Ran baseline and documented performance (0.2595 score)
- ✅ Analyzed architecture (100-step window, stateless GRU)
- ✅ Understood ONNX export and optimization
- ✅ Identified 6 potential improvements

**Key Findings:**
1. Baseline achieves **0.2595** (t0: 0.388, t1: 0.131)
2. Model performs 3x better on t0 than t1
3. Uses simple 100-step sliding window approach
4. ONNX provides fast inference (273 KB model)
5. Architecture is simple - lots of room for improvement

---

---

# Day 2 - Sunday, January 4, 2026
**Focus:** Create and submit v0 baseline

## Tasks
- [x] Review submission requirements and format
- [x] Create v0 basic solution (copy/modify baseline)
- [x] Test v0 solution locally with scorer
- [x] Package and submit v0 to leaderboard

---

## Submission Requirements (Reviewed)

### Required Structure
```
submission.zip
├── solution.py         # Must be at root with PredictionModel class
└── baseline.onnx       # Or other model files
```

### Key Requirements
- ✅ `solution.py` at root level
- ✅ `PredictionModel` class with `predict()` method
- ✅ Returns `None` when `need_prediction=False`
- ✅ Returns `np.array([t0, t1])` when `need_prediction=True`
- ✅ Handles sequence resets (new `seq_ix`)

### Resource Limits
- CPU: 1 vCPU core
- RAM: 16 GB
- Time: 90 minutes max
- No GPU, no internet
- Environment: `python:3.11-slim-bookworm`

---

## v0 Submission Created ✅

### Files
```
submissions/v0_baseline/
├── solution.py (4.8 KB)
└── baseline.onnx (273 KB)
```

### Local Test Results
```
Mean Weighted Pearson: 0.259505
  t0: 0.388378
  t1: 0.130631

Processing time: 6:47 (407 seconds)
Throughput: ~3,542 samples/sec
```

### Submission Package
```
File: submissions/v0_submission.zip
Size: 253 KB (compressed from 277 KB)
Contents:
  - solution.py (compressed)
  - baseline.onnx (compressed)
```

### Submission Status ✅

**Submitted:** Saturday, January 3, 2026 @ 10:30 PM ET
**Status:** Complete
**Processing time:** ~11 hours (completed Sunday morning)

### Leaderboard Results 🎉

**Official Score: 0.2761**
**Rank: 27/2846** (top 0.95%!)
**Submissions: 1**

### Score Analysis

**Comparison:**
- Local validation: 0.2595
- Test leaderboard: **0.2761**
- Difference: **+0.0166** (+6.4% better on test!)

**Why test > validation?**
- Test set may have slightly different distribution
- Or validation set is harder
- Good sign: model generalizes well!

### Leaderboard Context

**Baseline tier (0.2761):**
- 🤖 example_solution: 0.2761 (rank 9)
- Many participants (ranks 9-31) have exactly 0.2761
- This is the "everyone who submitted baseline unchanged" tier

**Top performers:**
- #1: insuperabilehart - **0.3051** (+10.5% over baseline, 25 submissions)
- #2: cteceliker - **0.2931** (+6.2% over baseline, 4 submissions)
- #3: aks - **0.2924** (+5.9% over baseline, 9 submissions)
- #6: sultanmunirov - **0.2861** (+3.6% over baseline, 15 submissions)

**Gap analysis:**
- To reach #1: Need **+0.029** improvement (~10.5% gain)
- To reach top 5: Need **+0.015** improvement (~5.4% gain)
- To beat baseline tier: Need **+0.001** improvement (any improvement moves us up)

---

## Next Steps After Submission

1. **Monitor leaderboard** for v0 results
2. **Analyze gap** between validation and test scores
3. **Plan improvements** for v1:
   - Increase model capacity (hidden size, layers)
   - Add custom weighted loss
   - Try longer context window (150 steps)
   - Add dropout for regularization

---

## Day 2 Summary ✅

**Completed:**
- ✅ Reviewed submission requirements and packaging
- ✅ Created v0 baseline submission (copy of provided example)
- ✅ Tested locally (0.2595 score)
- ✅ Packaged and submitted to leaderboard
- ✅ **Achieved rank 27/2846 (0.2761) - top 1%!**

**Key Findings:**
- Test score (0.2761) > validation score (0.2595) - good generalization!
- Matched official baseline exactly
- 22 people ahead using baseline tier (0.2761)
- Top performer at 0.3051 (+10.5% over baseline)
- Clear path to improvement: bigger models, better loss, feature engineering

**What top performers are doing:**
- Multiple submissions (4-25 attempts)
- Likely: larger models, custom loss functions, feature engineering
- Need to iterate and experiment

---

---

# Day 3 - Sunday, January 5, 2026
**Focus:** Iterative improvements - test one change at a time

## Model Versions & Locations

```
submissions/
├── v0_baseline/          # Baseline (provided example)
│   ├── solution.py
│   ├── baseline.onnx
│   └── v0_submission.zip
├── v1_improved_arch/     # Bigger architecture (in progress)
│   ├── solution.py       (to be created)
│   ├── v1_model.onnx     (to be created)
│   └── v1_submission.zip (to be created)
├── v2_weighted_loss/     # v1 + weighted loss (planned)
└── v3_features/          # v2 + feature engineering (planned)

models/                   # Trained PyTorch models
├── v1_improved_gru.pt    # v1 checkpoint (training now)
├── v2_weighted.pt        # v2 checkpoint (planned)
└── v3_features.pt        # v3 checkpoint (planned)

src/
├── models/
│   ├── improved_gru.py   # v1 model architecture
│   └── ...
└── training/
    ├── train_v1.py       # v1 training script
    └── ...
```

## Strategy
Test changes incrementally to understand what actually works:
1. v0 → v1: Bigger architecture only
2. v1 → v2: Add weighted loss
3. v2 → v3: Add feature engineering
4. Track scores after each submission

## Baseline (v0)
- **Local validation:** 0.2595
- **Test (leaderboard):** 0.2761
- **Rank:** 27/2846

---

## v1: Improved Architecture

### Changes from v0:
- Hidden size: 128 → 256 (more capacity)
- Layers: 1 → 2 (deeper network)
- Added dropout: 0.3 (GRU) + 0.2 (Dense)
- Added Dense layer before output
- **Loss:** Standard MSE (same as baseline)

### Tasks
- [x] Analyze test vs validation gap
- [x] Design v1 architecture
- [x] Implement v1 model (PyTorch)
- [x] Train on training set
- [x] Test on validation set
- [ ] ~~Export to ONNX~~ (skipped - model worse than baseline)
- [ ] ~~Submit v1 to leaderboard~~ (skipped - model worse than baseline)

### Results ❌ FAILED
- **Training loss:** 0.9307
- **Validation loss:** 2.4405 (gap = 1.51 → severe overfitting!)
- **Local validation WPCC:** 0.1788
- **Test (leaderboard):** ❌ NOT SUBMITTED
- **Change from v0:** -0.0807 (-31.1%)

### Root Cause: Overfitting
Bigger model (256 hidden, 2 layers, dropout) learned training data but couldn't generalize.
**Key lesson:** Start with baseline architecture, change ONE thing at a time.

---

## v2: Weighted Loss ❌ FAILED (WORSE THAN v1!)

**Status:** ✅ Completed - DISASTROUS results, NOT submitted

### Hypothesis (WRONG!)
"If we weight t0 4x in the loss to match WPCC (0.8*t0 + 0.2*t1), the model will optimize for what we're evaluated on."

### Changes from v0:
- ⬇️ **Architecture:** Same as baseline (64 hidden, 1 layer, light dropout)
- ❌ **Loss function:** MSE → Weighted MSE (t0=4x, t1=1x)
- Same 32 raw features

### Model Stats
- **Parameters:** 20,962
- **Training samples:** 9.6M from 10,721 sequences
- **Validation samples:** 1.3M from 1,444 sequences
- **Epochs:** 10 (trained 8:18AM - 3:54PM, ~7.5 hours)

### Results ❌ CATASTROPHIC FAILURE
- **Training loss:** 2.010 → 1.787 (decreasing)
- **Validation loss:** 5.071 → 6.856 (increasing!)
- **Local validation WPCC:** 0.0908 (t0: 0.109, t1: 0.016)
- **Test (leaderboard):** ❌ NOT SUBMITTED
- **Change from v0:** -0.1687 (-65.0%) - WORSE than v1!

### Root Cause: Wrong Optimization Target
**Critical mistake:** Minimizing weighted MSE ≠ maximizing Pearson correlation!
- Weighted loss forced model to heavily prioritize t0 absolute error
- This distorted predictions and destroyed correlation for BOTH targets
- Model overfitted to weighted loss (val loss increased every epoch)
- **Lesson:** Don't optimize for a proxy metric - it can backfire spectacularly

### Key Insights
1. **MSE and correlation are different objectives**
   - Low MSE doesn't guarantee high correlation
   - Weighting makes it worse by distorting the prediction distribution
2. **Baseline's standard MSE is actually working well**
   - Maybe we shouldn't mess with the loss function
3. **Both v1 and v2 made things worse**
   - v1: Overfitting from too much capacity
   - v2: Wrong loss function destroyed correlation
4. **Need a different approach for v3**

---

## v3: What Should We Try Next?

**Current situation:** Both attempts to improve baseline have failed badly.

### What We've Learned (The Hard Way)
1. ❌ **Bigger model doesn't help** (v1: overfitting)
2. ❌ **Weighted loss backfires** (v2: wrong optimization target)
3. ✅ **Baseline is actually pretty good** (0.2595 local, 0.2761 test)

### Possible Directions for v3

#### Option A: Keep It Simple, Tune Better
- Go back to baseline architecture + standard MSE
- Focus on training improvements:
  - Better learning rate schedule
  - More epochs with early stopping
  - Different optimizer settings
  - Gradient clipping adjustments

#### Option B: Feature Engineering
- Keep baseline arch + MSE (what works)
- Add hand-crafted features:
  - Order book imbalance: `(bid_vol - ask_vol) / (bid_vol + ask_vol)`
  - Spread features
  - Rolling statistics (mean, std over last N steps)
  - Momentum indicators
- Risk: More features might help or might add noise

#### Option C: Data Augmentation
- Keep everything same as baseline
- Train with more data:
  - Combine train + validation sets
  - Use longer sequences or different windows

#### Option D: Different Architecture (Risky)
- Try Transformer instead of GRU
- Try CNN for capturing patterns
- Risk: Could easily overfit like v1

### Recommendation
**Start with Option A or B** - low risk, builds on what works.

What do you think?

---

## Day 3 Summary ✅

**Completed:**
- ✅ Trained v1 model (bigger architecture)
- ✅ Validated v1 → 0.1788 WPCC (failed - overfitting)
- ✅ Designed v2 approach (weighted loss)
- ✅ Implemented v2 model (baseline arch + weighted MSE)
- ✅ Started v2 training (8:18AM)

**Key Findings:**
- v1 overfitted badly (67k params too many, dropout didn't help)
- Decision: Revert to baseline architecture for v2
- Only change loss function to test one variable at a time

**Blockers:** None - v2 training overnight

---

# Day 4 - Wednesday, January 8, 2026
**Focus:** Complete v2, analyze failures, plan better v3 strategy

## Tasks Completed
- [x] Completed v2 training (~7.5 hours, 10 epochs)
- [x] Validated v2 on validation set
- [x] Analyzed v2 failure (weighted loss backfired)
- [x] Documented both v1 and v2 failures
- [x] Updated all documentation with results

## v2 Training Results
- **Started:** 8:18AM
- **Completed:** ~3:54PM
- **Duration:** ~7.5 hours (10 epochs)
- **Training loss:** 2.010 → 1.787
- **Validation loss:** 5.071 → 6.856 (increasing = bad sign!)
- **Final WPCC:** 0.0908 (-65% vs baseline)

## Critical Insights from v1 & v2 Failures

### What We Tried & Why It Failed

| Version | Change | Hypothesis | Result | Why Failed |
|---------|--------|-----------|---------|------------|
| v1 | Bigger model (256h, 2L) | More capacity = better learning | 0.1788 (-31%) | Overfitting - too complex for data |
| v2 | Weighted loss (4x t0) | Match eval metric weights | 0.0908 (-65%) | MSE ≠ correlation, distorted predictions |

### Key Lessons
1. **Baseline is surprisingly good** (0.2595 local, 0.2761 test)
2. **Bigger isn't better** - led to overfitting
3. **Don't optimize wrong metric** - weighted MSE destroyed correlation
4. **Change one thing at a time** - confirmed this principle is critical
5. **Need fundamentally different approach** - can't just tweak baseline

## v3 Strategy Discussion

### Options Considered

**A. Better Training Hyperparameters** (Low Risk)
- Keep baseline arch + MSE
- Tune: learning rate schedule, more epochs, early stopping
- Pro: Safe, builds on what works
- Con: Unlikely to get big gains

**B. Feature Engineering** (Moderate Risk)  
- Keep baseline arch + MSE
- Add: order book imbalance, spread, rolling stats, momentum
- Pro: Only thing we haven't tried yet
- Con: More features could add noise

**C. More Training Data** (Low Risk)
- Combine train + validation sets
- Pro: More data usually helps
- Con: Lose validation set for local testing

**D. Different Architecture** (High Risk)
- Try Transformer, CNN, or other architectures
- Pro: Could capture different patterns
- Con: Likely to overfit like v1

### Decision: v3 Strategy

**Critical Realization:** We've been trying to "improve" baseline without knowing if we can even **replicate** it!

**New Approach:**
1. First, find architecture that matches baseline's 0.2595 validation score
2. Once we can replicate it, we know our training pipeline works
3. THEN we can make actual improvements with confidence

### v3 Plan: Systematic Architecture Search
Try different GRU configs with standard MSE loss until we match baseline:
- [x] Try 1: GRU(32 hidden, 1 layer) → **0.047 (-81.88%)** ❌ TERRIBLE
- [ ] Try 2: GRU(64 hidden, 1 layer) 
- [ ] Try 3: GRU(128 hidden, 1 layer)
- [ ] Try 4: GRU(64 hidden, 2 layers)
- [ ] Try 5: GRU(128 hidden, 2 layers)

**Stop when:** We get ~0.25+ validation score (close to baseline)
**Then:** We have a working baseline replica to improve from!

### v3 Try 1 Results (GRU 32h, 1L)
- **Trained:** 10 epochs, ~3.5 hours
- **Model:** `models/v3_h32_l1.pt` (91KB, only 6,898 params!)
- **Validation WPCC:** 0.047 
  - t0: 0.058
  - t1: 0.001 (basically zero!)
- **Result:** **COMPLETE FAILURE** - worse than v1 and v2 combined
- **Analysis:** Model way too small - can't learn anything meaningful from the data

---

## Model Versions & Locations

| Version | Location | Status |
|---------|----------|--------|
| v0 | `submissions/v0_baseline/` | ✅ Submitted (0.2761) |
| v1 | `models/v1_improved_gru.pt` | ❌ Failed - overfitting |
| v2 | `models/v2_weighted_gru.pt` | ❌ Failed - weighted loss backfired |
| v3 Try 1 | `models/v3_h32_l1.pt` | ❌ Failed - way too small (0.047) |

**Training code:**
- v1: `src/models/improved_gru.py`, `src/training/train_v1.py`
- v2: `src/models/weighted_gru.py`, `src/training/train_v2.py`
- Validation: `src/evaluation/validate_v{1,2}.py`

## Score Tracking Table

| Version | Architecture | Loss | Features | Params | Val Score | Test Score | Rank | Delta |
|---------|-------------|------|----------|--------|-----------|------------|------|-------|
| v0 | GRU(64,1) | MSE | 32 raw | ~20k | **0.2595** | **0.2761** | **27** | - |
| v1 | GRU(256,2) | MSE | 32 raw | 67k | 0.1788 | ❌ | - | -0.0807 (-31%) |
| v2 | GRU(64,1) | Weighted MSE | 32 raw | 21k | **0.0908** | ❌ | - | **-0.1687 (-65%)** |
| v3 Try 1 | GRU(32,1) | MSE | 32 raw | 7k | **0.047** | ❌ | - | **-0.2125 (-81.88%)** 💀 |

**Key insight:** Baseline is still the best! Every attempt has made it worse. v3 Try 1 is the worst yet!

---

## Notes
