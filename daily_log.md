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

### Submission Status ⏳

**Submitted:** Saturday, January 3, 2026 @ 10:30 PM ET
**Status:** Processing (waiting for results)
**Current leaderboard position:** 2780/2794 (temporary - no score yet)
**Estimated completion:** 10-30 minutes

Check status at: https://predictorium.wundernn.io/leaderboard

---

## How to Submit

### Web Interface
1. Go to: https://predictorium.wundernn.io/submit
2. Upload: `submissions/v0_submission.zip`
3. Wait for scoring (~90 minutes max)
4. Check leaderboard: https://predictorium.wundernn.io/leaderboard

---

## Expected Leaderboard Performance

**Local Validation Score:** 0.2595

**Expected Test Score:** ~0.25-0.27 (assuming similar distribution)

**Note:** Leaderboard score may differ slightly from validation due to:
- Different data distribution in test set
- Random noise in targets
- Model generalization

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
- ✅ Tested locally (0.2595 score matches baseline)
- ✅ Packaged and submitted to leaderboard

**Key Actions:**
- Established baseline on leaderboard
- Verified submission pipeline works
- Ready to iterate and improve

---

## Notes

- This is our baseline - just copied the provided example
- Goal is to establish a benchmark on the leaderboard
- All improvements will be measured against this 0.2595 score
- Quick submission to understand the evaluation process
