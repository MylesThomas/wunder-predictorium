# Day 1 - Saturday, January 3, 2026
**Focus:** Run and understand the baseline

---

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

**Next Steps (Tomorrow):**
- Implement improved GRU with larger capacity
- Add custom weighted loss function
- Train and compare to baseline
- Target: +0.02-0.05 improvement

---

## Notes


