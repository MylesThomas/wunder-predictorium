# v2 Workflow Checklist

**Model:** Weighted Loss GRU (baseline arch + weighted MSE)  
**Started:** Jan 5, 2026 ~11:45pm ET  
**Status:** 🔄 Training in progress

---

## Quick Summary

**What changed from v1:**
- ⬇️ Reverted to baseline architecture (64 hidden, 1 layer)
- ✅ Added weighted MSE loss (t0=4x, t1=1x)
- 🎯 Goal: Test if weighted loss helps without complexity

**Why this approach:**
v1 failed due to overfitting (bigger model). Following principle: **change ONE thing at a time**.

---

## Workflow Steps

### ✅ 1. Design & Implement
- [x] Create model architecture (`src/models/weighted_gru.py`)
- [x] Create training script (`src/training/train_v2.py`)
- [x] Create validation script (`src/evaluation/validate_v2.py`)

### 🔄 2. Training (IN PROGRESS)
- [x] Start training
- [ ] Wait for completion (~2-3 hours)
- [ ] Check terminal 8 for final results

**Command:**
```bash
# Running in terminal 8
cd /Users/thomasmyles/dev/wunder-predictorium
source venv/bin/activate
python src/training/train_v2.py
```

**Check progress:**
```bash
tail -100 /Users/thomasmyles/.cursor/projects/Users-thomasmyles-dev-wunder-predictorium/terminals/8.txt
```

### 3. Validation
Once training completes, validate the model:

```bash
cd /Users/thomasmyles/dev/wunder-predictorium
source venv/bin/activate
python src/evaluation/validate_v2.py
```

**Decision point:**
- ✅ If score > 0.2595 (baseline): Proceed to ONNX export
- ❌ If score < 0.2595: Analyze and plan v3

### 4. ONNX Export (if validation passes)

Create export script: `src/export/export_v2_to_onnx.py`

```python
import torch
from src.models.weighted_gru import WeightedGRU

# Load trained model
model = WeightedGRU()
checkpoint = torch.load('models/v2_weighted_gru.pt')
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

# Create dummy input: (batch=1, seq_len=100, features=32)
dummy_input = torch.randn(1, 100, 32)

# Export to ONNX
torch.onnx.export(
    model,
    dummy_input,
    'submissions/v2_weighted_loss/v2_model.onnx',
    input_names=['input'],
    output_names=['output'],
    dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}},
    opset_version=14
)
```

**Run:**
```bash
python src/export/export_v2_to_onnx.py
```

### 5. Create Submission Package

**Create directory:**
```bash
mkdir -p submissions/v2_weighted_loss
```

**Copy solution.py from v0 and modify:**
```bash
cp submissions/v0_baseline/solution.py submissions/v2_weighted_loss/solution.py
```

**Update `solution.py`:**
- Change `baseline.onnx` → `v2_model.onnx` in the file
- Update docstring with v2 details

**Verify ONNX is there:**
```bash
ls -lh submissions/v2_weighted_loss/
# Should see: solution.py, v2_model.onnx
```

### 6. Local Testing

Test the submission package:

```bash
cd wnn_predictorium_starterpack
python example_solution/solution.py  # Replace with ../submissions/v2_weighted_loss/solution.py path
```

### 7. Package for Submission

```bash
cd submissions/v2_weighted_loss
zip -r v2_submission.zip solution.py v2_model.onnx
```

**Verify:**
```bash
unzip -l v2_submission.zip
# Should show: solution.py, v2_model.onnx
```

### 8. Submit to Leaderboard

1. Go to: https://predictorium.wundernn.io/submissions
2. Upload `v2_submission.zip`
3. Wait for scoring (~5-10 minutes)
4. Record score in daily log

### 9. Document Results

Update `daily_log.md`:
- Training loss
- Validation loss  
- Local validation WPCC
- Test score (from leaderboard)
- Rank
- Delta from v0

---

## Model Details

**Architecture:**
```
Input (32 features)
  ↓
GRU(input=32, hidden=64, layers=1, batch_first=True)
  ↓
Last timestep → (batch, 64)
  ↓
Linear(64→32) + ReLU
  ↓
Dropout(0.1)
  ↓
Linear(32→2) → (t0, t1)
```

**Training Config:**
- Loss: Weighted MSE (t0=4.0x, t1=1.0x)
- Optimizer: AdamW (lr=0.001, weight_decay=0.01)
- Scheduler: ReduceLROnPlateau (factor=0.5, patience=1)
- Batch size: 256
- Epochs: 10
- Gradient clipping: max_norm=1.0

**Data:**
- Training: 9.6M samples from 10,721 sequences
- Validation: 1.3M samples from 1,444 sequences
- Window size: 100 timesteps
- Features: 32 (p0-p11, v0-v11, dp0-dp3, dv0-dv3)

---

## Key Files

```
src/models/weighted_gru.py          # Model architecture
src/training/train_v2.py            # Training script
src/evaluation/validate_v2.py       # Validation script
models/v2_weighted_gru.pt           # Trained checkpoint (after training)
submissions/v2_weighted_loss/       # Submission package (after export)
```

---

## Expected Timeline

| Step | Duration |
|------|----------|
| Training | ~2-3 hours |
| Validation | ~5 minutes |
| ONNX Export | <1 minute |
| Packaging | <1 minute |
| Submission | 5-10 minutes |
| **Total** | **~2.5-3.5 hours** |

---

## All Commands (Copy-Paste Ready)

### Check training progress:
```bash
tail -100 /Users/thomasmyles/.cursor/projects/Users-thomasmyles-dev-wunder-predictorium/terminals/8.txt
```

### After training completes:
```bash
cd /Users/thomasmyles/dev/wunder-predictorium
source venv/bin/activate

# Validate
python src/evaluation/validate_v2.py

# If validation > baseline, export to ONNX
python src/export/export_v2_to_onnx.py

# Create submission package
mkdir -p submissions/v2_weighted_loss
cp submissions/v0_baseline/solution.py submissions/v2_weighted_loss/solution.py
# (manually update solution.py to use v2_model.onnx)

# Package for submission
cd submissions/v2_weighted_loss
zip -r v2_submission.zip solution.py v2_model.onnx
ls -lh v2_submission.zip

# Verify package
unzip -l v2_submission.zip
```

---

## Lessons from v1

❌ **What went wrong:**
- Increased model complexity (256 hidden, 2 layers, heavy dropout)
- Model overfitted (train loss 0.93, val loss 2.44)
- Validation score dropped from 0.2595 → 0.1788 (-31%)

✅ **What we learned:**
- Start simple, add complexity only if needed
- Change ONE thing at a time
- Monitor train/val gap for overfitting

✅ **v2 approach:**
- Keep baseline's simple architecture
- Only change: loss function (MSE → Weighted MSE)
- This way we know if improvement comes from loss, not architecture

