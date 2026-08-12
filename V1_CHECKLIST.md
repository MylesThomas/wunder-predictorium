# v1 Training & Submission Checklist

## Step 0: Start Training

**What:** Train the improved GRU model on the full training set

**Command:**
```bash
cd /Users/thomasmyles/dev/wunder-predictorium
source venv/bin/activate
python src/training/train_v1.py
```

**Training Configuration:**
- Device: CPU
- Model: ImprovedGRU (650,626 parameters)
- Hidden size: 256
- Layers: 2
- Dropout: 0.3 (GRU) + 0.2 (Dense)
- Batch size: 64
- Epochs: 10
- Learning rate: 0.001
- Optimizer: Adam
- Loss: MSE

**Expected Time:**
- Per epoch: ~10-15 minutes (CPU)
- Total: ~100-150 minutes (1.5-2.5 hours)

**Training Progress:**
```
Epoch 1/10
--------------------------------------------------
Training: 100% |████████████| 168/168
Train Loss: 0.XXXXX
Valid Loss: 0.XXXXX
✓ Saved best model (val_loss: 0.XXXXX)
```

**Output:**
- Best model saved to: `models/v1_improved_gru.pt`
- Model is saved whenever validation loss improves
- Training can be stopped early with Ctrl+C (will use best model so far)

**Check Training Status:**
```bash
# Monitor training output
tail -f ~/.cursor/projects/Users-thomasmyles-dev-wunder-predictorium/terminals/5.txt

# Check if model exists
ls -lh models/v1_improved_gru.pt
```

---

## When Training Completes

Training will save the best model to: `models/v1_improved_gru.pt`

---

## Step 1: Validate on Test Set

**What:** Test the trained model on validation.parquet using the official scorer

**Command:**
```bash
cd /Users/thomasmyles/dev/wunder-predictorium
source venv/bin/activate
python src/evaluation/validate_v1.py
```

**This will output:**
```
Mean Weighted Pearson: 0.XXXX
  t0: 0.XXXX
  t1: 0.XXXX
```

**Goal:** Beat v0 validation score of 0.2595

---

## Step 2: Export to ONNX

**What:** Convert PyTorch model to ONNX for fast inference

**Command:**
```bash
cd /Users/thomasmyles/dev/wunder-predictorium
source venv/bin/activate
python src/export/export_v1_to_onnx.py
```

**This creates:**
- `submissions/v1_improved_arch/v1_model.onnx`

**File size:** Should be ~2MB (bigger than v0's 273KB due to more parameters)

---

## Step 3: Create Submission Package

**What:** Copy solution.py and ONNX model to submission folder

**Files needed:**
```
submissions/v1_improved_arch/
├── solution.py          # ONNX inference wrapper
└── v1_model.onnx        # Exported model
```

**Command:**
```bash
cd /Users/thomasmyles/dev/wunder-predictorium/submissions/v1_improved_arch
zip -r ../v1_submission.zip .
```

**This creates:**
- `submissions/v1_submission.zip`

**Verify:**
```bash
unzip -l ../v1_submission.zip
# Should show:
#   - solution.py
#   - v1_model.onnx
```

---

## Step 4: Upload to Leaderboard

**Where:** https://predictorium.wundernn.io/submit

**Steps:**
1. Go to submit page
2. Click "Choose file" or drag-and-drop
3. Select: `submissions/v1_submission.zip`
4. Wait for upload (file size check)
5. Submission starts processing

**Expected:**
- Processing time: 10-90 minutes
- You'll see status on leaderboard

---

## Step 5: Document Results

**Update daily_log.md with:**
```markdown
### v1 Results
- **Local validation:** 0.XXXX
- **Test (leaderboard):** 0.XXXX (after ~30 min)
- **Rank:** XX/2846
- **Change from v0:** +0.XXXX
```

**Update Score Tracking Table:**
```
| v1 | Big GRU+dropout | MSE | 32 raw | 0.XXXX | 0.XXXX | XX | +0.XXXX |
```

---

## Quick Reference

**All commands in sequence:**
```bash
# 1. Check training finished
ls -lh models/v1_improved_gru.pt

# 2. Validate
python src/evaluation/validate_v1.py

# 3. Export to ONNX
python src/export/export_v1_to_onnx.py

# 4. Package submission
cd submissions/v1_improved_arch
zip -r ../v1_submission.zip .
cd ../..

# 5. Verify package
unzip -l submissions/v1_submission.zip

# 6. Upload at: https://predictorium.wundernn.io/submit
```

---

## Next Steps (After v1 Results)

**If v1 improves over v0:**
- Continue to v2: Add weighted loss
- Path: Bigger model helps! ✅

**If v1 same/worse than v0:**
- Rethink architecture
- Maybe simpler is better?
- Try different approach

---

## All Commands (Copy-Paste Ready)

```bash
# ============================================
# STEP 0: START TRAINING
# ============================================

cd /Users/thomasmyles/dev/wunder-predictorium
source venv/bin/activate
python src/training/train_v1.py

# Monitor training (in another terminal)
tail -f ~/.cursor/projects/Users-thomasmyles-dev-wunder-predictorium/terminals/5.txt

# Check if training is done
ls -lh models/v1_improved_gru.pt


# ============================================
# STEP 1: VALIDATE ON TEST SET
# ============================================

cd /Users/thomasmyles/dev/wunder-predictorium
source venv/bin/activate
python src/evaluation/validate_v1.py

# Expected output:
# Mean Weighted Pearson: 0.XXXX
#   t0: 0.XXXX
#   t1: 0.XXXX


# ============================================
# STEP 2: EXPORT TO ONNX
# ============================================

cd /Users/thomasmyles/dev/wunder-predictorium
source venv/bin/activate
python src/export/export_v1_to_onnx.py

# Check ONNX file created
ls -lh submissions/v1_improved_arch/v1_model.onnx


# ============================================
# STEP 3: CREATE SUBMISSION PACKAGE
# ============================================

cd /Users/thomasmyles/dev/wunder-predictorium/submissions/v1_improved_arch
zip -r ../v1_submission.zip .

# Verify package contents
unzip -l ../v1_submission.zip
# Should show:
#   - solution.py
#   - v1_model.onnx

# Check file size
ls -lh ../v1_submission.zip


# ============================================
# STEP 4: UPLOAD TO LEADERBOARD
# ============================================

# Go to: https://predictorium.wundernn.io/submit
# Upload: submissions/v1_submission.zip
# Wait for results (10-90 minutes)


# ============================================
# STEP 5: DOCUMENT RESULTS
# ============================================

# Update daily_log.md with results
# Update README.md model versions table
# Commit and push to GitHub

cd /Users/thomasmyles/dev/wunder-predictorium
git add daily_log.md README.md models/ submissions/v1_improved_arch/
git commit -m "v1: Improved architecture submission

- Hidden size: 256, Layers: 2, Dropout: 0.3
- Local validation: 0.XXXX
- Leaderboard score: 0.XXXX (rank XX)"
git push


# ============================================
# TROUBLESHOOTING
# ============================================

# If training fails, check:
ls -la models/
cat ~/.cursor/projects/Users-thomasmyles-dev-wunder-predictorium/terminals/5.txt

# If ONNX export fails, check PyTorch model exists:
ls -lh models/v1_improved_gru.pt

# If validation fails, check dataset paths:
ls -lh wnn_predictorium_starterpack/datasets/valid.parquet

# If zip fails, check files exist:
ls -la submissions/v1_improved_arch/

# Test ONNX model locally before submitting:
python -c "import onnxruntime as ort; sess = ort.InferenceSession('submissions/v1_improved_arch/v1_model.onnx'); print('✓ ONNX model loads successfully')"
```

---

## Files Created

I'll create these scripts for you:
- `src/evaluation/validate_v1.py` - Validation script
- `src/export/export_v1_to_onnx.py` - ONNX export script
- `submissions/v1_improved_arch/solution.py` - Inference wrapper

Ready when training completes! 🚀

