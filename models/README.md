# Models Directory

This directory contains our **custom-trained** model files (.pt PyTorch checkpoints).

## Why v0 is not here

`v0` is the **provided baseline** solution from the starter pack. It lives in `submissions/v0_baseline/` since we didn't train it - we just copied and submitted it.

## What's here

- `v1_improved_gru.pt` - Our first custom model (larger GRU, 2 layers, dropout)
- `v2_*.pt` - Future iterations...
- `v3_*.pt` - etc.

## Workflow

1. Train model → saves `.pt` file here
2. Export to ONNX → goes to `submissions/vN/model.onnx`
3. Package with `solution.py` → submit to leaderboard


