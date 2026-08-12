# Wunder Challenge: LOB Predictorium

My submission for the Wunder Neural Network Predictorium competition - predicting limit order book (LOB) mid-prices.

## 📚 Documentation

- **[Quick Start Guide](docs/QUICK_START.md)** - Get started in minutes
- **[Data Overview](docs/DATA.md)** - Features, sequences, and evaluation
- **[Submission Guide](docs/SUBMISSION_GUIDE.md)** - Technical requirements
- **[Competition Rules](docs/RULES.md)** - Fair play guidelines
- **[Timeline](docs/TIMELINE.md)** - Key dates and deadlines
- **[Prizes](docs/PRIZES.md)** - $13,600 prize pool
- **[FAQ](docs/FAQ.md)** - Common questions
- **[Get Help](docs/GET_HELP.md)** - Support channels

## Quick Setup

```bash
# Install dependencies with uv (fast!)
uv sync

# Or with pip
pip install -e .

# Explore the data
cd exploration
python data_exploration.py
```

## Project Structure

```
.
├── docs/                           # Competition documentation
├── exploration/                    # Data exploration scripts
│   └── ANALYSIS_SUMMARY.md        # Key insights from data analysis
├── models/                         # Trained PyTorch models (.pt files)
│   ├── v1_improved_gru.pt         # v1: Bigger architecture (failed)
│   ├── v2_weighted_gru.pt         # v2: Baseline + weighted loss (training...)
│   └── v3_features.pt             # v3: v2 + features (planned)
├── notebooks/                      # Jupyter notebooks
├── src/                           # Source code for solutions
│   ├── models/                    # Model architectures
│   │   └── improved_gru.py        # v1 GRU model
│   ├── training/                  # Training scripts
│   │   └── train_v1.py            # v1 training
│   └── wunder_predictorium/
│       └── __init__.py
├── submissions/                    # Submission packages
│   ├── v0_baseline/               # Baseline (provided example)
│   │   ├── solution.py
│   │   ├── baseline.onnx          # 273 KB
│   │   └── v0_submission.zip      # Score: 0.2761, Rank: 27
│   ├── v1_improved_arch/          # Bigger architecture (in progress)
│   │   ├── solution.py
│   │   ├── v1_model.onnx
│   │   └── v1_submission.zip
│   ├── v2_weighted_loss/          # v1 + weighted loss (planned)
│   └── v3_features/               # v2 + feature engineering (planned)
├── wnn_predictorium_starterpack/  # Official starter pack
│   ├── datasets/                  # Training and validation data
│   ├── example_solution/          # Baseline solution
│   └── utils.py                   # Helper functions
├── daily_log.md                   # Progress tracking
├── pyproject.toml                 # Project dependencies (uv)
└── .python-version                # Python version (3.11)
```

## Resources

- 🏆 [Competition Website](https://predictorium.wundernn.io/)
- 📊 [Leaderboard](https://predictorium.wundernn.io/leaderboard)
- 💬 [Discord Community](https://predictorium.wundernn.io/discord)
- 📖 [Documentation](https://predictorium.wundernn.io/docs/)

## Development Workflow

### Current Progress (as of Jan 8, 2026)
- **v0 (Baseline):** 0.2761 test score, Rank 27/2846 ✅
- **v1 (Bigger Arch):** 0.1788 local val - FAILED (overfitting) ❌
- **v2 (Weighted Loss):** 0.0908 local val - FAILED (weighted loss backfired) ❌
- **v3:** Planning next approach... 🤔

### Workflow
1. **Explore**: Completed - see `exploration/ANALYSIS_SUMMARY.md`
2. **Develop**: Models in `src/models/`, train with `src/training/`
3. **Train**: PyTorch models saved to `models/`
4. **Validate**: Test on validation set
5. **Export**: Convert to ONNX for fast inference
6. **Submit**: Package in `submissions/vX/` and upload to leaderboard
7. **Iterate**: Analyze results, make improvements

### Model Versions

| Version | Changes | Local Val | Test Score | Rank | Status |
|---------|---------|-----------|------------|------|--------|
| v0 | Baseline GRU (64h, 1L) | **0.2595** | **0.2761** | **27** | ✅ Submitted |
| v1 | Bigger arch (256h, 2L, dropout) | 0.1788 | ❌ | - | ❌ Failed (overfitting) |
| v2 | Baseline + weighted loss | 0.0908 | ❌ | - | ❌ Failed (wrong loss fn) |
| v3 | TBD | - | - | - | 🤔 Planning... |

**Key Finding:** Baseline is still the best after 2 failed improvement attempts!

Good luck! 🚀
