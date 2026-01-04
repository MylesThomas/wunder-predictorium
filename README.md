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
│   ├── QUICK_START.md             # Getting started guide
│   ├── DATA.md                    # Data overview and features
│   ├── SUBMISSION_GUIDE.md        # Submission requirements
│   ├── RULES.md                   # Competition rules
│   ├── TIMELINE.md                # Key dates
│   ├── PRIZES.md                  # Prize information
│   ├── FAQ.md                     # Frequently asked questions
│   └── GET_HELP.md                # Support channels
├── exploration/                    # Data exploration scripts
│   └── data_exploration.py        # Analyze train/valid datasets
├── models/                         # Trained models
├── notebooks/                      # Jupyter notebooks
├── src/                           # Source code for solution
├── submissions/                    # Submission files
├── wnn_predictorium_starterpack/  # Official starter pack
│   ├── datasets/                  # Training and validation data
│   ├── example_solution/          # Baseline solution
│   └── utils.py                   # Helper functions
├── pyproject.toml                 # Project dependencies (uv)
└── .python-version                # Python version (3.11)
```

## Resources

- 🏆 [Competition Website](https://predictorium.wundernn.io/)
- 📊 [Leaderboard](https://predictorium.wundernn.io/leaderboard)
- 💬 [Discord Community](https://predictorium.wundernn.io/discord)
- 📖 [Documentation](https://predictorium.wundernn.io/docs/)

## Development Workflow

1. **Explore**: Run `exploration/data_exploration.py` to understand the data
2. **Develop**: Create your model in `src/`
3. **Train**: Use `wnn_predictorium_starterpack/datasets/train.parquet`
4. **Validate**: Test on `wnn_predictorium_starterpack/datasets/valid.parquet`
5. **Export**: Convert to ONNX for fast inference
6. **Submit**: Zip your `solution.py` and model files

Good luck! 🚀
