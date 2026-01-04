# Predictorium Competition - Limit Order Book Price Prediction

**Competition URL:** https://wundernn.io/predictorium

## Overview

High-frequency trading competition hosted by Wunder Fund to predict future price movements from Limit Order Book (LOB) states. This is a real-world quantitative research problem that HFT firms solve daily.

## Timeline & Prizes

| Milestone | Date |
|-----------|------|
| Competition Start | Dec 31, 2025 |
| Submissions Close | Mar 1, 2026 |
| Final Scoring & Winners | Mar 15, 2026 |

**Prize Pool: $13,600**
- 🥇 1st: $5,000
- 🥈 2nd: $2,500
- 🥉 3rd: $1,700
- 4th: $1,300
- 5th: $1,000
- 6th: $800
- 7th: $700
- 8th: $600

## The Problem

Predict future price movements (targets t0, t1) from sequences of Limit Order Book states.

### What is a Limit Order Book?

A Limit Order Book (LOB) is the fundamental data structure in electronic trading that records all outstanding buy (bid) and sell (ask) orders for an asset at various price levels.

**Simple Example:**

```
┌─────────────────────────────────────┐
│       LIMIT ORDER BOOK (LOB)        │
├─────────────────────────────────────┤
│  ASK (Sell Orders)                  │
│  Price    Volume    (sellers)       │
│  $100.03    50      ████            │
│  $100.02   150      ████████        │
│  $100.01   200      ███████████     │
├─────────────────────────────────────┤
│  SPREAD: $0.02                      │
├─────────────────────────────────────┤
│  BID (Buy Orders)                   │
│  Price    Volume    (buyers)        │
│  $99.99    180      ██████████      │
│  $99.98    120      ███████         │
│  $99.97     80      ████            │
└─────────────────────────────────────┘
```

**Key LOB Features:**
- **Best Bid/Ask:** Highest buy price ($99.99) and lowest sell price ($100.01)
- **Spread:** Difference between best ask and best bid ($0.02)
- **Depth:** Total volume available at each price level
- **Imbalance:** Ratio of bid volume to ask volume (predicts direction)
- **Dynamics:** Orders constantly added, cancelled, and executed

**Why LOB Prediction is Hard:**
- Changes happen at millisecond timescales
- Order placement can be strategic (spoofing, iceberg orders)
- Noise-to-signal ratio is extremely high
- Regime shifts (volatile vs calm markets)
- Information asymmetry (informed vs uninformed traders)

**In This Competition:**
- Raw LOB data is preprocessed into 32 anonymized features
- Features likely include: price levels, volumes, spreads, imbalances, trade flow
- You receive sequences of LOB snapshots over time
- Goal: Use temporal patterns to predict future price movements

### Data Structure

- **Features:** N=32 anonymized features per market state
  - Prices, volumes, and trades
  - Designed to resemble production features
- **Context Window:** First 99 market states used for regime inference
- **Prediction Target:** Targets (t0, t1) for remaining states in sequence
- **Data Source:** Real market data spanning different time periods and conditions

### Key Challenges

1. **Noisy & Non-Stationary:** LOB dynamics violate standard time-series assumptions
2. **Multiple Regimes:** Different market conditions require adaptive approaches
3. **Computational Efficiency:** Must run efficiently on standard CPU (HFT constraint)
4. **Complexity:** Real-world trading problem with production-level difficulty

## Current Leaderboard (as of Jan 3, 2026)

| Rank | Name | Best Score |
|------|------|------------|
| 1 | aks | 0.2924 |
| 2 | insuperabilehart | 0.2923 |
| 3 | AmorfEvo | 0.2921 |
| 7 | 🤖 example_solution | 0.2761 |

*Top scores are clustered around 0.29, with baseline example solution at 0.2761*

## Potential Approaches

### 1. Neural Network Architectures
- **LSTMs/GRUs:** Handle sequential dependencies in LOB states
- **Transformers:** Attention mechanisms for regime detection
- **Temporal CNNs:** Efficient for time-series with fixed windows
- **Constraint:** Must be lightweight enough for CPU inference

### 2. Tree-Based Methods
- **XGBoost/LightGBM:** Fast inference, good with tabular features
- **Feature Engineering:** Rolling statistics, order imbalances, price spreads
- **Advantage:** Naturally efficient on CPU

### 3. Ensemble Strategies
- Combine neural and tree-based models
- Separate models for different market regimes
- Online learning for non-stationarity

### 4. Feature Engineering Ideas
- Order book imbalance (bid vs ask volumes)
- Price level changes and momentum
- Trade flow toxicity indicators
- Microstructure features (spread, depth)
- Rolling statistics over context window

## Technical Requirements

- **Inference Speed:** Must be production-viable on standard CPU
- **Memory:** Efficient handling of sequences
- **Robustness:** Handle different market regimes
- **Metrics:** Unknown (likely MSE, MAE, or directional accuracy)

## Project Structure (Proposed)

```
predictorium/
├── data/
│   ├── raw/              # Original competition data
│   ├── processed/        # Cleaned and engineered features
│   └── submissions/      # Submission files
├── models/
│   ├── baseline/         # Simple baseline models
│   ├── neural/           # LSTM, Transformer experiments
│   ├── tree/             # XGBoost, LightGBM
│   └── ensemble/         # Combined approaches
├── notebooks/
│   ├── EDA.ipynb        # Exploratory data analysis
│   ├── feature_engineering.ipynb
│   └── model_experiments.ipynb
├── src/
│   ├── data_loader.py
│   ├── features.py
│   ├── models.py
│   ├── train.py
│   └── predict.py
├── scripts/
│   ├── train_baseline.py
│   └── generate_submission.py
└── README.md
```

## Next Steps

1. ✅ Create project documentation
2. ⬜ Access competition data (register on wundernn.io)
3. ⬜ EDA: Understand data distribution and patterns
4. ⬜ Implement baseline solution
5. ⬜ Feature engineering pipeline
6. ⬜ Model experimentation
7. ⬜ Optimize for CPU efficiency
8. ⬜ Generate and submit predictions

## Resources

- **Competition Site:** https://wundernn.io/predictorium
- **Host:** Wunder Fund (HFT firm operating since 2014)
- **Discord:** Available on competition site for community discussion
- **Documentation:** Available on competition site

## Notes

- This is a production-adjacent problem - solutions could inform real trading strategies
- Wunder Fund has 10 years of successful trading proving this problem is solvable
- Fast feedback loops and hacker mindset emphasized
- CPU efficiency is critical (eliminates heavy deep learning approaches)
- Baseline example solution achieves 0.2761 - good starting benchmark

---

**Status:** 📋 Planning Phase  
**Last Updated:** Jan 3, 2026

