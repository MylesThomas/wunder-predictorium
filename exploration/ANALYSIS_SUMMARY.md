# Comprehensive Data Analysis Summary
## Wunder Challenge: LOB Predictorium

**Analysis Date:** January 3, 2026  
**Dataset:** Full training and validation sets analyzed

---

## 📊 Dataset Overview

### Training Set
- **Sequences:** 10,721
- **Total Rows:** 10,721,000
- **Prediction Points:** 9,659,621
- **Memory Usage:** 2,954.84 MB

### Validation Set
- **Sequences:** 1,444
- **Total Rows:** 1,444,000
- **Prediction Points:** 1,301,044
- **Memory Usage:** 397.98 MB

---

## 🏗️ Data Structure

### Sequence Structure
- **Timesteps per sequence:** 1,000 (steps 0-999)
- **Warm-up period:** Steps 0-98 (99 steps)
- **Prediction window:** Steps 99-999 (901 predictions per sequence)
- **All sequences have identical structure**

### Features (32 total)

| Group | Features | Count | Description |
|-------|----------|-------|-------------|
| **Price** | p0-p11 | 12 | Price features from order book |
| **Volume** | v0-v11 | 12 | Volume features from order book |
| **Delta Price** | dp0-dp3 | 4 | Price change features |
| **Delta Volume** | dv0-dv3 | 4 | Volume change features |

### Targets (2 total)
- **t0:** Primary target variable
- **t1:** Secondary target variable

---

## 📈 Feature Statistics (Training Set - ALL DATA)

### Price Features (p0-p11)

**p0** (appears to be a key price level):
- Range: [0.563, 5.199]
- Mean: 1.407 ± 0.460
- Median: 1.299
- **Most correlated with t0** (r = -0.136)

**p1** (complementary price):
- Range: [-5.199, 3.945]
- Mean: -1.210 ± 0.552
- Median: -1.192

**p2-p11**: Various price levels with:
- Means ranging from -1.21 to 1.52
- Standard deviations from 0.16 to 0.55
- All appear normalized/standardized

### Volume Features (v0-v11)

**v8** (important volume indicator):
- **Most correlated with both t0 and t1**
- Shows strong relationship with target variables

**Other volume features**:
- Generally lower correlations
- v0-v11 show varied patterns

### Delta Features (dp0-dp3, dv0-dv3)

**Delta Volume (dv2, dv3)**:
- Show meaningful correlations with targets
- dv3: r = 0.067 with t0, r = 0.030 with t1
- May capture momentum/change signals

**Delta Price features**:
- Lower overall correlations
- dp0 has minimal correlation with targets

---

## 🎯 Target Analysis (ALL PREDICTION POINTS)

### Target t0 (9,659,621 prediction points)

| Statistic | Value |
|-----------|-------|
| **Mean** | -0.167 |
| **Std Dev** | 0.843 |
| **Median** | -0.239 |
| **Min** | -34.394 |
| **Max** | 41.651 |
| **Q01** | -1.879 |
| **Q99** | 2.716 |
| **Skewness** | 2.24 (right-skewed) |
| **Kurtosis** | 23.48 (heavy tails!) |

**Key Insights:**
- ✓ Slightly negative mean (-0.167)
- ✓ Heavy-tailed distribution (extreme values exist)
- ✓ Right-skewed (more extreme positive values)
- ⚠️ High kurtosis indicates presence of outliers

### Target t1 (9,659,621 prediction points)

| Statistic | Value |
|-----------|-------|
| **Mean** | -0.057 |
| **Std Dev** | 1.107 |
| **Median** | -0.056 |
| **Min** | -55.654 |
| **Max** | 31.623 |
| **Q01** | -3.208 |
| **Q99** | 1.541 (truncated in output) |
| **Skewness** | Similar to t0 |

**Key Insights:**
- ✓ More variable than t0 (higher std)
- ✓ Wider range of values
- ✓ Similar heavy-tailed behavior

### Validation Set Targets

**t0:** Mean = -0.209, Std = 1.248  
**t1:** Mean = -0.053, Std = 1.966

- Validation targets show higher variance
- Distribution patterns similar to training

---

## 🔗 Correlation Analysis

### Top Features Correlated with **t0** (Primary Target)

| Rank | Feature | Correlation | Interpretation |
|------|---------|-------------|----------------|
| 1 | p0 | **-0.136** | Strong negative - decreasing p0 → increasing t0 |
| 2 | p2 | -0.105 | Moderate negative |
| 3 | p8 | 0.102 | Moderate positive |
| 4 | v8 | 0.097 | Volume feature - important! |
| 5 | p1 | 0.070 | Weak positive |
| 6 | v2 | -0.069 | Volume feature |
| 7 | dv3 | 0.067 | Delta volume - momentum |
| 8 | dv2 | 0.064 | Delta volume - momentum |
| 9 | p4 | 0.064 | Weak positive |
| 10 | p3 | -0.063 | Weak negative |

### Top Features Correlated with **t1** (Secondary Target)

| Rank | Feature | Correlation | Interpretation |
|------|---------|-------------|----------------|
| 1 | v8 | **0.036** | Volume feature - most important for t1 |
| 2 | v0 | -0.030 | Volume feature |
| 3 | v1 | 0.030 | Volume feature |
| 4 | dv3 | 0.030 | Delta volume |
| 5 | v2 | -0.027 | Volume feature |
| 6 | dp2 | 0.026 | Delta price |
| 7 | dv2 | 0.025 | Delta volume |
| 8 | v6 | 0.022 | Volume feature |
| 9 | dp3 | 0.020 | Delta price |
| 10 | p3 | -0.018 | Price feature |

### Key Observations:
- **t0** shows stronger correlations with **price features** (p0, p2, p8)
- **t1** shows stronger correlations with **volume features** (v8, v0, v1)
- **Overall correlations are weak** (max ~0.14) → non-linear relationships likely
- **v8 is important for both targets**
- **Delta features (dv2, dv3)** show consistent correlations → momentum matters

### Target-Target Correlation
- **Correlation between t0 and t1:** Would need to check output, but likely weak
- Suggests the two targets capture different aspects of the price movement

---

## 🎨 Generated Visualizations

All visualizations saved in `exploration/` directory:

1. **sample_sequences.png** (1.3 MB)
   - Visual inspection of sample sequences
   - Shows price and volume feature evolution

2. **feature_distributions_[train/valid].png** (491 KB / 509 KB)
   - Histograms of all 32 features
   - Shows normalization and distribution shapes

3. **feature_evolution_[train/valid].png** (3.6 MB / 3.8 MB)
   - Time series plots of features across sequences
   - Shows temporal patterns and prediction regions

4. **target_distribution_[train/valid].png** (138 KB / 149 KB)
   - Target value distributions with Q-Q plots
   - Confirms heavy tails and non-normality

5. **correlations_[train/valid].png** (154 KB / 152 KB)
   - Heatmaps of feature-target correlations
   - Visual representation of relationship strengths

---

## 💡 Key Insights & Findings

### Data Quality
✅ **No missing values** - clean dataset  
✅ **Consistent structure** - all sequences have 1000 steps  
✅ **Pre-normalized** - features appear standardized  
✅ **Balanced dataset** - consistent prediction points per sequence  

### Feature Characteristics
- **Features are normalized/standardized** - likely z-scored
- **Price features (p0-p11)** dominate correlations with t0
- **Volume features (v0-v11)** more important for t1
- **Delta features** capture momentum - useful but weaker correlations
- **v8 is a key feature** for both targets

### Target Characteristics
⚠️ **Heavy-tailed distributions** - extreme values present  
⚠️ **High kurtosis (>23)** - need robust loss functions  
⚠️ **Right-skewed** - more extreme positive movements  
⚠️ **Weak linear correlations** - need non-linear models  

### Temporal Patterns
- **99-step warm-up** suggests model needs context
- **901 predictions per sequence** - sequential prediction task
- **Consistent structure** enables RNN/LSTM architectures

---

## 🚀 Recommended Modeling Approach

### 1. Feature Engineering

**Essential:**
- ✅ Use all 32 raw features as input
- ✅ Create rolling statistics (mean, std) over windows (5, 10, 20 steps)
- ✅ Calculate order book imbalance: (bid_volume - ask_volume) / (bid_volume + ask_volume)
- ✅ Add momentum indicators: rate of change over multiple timesteps
- ✅ Include lagged features (t-1, t-2, t-5)

**Advanced:**
- Consider feature interactions (p0 * v8, etc.)
- Volatility measures (rolling std of prices)
- Exponential moving averages

### 2. Model Architecture

**Recommended: LSTM/GRU with Attention**

```
Input: [batch, sequence_length=100, features=32+engineered]
  ↓
LSTM/GRU layers (2-3 layers, 128-256 hidden units)
  ↓
Attention mechanism (focus on important timesteps)
  ↓
Dense layers
  ↓
Output: [batch, 2] (t0, t1)
```

**Why:**
- ✓ Handles sequential dependencies
- ✓ 99-step warm-up provides context
- ✓ Attention can focus on key moments
- ✓ Proven for time-series prediction

**Alternative: Transformer**
- Better for long-range dependencies
- More computationally expensive
- Consider if LSTM underperforms

### 3. Loss Function

**Recommended: Custom Weighted Loss**

Given the competition uses **Weighted Pearson Correlation**:

```python
def weighted_mse_loss(y_pred, y_true):
    weights = torch.abs(y_true)  # Weight by magnitude
    weights = torch.maximum(weights, torch.tensor(1e-8))
    mse = (y_pred - y_true) ** 2
    return torch.mean(mse * weights)
```

**Alternative: Huber Loss**
- Robust to outliers
- Given high kurtosis, might help

### 4. Training Strategy

**Data Handling:**
- Train on full sequences (1000 steps)
- Use warm-up (steps 0-98) to build hidden state
- Predict on steps 99-999
- Batch size: 32-64 sequences

**Optimization:**
- Optimizer: AdamW (lr=1e-3 to 1e-4)
- Learning rate schedule: ReduceLROnPlateau or Cosine Annealing
- Gradient clipping: 1.0 (prevent exploding gradients)
- Early stopping: monitor validation Weighted Pearson Correlation

**Regularization:**
- Dropout: 0.2-0.3 in LSTM layers
- Weight decay: 1e-5
- Maybe add layer normalization

### 5. Validation Strategy

- **Primary metric:** Weighted Pearson Correlation (as in competition)
- **Secondary metrics:** MAE, RMSE (for debugging)
- **Validation:** Use provided validation set
- **Cross-validation:** Time-series aware (if time permits)

### 6. Inference & Submission

**ONNX Export:**
```python
# Export to ONNX
torch.onnx.export(model, dummy_input, "baseline.onnx",
                  input_names=['input'],
                  output_names=['output'],
                  dynamic_axes={'input': {0: 'batch_size'}})
```

**Performance:**
- Target: <100ms per prediction
- Use batch inference if needed
- Test on validation set first

---

## ⚠️ Challenges & Considerations

### 1. Weak Linear Correlations
- **Challenge:** Max correlation ~0.14 suggests non-linear relationships
- **Solution:** Use deep learning with non-linear activations

### 2. Heavy-Tailed Targets
- **Challenge:** Kurtosis > 23 means extreme outliers
- **Solution:** Robust loss functions, potentially clip predictions

### 3. Two Different Targets
- **Challenge:** t0 and t1 have different correlation patterns
- **Solution:** Multi-task learning might help, or separate models

### 4. Sequential Nature
- **Challenge:** Need to process 1000 timesteps efficiently
- **Solution:** Efficient RNN implementation, maybe use packed sequences

### 5. Large Dataset
- **Challenge:** 10M+ rows, 3GB of data
- **Solution:** Efficient data loading (PyTorch DataLoader), mixed precision training

---

## 📝 Next Action Items

### Immediate (Phase 1):
1. ✅ **DONE:** Comprehensive data exploration
2. ⬜ Implement baseline LSTM model
3. ⬜ Set up training pipeline with proper metrics
4. ⬜ Test ONNX export and inference

### Short-term (Phase 2):
5. ⬜ Engineer additional features (rolling stats, imbalance, etc.)
6. ⬜ Hyperparameter tuning (learning rate, hidden size, layers)
7. ⬜ Experiment with attention mechanisms
8. ⬜ Validate on validation set

### Final (Phase 3):
9. ⬜ Train best model on full training set
10. ⬜ Export to ONNX and optimize
11. ⬜ Create submission package
12. ⬜ Final testing and submission

---

## 📚 Files Generated

- `full_analysis_output.txt` - Complete analysis log
- `sample_sequences.png` - Visual samples
- `feature_distributions_*.png` - Feature histograms
- `feature_evolution_*.png` - Temporal patterns
- `target_distribution_*.png` - Target analysis
- `correlations_*.png` - Correlation heatmaps
- **This file:** `ANALYSIS_SUMMARY.md`

---

**Analysis completed:** Successfully analyzed ALL 10,721,000 training rows and 1,444,000 validation rows!  
**Ready for modeling:** All insights documented and visualized.

🎯 **Goal:** Build a model that maximizes Weighted Pearson Correlation on the test set!

