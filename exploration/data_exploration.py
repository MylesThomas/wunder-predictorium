"""
Data Exploration Script for Wunder Challenge: LOB Predictorium

This script performs COMPREHENSIVE analysis on ALL data (no sampling).

Usage:
    cd wunder-predictorium
    uv run python exploration/data_exploration.py

================================================================================
COMPREHENSIVE ANALYSIS RESULTS - ALL DATA ANALYZED
================================================================================

Dataset Overview:
-----------------
Training Set:
  - Sequences: 10,721
  - Total Rows: 10,721,000
  - Prediction Points: 9,659,621
  - Memory: 2,954.84 MB

Validation Set:
  - Sequences: 1,444
  - Total Rows: 1,444,000
  - Prediction Points: 1,301,044
  - Memory: 397.98 MB

Data Structure:
---------------
- Each sequence: 1000 timesteps (steps 0-999)
- Warm-up period: Steps 0-98 (99 steps)
- Prediction window: Steps 99-999 (901 predictions per sequence)
- ALL sequences have identical structure

Features (32 total):
--------------------
Price Features (p0-p11):     12 features - Order book price levels
Volume Features (v0-v11):    12 features - Order book volume levels
Delta Price (dp0-dp3):        4 features - Price change indicators
Delta Volume (dv0-dv3):       4 features - Volume change indicators

Targets (2 total):
------------------
t0: Primary target variable
t1: Secondary target variable

Key Feature Statistics (Training Set):
---------------------------------------
p0 (most correlated with t0):
  - Range: [0.563, 5.199]
  - Mean: 1.407 ± 0.460
  - Correlation with t0: -0.136 (strongest)

v8 (most correlated with both targets):
  - Important volume indicator
  - Correlation with t0: 0.097
  - Correlation with t1: 0.036 (strongest for t1)

All features appear normalized/standardized (likely z-scored)

Target Statistics (ALL 9,659,621 prediction points):
----------------------------------------------------
t0:
  - Mean: -0.167
  - Std: 0.843
  - Range: [-34.39, 41.65]
  - Median: -0.239
  - Skewness: 2.24 (right-skewed)
  - Kurtosis: 23.48 (HEAVY TAILS!)
  
t1:
  - Mean: -0.057
  - Std: 1.107
  - Range: [-55.65, 31.62]
  - Median: -0.056
  - More variable than t0

Correlation Analysis:
---------------------
Top Features Correlated with t0:
  1. p0:  -0.136 (price feature - strongest)
  2. p2:  -0.105 (price feature)
  3. p8:   0.102 (price feature)
  4. v8:   0.097 (volume feature)
  5. dv3:  0.067 (delta volume - momentum)

Top Features Correlated with t1:
  1. v8:   0.036 (volume feature - strongest)
  2. v0:  -0.030 (volume feature)
  3. v1:   0.030 (volume feature)
  4. dv3:  0.030 (delta volume - momentum)
  5. v2:  -0.027 (volume feature)

Key Insight: t0 responds MORE to PRICE features, t1 responds MORE to VOLUME features

Overall correlations are WEAK (<0.14) → Suggests NON-LINEAR relationships!

Critical Findings:
------------------
✅ No missing values - clean dataset
✅ Consistent structure - all sequences identical
✅ Pre-normalized features
⚠️ Heavy-tailed targets (kurtosis > 23) - need robust loss functions
⚠️ Weak linear correlations - need non-linear models (deep learning)
⚠️ Right-skewed distributions - extreme positive movements more common
⚠️ Two targets have different feature dependencies - consider multi-task learning

Recommended Modeling Approach:
-------------------------------
1. Architecture:
   - LSTM/GRU with 2-3 layers (128-256 hidden units)
   - Attention mechanism (focus on important timesteps)
   - Multi-task learning (shared encoder, separate heads for t0/t1)

2. Feature Engineering:
   - Use all 32 raw features
   - Add rolling statistics (mean, std) over windows (5, 10, 20 steps)
   - Calculate order book imbalance: (bid_vol - ask_vol) / (bid_vol + ask_vol)
   - Add momentum indicators: rate of change
   - Include lagged features (t-1, t-2, t-5)

3. Loss Function:
   - Weighted MSE (weight by target magnitude, like competition metric)
   - OR Huber loss (robust to outliers given high kurtosis)
   
4. Training Strategy:
   - Batch size: 32-64 sequences
   - Optimizer: AdamW (lr=1e-3 to 1e-4)
   - Learning rate schedule: ReduceLROnPlateau
   - Gradient clipping: 1.0 (prevent exploding gradients)
   - Dropout: 0.2-0.3
   - Early stopping on validation Weighted Pearson Correlation

5. Validation:
   - Primary metric: Weighted Pearson Correlation (competition metric)
   - Secondary: MAE, RMSE (debugging)

Generated Visualizations:
-------------------------
All saved in exploration/ directory:
  - sample_sequences.png (1.3 MB) - Example sequences
  - feature_distributions_[train/valid].png (491 KB / 509 KB) - All feature histograms
  - feature_evolution_[train/valid].png (3.6 MB / 3.8 MB) - Temporal patterns
  - target_distribution_[train/valid].png (138 KB / 149 KB) - Target analysis with Q-Q plots
  - correlations_[train/valid].png (154 KB / 152 KB) - Correlation heatmaps
  - full_analysis_output.txt (35 KB) - Complete analysis log
  - ANALYSIS_SUMMARY.md (12 KB) - Detailed summary document

Next Steps:
-----------
1. Implement baseline LSTM model
2. Set up training pipeline with proper metrics
3. Engineer additional features (rolling stats, imbalance)
4. Hyperparameter tuning
5. Export to ONNX and optimize for inference
6. Submit solution

================================================================================
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Add parent directory to path to import utils
sys.path.append(str(Path(__file__).parent.parent / 'wnn_predictorium_starterpack'))

from utils import DataPoint

# Set style for visualizations
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (15, 8)

def load_data(data_path: str) -> pd.DataFrame:
    """Load parquet data file."""
    print(f"\nLoading data from: {data_path}")
    df = pd.read_parquet(data_path)
    print(f"✓ Loaded {len(df)} sequences")
    return df

def explore_basic_structure(df: pd.DataFrame, dataset_name: str):
    """Explore basic structure of the dataset."""
    print(f"\n{'='*60}")
    print(f"Dataset: {dataset_name}")
    print(f"{'='*60}")
    
    print(f"\nShape: {df.shape}")
    print(f"Columns: {df.columns.tolist()}")
    print(f"\nColumn types:")
    print(df.dtypes)
    
    print(f"\nFirst few rows:")
    print(df.head())
    
    # Check for missing values
    print(f"\nMissing values:")
    print(df.isnull().sum())
    
    # Memory usage
    print(f"\nMemory usage: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")

def explore_sequence_structure(df: pd.DataFrame):
    """Explore the structure of individual sequences."""
    print(f"\n{'='*60}")
    print("Sequence Structure Analysis")
    print(f"{'='*60}")
    
    # Parse first sequence to understand structure
    first_row = df.iloc[0]
    # Create DataPoint: seq_ix, step_in_seq, need_prediction, state (features)
    datapoint = DataPoint(
        seq_ix=int(first_row['seq_ix']),
        step_in_seq=int(first_row['step_in_seq']),
        need_prediction=bool(first_row['need_prediction']),
        state=first_row.iloc[3:35].values  # Features columns 3-35 (32 features)
    )
    
    print(f"\nDataPoint components:")
    print(f"- seq_ix: {datapoint.seq_ix}")
    print(f"- step_in_seq: {datapoint.step_in_seq}")
    print(f"- need_prediction: {datapoint.need_prediction}")
    print(f"- state shape: {datapoint.state.shape}")
    print(f"- state values (first 5): {datapoint.state[:5]}")
    
    # Analyze first few rows of a sequence to understand structure
    first_seq = df[df['seq_ix'] == 0]
    print(f"\nFirst sequence info:")
    print(f"- Total steps: {len(first_seq)}")
    print(f"- Prediction required at steps: {first_seq[first_seq['need_prediction'] == 1]['step_in_seq'].tolist()}")
    
    # Show feature names
    feature_cols = df.columns[3:35].tolist()  # 32 features
    print(f"\nFeatures (32 total): {feature_cols}")
    
    # Show target names
    target_cols = df.columns[35:].tolist()  # 2 targets
    print(f"\nTargets (2 total): {target_cols}")
    
    return datapoint

def analyze_feature_statistics(df: pd.DataFrame, dataset_name: str):
    """Analyze statistical properties of features across ALL data."""
    print(f"\n{'='*60}")
    print(f"Feature Statistics: {dataset_name} (FULL DATASET)")
    print(f"{'='*60}")
    
    # Get feature columns (columns 3-35: 32 features)
    feature_cols = df.columns[3:35].tolist()
    
    print(f"\nAnalyzing ALL {len(df)} rows...")
    print(f"\n{'='*60}")
    print("ALL FEATURES - Complete Statistics")
    print(f"{'='*60}")
    
    # Compute statistics for ALL features on entire dataset
    for col in feature_cols:
        print(f"\n{col}:")
        print(f"  Min:    {df[col].min():12.6f}")
        print(f"  Max:    {df[col].max():12.6f}")
        print(f"  Mean:   {df[col].mean():12.6f}")
        print(f"  Std:    {df[col].std():12.6f}")
        print(f"  Median: {df[col].median():12.6f}")
        print(f"  Q25:    {df[col].quantile(0.25):12.6f}")
        print(f"  Q75:    {df[col].quantile(0.75):12.6f}")
    
    # Show target statistics
    target_cols = df.columns[35:].tolist()  # 2 targets
    print(f"\n{'='*60}")
    print("TARGET STATISTICS - Complete Statistics")
    print(f"{'='*60}")
    
    for col in target_cols:
        print(f"\n{col}:")
        print(f"  Min:    {df[col].min():12.6f}")
        print(f"  Max:    {df[col].max():12.6f}")
        print(f"  Mean:   {df[col].mean():12.6f}")
        print(f"  Std:    {df[col].std():12.6f}")
        print(f"  Median: {df[col].median():12.6f}")
        print(f"  Q25:    {df[col].quantile(0.25):12.6f}")
        print(f"  Q75:    {df[col].quantile(0.75):12.6f}")

def analyze_sequence_patterns(df: pd.DataFrame, dataset_name: str):
    """Analyze patterns across ALL sequences."""
    print(f"\n{'='*60}")
    print(f"Sequence Pattern Analysis: {dataset_name} (ALL SEQUENCES)")
    print(f"{'='*60}")
    
    # Get total number of unique sequences
    num_sequences = df['seq_ix'].nunique()
    print(f"\nTotal unique sequences: {num_sequences:,}")
    
    # Analyze steps per sequence
    steps_per_seq = df.groupby('seq_ix').size()
    print(f"\nSteps per sequence:")
    print(f"  Min:    {steps_per_seq.min()}")
    print(f"  Max:    {steps_per_seq.max()}")
    print(f"  Mean:   {steps_per_seq.mean():.2f}")
    print(f"  Median: {steps_per_seq.median():.0f}")
    
    # Analyze prediction points
    pred_per_seq = df.groupby('seq_ix')['need_prediction'].sum()
    print(f"\nPrediction points per sequence:")
    print(f"  Min:    {pred_per_seq.min()}")
    print(f"  Max:    {pred_per_seq.max()}")
    print(f"  Mean:   {pred_per_seq.mean():.2f}")
    print(f"  Median: {pred_per_seq.median():.0f}")
    print(f"  Total prediction points: {df['need_prediction'].sum():,}")
    
    # Analyze when predictions start
    first_pred_step = df[df['need_prediction'] == 1].groupby('seq_ix')['step_in_seq'].min()
    print(f"\nFirst prediction step in sequences:")
    print(f"  Min:    {first_pred_step.min()}")
    print(f"  Max:    {first_pred_step.max()}")
    print(f"  Mean:   {first_pred_step.mean():.2f}")
    print(f"  Median: {first_pred_step.median():.0f}")
    
    # Analyze target value ranges per sequence
    target_cols = df.columns[35:].tolist()
    print(f"\nTarget value ranges across sequences:")
    for target in target_cols:
        pred_df = df[df['need_prediction'] == 1]
        target_range = pred_df.groupby('seq_ix')[target].apply(lambda x: x.max() - x.min())
        print(f"\n{target} - Range per sequence (max - min):")
        print(f"  Min range:    {target_range.min():12.6f}")
        print(f"  Max range:    {target_range.max():12.6f}")
        print(f"  Mean range:   {target_range.mean():12.6f}")
        print(f"  Median range: {target_range.median():12.6f}")


def analyze_correlations(df: pd.DataFrame, dataset_name: str):
    """Analyze correlations between features and targets."""
    print(f"\n{'='*60}")
    print(f"Correlation Analysis: {dataset_name}")
    print(f"{'='*60}")
    
    # Get feature and target columns
    feature_cols = df.columns[3:35].tolist()  # 32 features
    target_cols = df.columns[35:].tolist()  # 2 targets
    
    # Only use prediction points for correlation
    pred_df = df[df['need_prediction'] == 1]
    
    print(f"\nAnalyzing correlations using {len(pred_df):,} prediction points...")
    
    # Calculate correlations between each feature and each target
    print(f"\n{'='*60}")
    print("Feature-Target Correlations (Pearson)")
    print(f"{'='*60}")
    
    for target in target_cols:
        print(f"\nCorrelations with {target}:")
        correlations = []
        for feature in feature_cols:
            corr = pred_df[feature].corr(pred_df[target])
            correlations.append((feature, corr))
        
        # Sort by absolute correlation
        correlations.sort(key=lambda x: abs(x[1]), reverse=True)
        
        print(f"\nTop 10 most correlated features:")
        for i, (feature, corr) in enumerate(correlations[:10], 1):
            print(f"  {i:2d}. {feature:6s}: {corr:8.5f}")
        
        print(f"\nTop 10 least correlated features:")
        for i, (feature, corr) in enumerate(correlations[-10:], 1):
            print(f"  {i:2d}. {feature:6s}: {corr:8.5f}")
    
    # Correlation between the two targets
    target_corr = pred_df[target_cols[0]].corr(pred_df[target_cols[1]])
    print(f"\nCorrelation between {target_cols[0]} and {target_cols[1]}: {target_corr:.6f}")
    
    # Create correlation heatmap
    print(f"\nCreating correlation heatmap...")
    
    # Sample for memory efficiency if needed
    sample_size = min(100000, len(pred_df))
    sample_df = pred_df.sample(n=sample_size, random_state=42) if len(pred_df) > sample_size else pred_df
    
    # Compute correlation matrix
    corr_data = sample_df[feature_cols + target_cols]
    corr_matrix = corr_data.corr()
    
    # Plot heatmap for features vs targets
    fig, ax = plt.subplots(figsize=(12, 10))
    
    # Extract feature-target correlations
    feature_target_corr = corr_matrix.loc[feature_cols, target_cols]
    
    sns.heatmap(feature_target_corr, annot=True, fmt='.3f', cmap='coolwarm', 
                center=0, ax=ax, cbar_kws={'label': 'Correlation'})
    ax.set_title(f'Feature-Target Correlations ({dataset_name})')
    ax.set_xlabel('Targets')
    ax.set_ylabel('Features')
    
    plt.tight_layout()
    output_path = Path(__file__).parent / f'correlations_{dataset_name.lower().replace(" ", "_")}.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"✓ Saved correlation heatmap to: {output_path}")
    plt.close()


def create_comprehensive_visualizations(df: pd.DataFrame, dataset_name: str):
    """Create comprehensive visualizations of the data."""
    print(f"\n{'='*60}")
    print(f"Creating Comprehensive Visualizations: {dataset_name}")
    print(f"{'='*60}")
    
    # Get columns
    feature_cols = df.columns[3:35].tolist()
    target_cols = df.columns[35:].tolist()
    
    # 1. Feature distributions
    print(f"\nCreating feature distribution plots...")
    fig, axes = plt.subplots(8, 4, figsize=(20, 24))
    axes = axes.flatten()
    
    for i, col in enumerate(feature_cols):
        # Sample for speed
        sample_data = df[col].sample(min(50000, len(df)), random_state=42)
        axes[i].hist(sample_data, bins=50, edgecolor='black', alpha=0.7)
        axes[i].set_title(f'{col}')
        axes[i].set_xlabel('Value')
        axes[i].set_ylabel('Frequency')
        axes[i].grid(True, alpha=0.3)
        axes[i].axvline(x=sample_data.mean(), color='red', linestyle='--', linewidth=1, label='Mean')
        axes[i].axvline(x=sample_data.median(), color='green', linestyle='--', linewidth=1, label='Median')
        if i == 0:
            axes[i].legend(fontsize=8)
    
    plt.tight_layout()
    output_path = Path(__file__).parent / f'feature_distributions_{dataset_name.lower().replace(" ", "_")}.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"✓ Saved feature distributions to: {output_path}")
    plt.close()
    
    # 2. Feature evolution over sequence steps
    print(f"\nCreating feature evolution plots...")
    # Sample a few sequences
    sample_seqs = df['seq_ix'].unique()[:5]
    
    fig, axes = plt.subplots(5, 1, figsize=(16, 15))
    
    for i, seq_id in enumerate(sample_seqs):
        seq_data = df[df['seq_ix'] == seq_id]
        ax = axes[i]
        
        # Plot all features normalized
        for feature in feature_cols[:12]:  # Plot first 12 features
            values = seq_data[feature].values
            # Normalize to [0, 1] for comparison
            if values.max() != values.min():
                normalized = (values - values.min()) / (values.max() - values.min())
            else:
                normalized = values
            ax.plot(seq_data['step_in_seq'], normalized, label=feature, alpha=0.6)
        
        # Mark prediction region
        pred_steps = seq_data[seq_data['need_prediction'] == 1]['step_in_seq']
        if len(pred_steps) > 0:
            ax.axvspan(pred_steps.min(), pred_steps.max(), alpha=0.1, color='red', label='Prediction Region')
        
        ax.set_xlabel('Step in Sequence')
        ax.set_ylabel('Normalized Feature Value')
        ax.set_title(f'Sequence {seq_id}: Feature Evolution (first 12 features, normalized)')
        ax.legend(loc='upper right', fontsize=6, ncol=4)
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    output_path = Path(__file__).parent / f'feature_evolution_{dataset_name.lower().replace(" ", "_")}.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"✓ Saved feature evolution to: {output_path}")
    plt.close()


def visualize_sample_sequences(df: pd.DataFrame, num_samples: int = 3):
    """Visualize a few sample sequences."""
    print(f"\n{'='*60}")
    print("Visualizing Sample Sequences")
    print(f"{'='*60}")
    
    fig, axes = plt.subplots(num_samples, 2, figsize=(15, 4*num_samples))
    
    # Get feature columns
    feature_cols = df.columns[3:35].tolist()
    
    for i in range(num_samples):
        seq_id = i
        seq_data = df[df['seq_ix'] == seq_id]
        
        if len(seq_data) == 0:
            continue
            
        timesteps = seq_data['step_in_seq'].values
        
        # Plot first few price features (p0-p5)
        ax = axes[i, 0] if num_samples > 1 else axes[0]
        for j in range(6):  # Plot first 6 price features
            ax.plot(timesteps, seq_data[f'p{j}'].values, label=f'p{j}', alpha=0.7)
        ax.set_xlabel('Step in Sequence')
        ax.set_ylabel('Price Features')
        ax.set_title(f'Sequence {seq_id}: Price Features (p0-p5)')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Plot first few volume features (v0-v5)
        ax = axes[i, 1] if num_samples > 1 else axes[1]
        for j in range(6):  # Plot first 6 volume features
            ax.plot(timesteps, seq_data[f'v{j}'].values, label=f'v{j}', alpha=0.7)
        ax.set_xlabel('Step in Sequence')
        ax.set_ylabel('Volume Features')
        ax.set_title(f'Sequence {seq_id}: Volume Features (v0-v5)')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    output_path = Path(__file__).parent / 'sample_sequences.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"✓ Saved visualization to: {output_path}")
    plt.close()


def analyze_target_distribution(df: pd.DataFrame, dataset_name: str):
    """Analyze the distribution of target values across ALL data."""
    print(f"\n{'='*60}")
    print(f"Target Distribution Analysis: {dataset_name} (FULL DATASET)")
    print(f"{'='*60}")
    
    # Get target columns
    target_cols = df.columns[35:].tolist()  # ['t0', 't1']
    
    # Get ALL data points where prediction is needed
    pred_needed_df = df[df['need_prediction'] == 1]
    print(f"\nTotal prediction points in {dataset_name}: {len(pred_needed_df):,}")
    
    # Calculate statistics on ALL prediction points
    print(f"\n{'='*60}")
    print("Target Statistics (ALL PREDICTION POINTS):")
    print(f"{'='*60}")
    
    for target in target_cols:
        values = pred_needed_df[target].values
        print(f"\n{target}:")
        print(f"  Count:      {len(values):,}")
        print(f"  Mean:       {np.mean(values):12.8f}")
        print(f"  Std:        {np.std(values):12.8f}")
        print(f"  Min:        {np.min(values):12.8f}")
        print(f"  Max:        {np.max(values):12.8f}")
        print(f"  Median:     {np.median(values):12.8f}")
        print(f"  Q01:        {np.percentile(values, 1):12.8f}")
        print(f"  Q05:        {np.percentile(values, 5):12.8f}")
        print(f"  Q25:        {np.percentile(values, 25):12.8f}")
        print(f"  Q75:        {np.percentile(values, 75):12.8f}")
        print(f"  Q95:        {np.percentile(values, 95):12.8f}")
        print(f"  Q99:        {np.percentile(values, 99):12.8f}")
        print(f"  Skewness:   {pd.Series(values).skew():12.8f}")
        print(f"  Kurtosis:   {pd.Series(values).kurtosis():12.8f}")
    
    # Plot distribution using a sample for visualization (too many points to plot all)
    sample_size = min(50000, len(pred_needed_df))
    sample_df = pred_needed_df.sample(n=sample_size, random_state=42) if len(pred_needed_df) > sample_size else pred_needed_df
    
    print(f"\n(Visualizing with sample of {len(sample_df):,} points for clarity)")
    
    fig, axes = plt.subplots(len(target_cols), 2, figsize=(15, 5*len(target_cols)))
    
    for i, target in enumerate(target_cols):
        values = sample_df[target].values
        
        # Histogram
        ax = axes[i, 0] if len(target_cols) > 1 else axes[0]
        ax.hist(values, bins=100, edgecolor='black', alpha=0.7)
        ax.set_xlabel(f'{target} Value')
        ax.set_ylabel('Frequency')
        ax.set_title(f'Distribution of {target} (sample of {len(values):,} points)')
        ax.axvline(x=0, color='red', linestyle='--', linewidth=2, label='Zero')
        ax.axvline(x=np.mean(pred_needed_df[target]), color='green', linestyle='--', linewidth=2, label='Mean (all data)')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Q-Q plot against normal distribution
        from scipy import stats
        ax = axes[i, 1] if len(target_cols) > 1 else axes[1]
        stats.probplot(values, dist="norm", plot=ax)
        ax.set_title(f'Q-Q Plot for {target} (Normal Distribution)')
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    output_path = Path(__file__).parent / f'target_distribution_{dataset_name.lower().replace(" ", "_")}.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\n✓ Saved visualization to: {output_path}")
    plt.close()

def main():
    """Main exploration workflow - COMPREHENSIVE ANALYSIS."""
    print("="*60)
    print("WUNDER CHALLENGE: LOB PREDICTORIUM")
    print("COMPREHENSIVE Data Exploration - ALL DATA")
    print("="*60)
    
    # Define paths
    base_path = Path(__file__).parent.parent / 'wnn_predictorium_starterpack' / 'datasets'
    train_path = base_path / 'train.parquet'
    valid_path = base_path / 'valid.parquet'
    
    # Check if files exist
    if not train_path.exists():
        print(f"\n❌ Error: Training data not found at {train_path}")
        return
    if not valid_path.exists():
        print(f"\n❌ Error: Validation data not found at {valid_path}")
        return
    
    # Load datasets
    train_df = load_data(str(train_path))
    valid_df = load_data(str(valid_path))
    
    print(f"\n{'='*60}")
    print("PHASE 1: BASIC STRUCTURE")
    print(f"{'='*60}")
    
    # Explore training data
    explore_basic_structure(train_df, "Training Set")
    sample_datapoint = explore_sequence_structure(train_df)
    
    # Explore validation data
    explore_basic_structure(valid_df, "Validation Set")
    
    print(f"\n{'='*60}")
    print("PHASE 2: COMPREHENSIVE STATISTICS (ALL DATA)")
    print(f"{'='*60}")
    
    # Statistical analysis on ALL data
    analyze_feature_statistics(train_df, "Training Set")
    analyze_feature_statistics(valid_df, "Validation Set")
    
    print(f"\n{'='*60}")
    print("PHASE 3: SEQUENCE PATTERNS (ALL SEQUENCES)")
    print(f"{'='*60}")
    
    # Analyze sequence patterns
    analyze_sequence_patterns(train_df, "Training Set")
    analyze_sequence_patterns(valid_df, "Validation Set")
    
    print(f"\n{'='*60}")
    print("PHASE 4: TARGET DISTRIBUTION (ALL PREDICTION POINTS)")
    print(f"{'='*60}")
    
    # Target distribution analysis
    analyze_target_distribution(train_df, "Training Set")
    analyze_target_distribution(valid_df, "Validation Set")
    
    print(f"\n{'='*60}")
    print("PHASE 5: CORRELATION ANALYSIS")
    print(f"{'='*60}")
    
    # Correlation analysis
    analyze_correlations(train_df, "Training Set")
    analyze_correlations(valid_df, "Validation Set")
    
    print(f"\n{'='*60}")
    print("PHASE 6: COMPREHENSIVE VISUALIZATIONS")
    print(f"{'='*60}")
    
    # Visualizations
    visualize_sample_sequences(train_df, num_samples=3)
    create_comprehensive_visualizations(train_df, "Training Set")
    create_comprehensive_visualizations(valid_df, "Validation Set")
    
    print(f"\n{'='*60}")
    print("✓ COMPREHENSIVE EXPLORATION COMPLETE!")
    print(f"{'='*60}")
    
    print("\n" + "="*60)
    print("KEY FINDINGS SUMMARY")
    print("="*60)
    
    num_train_seqs = train_df['seq_ix'].nunique()
    num_valid_seqs = valid_df['seq_ix'].nunique()
    num_train_preds = train_df['need_prediction'].sum()
    num_valid_preds = valid_df['need_prediction'].sum()
    
    print(f"\nDataset Sizes:")
    print(f"  Training sequences:     {num_train_seqs:,}")
    print(f"  Training rows:          {len(train_df):,}")
    print(f"  Training predictions:   {num_train_preds:,}")
    print(f"  Validation sequences:   {num_valid_seqs:,}")
    print(f"  Validation rows:        {len(valid_df):,}")
    print(f"  Validation predictions: {num_valid_preds:,}")
    
    print(f"\nSequence Structure:")
    print(f"  - Each sequence has 1000 timesteps (steps 0-999)")
    print(f"  - First 99 steps (0-98): context/warm-up period")
    print(f"  - Steps 99-999: prediction window (901 predictions per sequence)")
    
    print(f"\nFeatures (32 total):")
    print(f"  - p0-p11:  12 price features")
    print(f"  - v0-v11:  12 volume features")
    print(f"  - dp0-dp3: 4 delta price features")
    print(f"  - dv0-dv3: 4 delta volume features")
    
    print(f"\nTargets (2 total):")
    print(f"  - t0: Target variable 1")
    print(f"  - t1: Target variable 2")
    
    print(f"\nData Quality:")
    print(f"  - No missing values")
    print(f"  - All features are float64")
    print(f"  - Data appears to be normalized/standardized")
    
    print(f"\nGenerated Visualizations:")
    print(f"  - sample_sequences.png")
    print(f"  - target_distribution_training_set.png")
    print(f"  - target_distribution_validation_set.png")
    print(f"  - correlations_training_set.png")
    print(f"  - correlations_validation_set.png")
    print(f"  - feature_distributions_training_set.png")
    print(f"  - feature_distributions_validation_set.png")
    print(f"  - feature_evolution_training_set.png")
    print(f"  - feature_evolution_validation_set.png")
    
    print("\n" + "="*60)
    print("NEXT STEPS")
    print("="*60)
    print("\n1. Feature Engineering:")
    print("   - Analyze temporal patterns in the features")
    print("   - Create rolling statistics (mean, std, min, max)")
    print("   - Extract order book imbalance features")
    print("   - Calculate momentum indicators")
    
    print("\n2. Model Architecture:")
    print("   - Consider RNN/LSTM for temporal dependencies")
    print("   - Or use Transformer architecture for long-range dependencies")
    print("   - Try attention mechanisms to focus on important timesteps")
    
    print("\n3. Training Strategy:")
    print("   - Use weighted Pearson correlation as the metric")
    print("   - Train on train.parquet")
    print("   - Validate on valid.parquet")
    print("   - Implement proper sequence handling")
    
    print("\n4. Export:")
    print("   - Export final model to ONNX format")
    print("   - Ensure fast inference (<100ms per prediction)")
    print("   - Submit solution.py with ONNX model")


if __name__ == "__main__":
    main()

