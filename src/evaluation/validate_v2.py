"""
Validate v2 model on validation set
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

import torch
import pandas as pd
import numpy as np
from tqdm import tqdm
from scipy.stats import pearsonr

from src.models.weighted_gru import WeightedGRU

def weighted_pearson(pred, target):
    """Calculate WPCC (Weighted Pearson Correlation Coefficient)"""
    t0_corr, _ = pearsonr(pred[:, 0], target[:, 0])
    t1_corr, _ = pearsonr(pred[:, 1], target[:, 1])
    wpcc = 0.8 * t0_corr + 0.2 * t1_corr
    return wpcc, t0_corr, t1_corr


def main():
    print("="*60)
    print("v2 Model Validation")
    print("="*60)
    print()
    
    # Load model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = WeightedGRU().to(device)
    
    checkpoint_path = 'models/v2_weighted_gru.pt'
    print(f"Loading model from {checkpoint_path}...")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    print("✓ Model loaded")
    print(f"  Epoch: {checkpoint['epoch']}")
    print(f"  Train loss: {checkpoint['train_loss']:.6f}")
    print(f"  Valid loss: {checkpoint['val_loss']:.6f}")
    print()
    
    # Load validation data
    print("Running validation...")
    df = pd.read_parquet('wnn_predictorium_starterpack/datasets/valid.parquet')
    
    # All 32 features in correct order
    feature_cols = (
        [f'p{i}' for i in range(12)] +  # 12 price features
        [f'v{i}' for i in range(12)] +  # 12 volume features
        [f'dp{i}' for i in range(4)] +  # 4 trade price features
        [f'dv{i}' for i in range(4)]    # 4 trade volume features
    )
    
    all_predictions = []
    all_targets = []
    
    # Process by sequence
    for seq_ix in tqdm(df['seq_ix'].unique()):
        seq_df = df[df['seq_ix'] == seq_ix].sort_values('step_in_seq')
        states = seq_df[feature_cols].values
        targets = seq_df[['t0', 't1']].values
        needs_pred = seq_df['need_prediction'].values
        
        history = []
        
        for i in range(len(states)):
            history.append(states[i])
            
            if needs_pred[i] and len(history) >= 100:
                # Take last 100 steps
                history_window = history[-100:]
                x = torch.FloatTensor(history_window).unsqueeze(0)  # (1, 100, 32)
                x = x.to(device)
                
                with torch.no_grad():
                    pred = model(x).cpu().numpy()[0]
                
                all_predictions.append(pred)
                all_targets.append(targets[i])
    
    # Calculate metrics
    predictions = np.array(all_predictions)
    targets = np.array(all_targets)
    
    wpcc, t0_corr, t1_corr = weighted_pearson(predictions, targets)
    
    print()
    print("="*60)
    print("VALIDATION RESULTS")
    print("="*60)
    print(f"Mean Weighted Pearson: {wpcc:.6f}")
    print(f"  t0: {t0_corr:.6f}")
    print(f"  t1: {t1_corr:.6f}")
    print("="*60)
    print()
    
    # Compare to baseline
    baseline_score = 0.2595
    delta = wpcc - baseline_score
    pct_change = (delta / baseline_score) * 100
    
    print(f"Baseline (v0) validation: {baseline_score:.6f}")
    print(f"v2 validation: {wpcc:.6f}")
    print(f"Change: {delta:+.6f} ({pct_change:+.2f}%)")
    print()
    
    if wpcc > baseline_score:
        print("✓ v2 better than baseline. Proceed to ONNX export!")
    else:
        print("✗ v2 worse than baseline. Consider different approach.")


if __name__ == '__main__':
    main()

