"""
Validation script for v1 model

Usage:
    cd /Users/thomasmyles/dev/wunder-predictorium
    source venv/bin/activate
    python src/evaluation/validate_v1.py
"""

import os
import sys
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

# Add paths
sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))
sys.path.append('wnn_predictorium_starterpack')

from src.models.improved_gru import ImprovedGRU
from wnn_predictorium_starterpack.utils import DataPoint, ScorerStepByStep


class V1PredictionModel:
    """Wrapper to use trained v1 model with the official scorer."""
    
    def __init__(self, model_path='models/v1_improved_gru.pt'):
        self.device = torch.device('cpu')
        self.window_size = 100
        self.current_seq_ix = None
        self.sequence_history = []
        
        # Load model
        print(f"Loading model from {model_path}...")
        checkpoint = torch.load(model_path, map_location=self.device)
        
        self.model = ImprovedGRU()
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.to(self.device)
        self.model.eval()
        
        print(f"✓ Model loaded")
        print(f"  Epoch: {checkpoint['epoch']}")
        print(f"  Train loss: {checkpoint['train_loss']:.6f}")
        print(f"  Valid loss: {checkpoint['val_loss']:.6f}")
        print()
    
    def predict(self, data_point: DataPoint) -> np.ndarray:
        """Generate prediction for a single data point."""
        # Reset on new sequence
        if self.current_seq_ix != data_point.seq_ix:
            self.current_seq_ix = data_point.seq_ix
            self.sequence_history = []
        
        # Update history
        self.sequence_history.append(data_point.state.copy())
        
        # Return None if no prediction needed
        if not data_point.need_prediction:
            return None
        
        # Get last 100 steps
        history_window = self.sequence_history[-self.window_size:]
        
        # Pad if needed (shouldn't happen)
        if len(history_window) < self.window_size:
            padding = [np.zeros_like(history_window[0])] * (self.window_size - len(history_window))
            history_window = padding + history_window
        
        # Convert to tensor
        x = torch.FloatTensor(history_window).unsqueeze(0)  # (1, 100, 32)
        x = x.to(self.device)
        
        # Predict
        with torch.no_grad():
            pred = self.model(x)
        
        return pred.cpu().numpy()[0]  # (2,)


def main():
    VALID_PATH = 'wnn_predictorium_starterpack/datasets/valid.parquet'
    MODEL_PATH = 'models/v1_improved_gru.pt'
    
    if not os.path.exists(MODEL_PATH):
        print(f"Error: Model not found at {MODEL_PATH}")
        print("Run training first: python src/training/train_v1.py")
        return
    
    print("=" * 60)
    print("v1 Model Validation")
    print("=" * 60)
    print()
    
    # Create model
    model = V1PredictionModel(MODEL_PATH)
    
    # Create scorer
    scorer = ScorerStepByStep(VALID_PATH)
    
    # Score
    print("Running validation...")
    results = scorer.score(model)
    
    print("\n" + "=" * 60)
    print("VALIDATION RESULTS")
    print("=" * 60)
    print(f"Mean Weighted Pearson: {results['weighted_pearson']:.6f}")
    for target, score in results.items():
        if target != 'weighted_pearson':
            print(f"  {target}: {score:.6f}")
    print("=" * 60)
    print()
    
    # Compare to baseline
    baseline_val = 0.2595
    improvement = results['weighted_pearson'] - baseline_val
    print(f"Baseline (v0) validation: {baseline_val:.6f}")
    print(f"v1 validation: {results['weighted_pearson']:.6f}")
    print(f"Change: {improvement:+.6f} ({improvement/baseline_val*100:+.2f}%)")
    print()
    
    if improvement > 0:
        print("✓ v1 improves over baseline! Continue to ONNX export.")
    elif improvement > -0.005:
        print("≈ v1 similar to baseline. Small difference, worth testing on leaderboard.")
    else:
        print("✗ v1 worse than baseline. Consider different approach.")


if __name__ == "__main__":
    main()


