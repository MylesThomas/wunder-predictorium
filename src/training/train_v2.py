"""
Train v2: Weighted Loss GRU

Key insight from v1: Bigger model overfits badly. 
Keep baseline architecture, just fix the loss function.

Weighted MSE: t0 gets 4x weight, t1 gets 1x weight (matches WPCC formula)
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
from tqdm import tqdm

from src.models.weighted_gru import WeightedGRU

class LOBDataset(Dataset):
    def __init__(self, parquet_path, window_size=100):
        print(f"Loading data from {parquet_path}...")
        self.df = pd.read_parquet(parquet_path)
        self.window_size = window_size
        
        # Group by sequence
        self.sequences = []
        
        # All 32 features in correct order
        feature_cols = (
            [f'p{i}' for i in range(12)] +  # 12 price features
            [f'v{i}' for i in range(12)] +  # 12 volume features
            [f'dp{i}' for i in range(4)] +  # 4 trade price features
            [f'dv{i}' for i in range(4)]    # 4 trade volume features
        )
        
        for seq_ix, group in self.df.groupby('seq_ix'):
            group = group.sort_values('step_in_seq')
            states = group[feature_cols].values
            targets = group[['t0', 't1']].values
            needs_pred = group['need_prediction'].values
            
            self.sequences.append({
                'states': states,
                'targets': targets,
                'needs_pred': needs_pred
            })
        
        # Build samples
        self.samples = []
        for seq in tqdm(self.sequences, desc="Building samples"):
            states = seq['states']
            targets = seq['targets']
            needs_pred = seq['needs_pred']
            
            for i in range(self.window_size, len(states)):
                if needs_pred[i]:
                    window = states[i-self.window_size:i]
                    target = targets[i]
                    self.samples.append((window, target))
        
        print(f"Created {len(self.samples)} samples from {len(self.sequences)} sequences")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        window, target = self.samples[idx]
        return torch.FloatTensor(window), torch.FloatTensor(target)


def weighted_mse_loss(pred, target, weights=torch.tensor([4.0, 1.0])):
    """
    MSE with per-target weights
    
    pred: (batch, 2)
    target: (batch, 2)
    weights: (2,) - [4.0, 1.0] means t0 is 4x more important
    
    Returns weighted mean squared error
    """
    if pred.device != weights.device:
        weights = weights.to(pred.device)
    
    # Element-wise squared error
    sq_error = (pred - target) ** 2  # (batch, 2)
    
    # Apply weights
    weighted_error = sq_error * weights  # broadcasts to (batch, 2)
    
    # Return mean
    return weighted_error.mean()


def train_epoch(model, loader, optimizer, device, weights):
    model.train()
    total_loss = 0
    
    for x, y in tqdm(loader, desc="Training"):
        x, y = x.to(device), y.to(device)
        
        optimizer.zero_grad()
        pred = model(x)
        loss = weighted_mse_loss(pred, y, weights)
        loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        optimizer.step()
        total_loss += loss.item()
    
    return total_loss / len(loader)


def validate(model, loader, device, weights):
    model.eval()
    total_loss = 0
    
    with torch.no_grad():
        for x, y in tqdm(loader, desc="Validating"):
            x, y = x.to(device), y.to(device)
            pred = model(x)
            loss = weighted_mse_loss(pred, y, weights)
            total_loss += loss.item()
    
    return total_loss / len(loader)


def main():
    # Config
    BATCH_SIZE = 256
    EPOCHS = 10
    LR = 0.001
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Loss weights: t0 is 4x more important than t1 (matches WPCC)
    LOSS_WEIGHTS = torch.tensor([4.0, 1.0])
    
    print("="*60)
    print("Training v2: Weighted Loss GRU")
    print("="*60)
    print(f"Device: {DEVICE}")
    print(f"Loss weights: t0={LOSS_WEIGHTS[0]:.1f}, t1={LOSS_WEIGHTS[1]:.1f}")
    print()
    
    # Load data
    train_dataset = LOBDataset('wnn_predictorium_starterpack/datasets/train.parquet')
    valid_dataset = LOBDataset('wnn_predictorium_starterpack/datasets/valid.parquet')
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    valid_loader = DataLoader(valid_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    
    # Model
    model = WeightedGRU().to(DEVICE)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {total_params:,}")
    print()
    
    # Optimizer & scheduler
    optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=0.01)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=1
    )
    
    # Training loop
    best_val_loss = float('inf')
    
    for epoch in range(EPOCHS):
        print(f"\nEpoch {epoch+1}/{EPOCHS}")
        print("-" * 60)
        
        train_loss = train_epoch(model, train_loader, optimizer, DEVICE, LOSS_WEIGHTS)
        val_loss = validate(model, valid_loader, DEVICE, LOSS_WEIGHTS)
        
        print(f"Train loss: {train_loss:.6f}")
        print(f"Valid loss: {val_loss:.6f}")
        
        scheduler.step(val_loss)
        current_lr = optimizer.param_groups[0]['lr']
        print(f"Learning rate: {current_lr:.6f}")
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            save_path = 'models/v2_weighted_gru.pt'
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': train_loss,
                'val_loss': val_loss,
            }, save_path)
            print(f"✓ Saved best model (val_loss: {val_loss:.6f})")
    
    print("\n" + "="*60)
    print("Training complete!")
    print(f"Best validation loss: {best_val_loss:.6f}")
    print(f"Model saved to: models/v2_weighted_gru.pt")
    print("="*60)


if __name__ == '__main__':
    main()

