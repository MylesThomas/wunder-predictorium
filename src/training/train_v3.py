"""
Train v3: Baseline Replication

Goal: Match baseline's 0.2595 validation score to verify our training works.

Approach:
- Try different simple GRU architectures
- Use standard MSE loss (like baseline)
- Keep everything else consistent

Once we match baseline, we know our pipeline is correct!
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
import argparse

from src.models.baseline_replica import BaselineReplicaGRU

class LOBDataset(Dataset):
    def __init__(self, parquet_path, window_size=100):
        print(f"Loading data from {parquet_path}...")
        self.df = pd.read_parquet(parquet_path)
        self.window_size = window_size
        
        # All 32 features in correct order
        feature_cols = (
            [f'p{i}' for i in range(12)] +  # 12 price features
            [f'v{i}' for i in range(12)] +  # 12 volume features
            [f'dp{i}' for i in range(4)] +  # 4 trade price features
            [f'dv{i}' for i in range(4)]    # 4 trade volume features
        )
        
        # Group by sequence
        self.sequences = []
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


def train_epoch(model, loader, optimizer, device, criterion):
    model.train()
    total_loss = 0
    
    for x, y in tqdm(loader, desc="Training"):
        x, y = x.to(device), y.to(device)
        
        optimizer.zero_grad()
        pred = model(x)
        loss = criterion(pred, y)
        loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        optimizer.step()
        total_loss += loss.item()
    
    return total_loss / len(loader)


def validate(model, loader, device, criterion):
    model.eval()
    total_loss = 0
    
    with torch.no_grad():
        for x, y in tqdm(loader, desc="Validating"):
            x, y = x.to(device), y.to(device)
            pred = model(x)
            loss = criterion(pred, y)
            total_loss += loss.item()
    
    return total_loss / len(loader)


def main():
    # Parse arguments for easy experimentation
    parser = argparse.ArgumentParser()
    parser.add_argument('--hidden', type=int, default=32, help='Hidden size (32, 64, 128)')
    parser.add_argument('--layers', type=int, default=1, help='Number of GRU layers (1 or 2)')
    parser.add_argument('--dropout', type=float, default=0.0, help='Dropout rate')
    parser.add_argument('--epochs', type=int, default=10, help='Number of epochs')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--batch_size', type=int, default=256, help='Batch size')
    args = parser.parse_args()
    
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    print("="*60)
    print("Training v3: Baseline Replication")
    print("="*60)
    print(f"Device: {DEVICE}")
    print(f"Architecture: GRU(hidden={args.hidden}, layers={args.layers}, dropout={args.dropout})")
    print(f"Training: epochs={args.epochs}, lr={args.lr}, batch_size={args.batch_size}")
    print(f"Loss: Standard MSE")
    print()
    
    # Load data
    train_dataset = LOBDataset('wnn_predictorium_starterpack/datasets/train.parquet')
    valid_dataset = LOBDataset('wnn_predictorium_starterpack/datasets/valid.parquet')
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=0)
    valid_loader = DataLoader(valid_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)
    
    # Model
    model = BaselineReplicaGRU(
        hidden_size=args.hidden,
        num_layers=args.layers,
        dropout=args.dropout
    ).to(DEVICE)
    
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {total_params:,}")
    print()
    
    # Standard MSE loss (like baseline)
    criterion = nn.MSELoss()
    
    # Optimizer & scheduler
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=2
    )
    
    # Training loop
    best_val_loss = float('inf')
    
    for epoch in range(args.epochs):
        print(f"\nEpoch {epoch+1}/{args.epochs}")
        print("-" * 60)
        
        train_loss = train_epoch(model, train_loader, optimizer, DEVICE, criterion)
        val_loss = validate(model, valid_loader, DEVICE, criterion)
        
        print(f"Train loss: {train_loss:.6f}")
        print(f"Valid loss: {val_loss:.6f}")
        
        scheduler.step(val_loss)
        current_lr = optimizer.param_groups[0]['lr']
        print(f"Learning rate: {current_lr:.6f}")
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            save_path = f'models/v3_h{args.hidden}_l{args.layers}.pt'
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': train_loss,
                'val_loss': val_loss,
                'config': {
                    'hidden_size': args.hidden,
                    'num_layers': args.layers,
                    'dropout': args.dropout
                }
            }, save_path)
            print(f"✓ Saved best model (val_loss: {val_loss:.6f})")
    
    print("\n" + "="*60)
    print("Training complete!")
    print(f"Best validation loss: {best_val_loss:.6f}")
    print(f"Model saved to: models/v3_h{args.hidden}_l{args.layers}.pt")
    print("="*60)
    print()
    print("Next: Run validation to get WPCC score:")
    print(f"  python src/evaluation/validate_v3.py --hidden {args.hidden} --layers {args.layers}")


if __name__ == '__main__':
    main()

