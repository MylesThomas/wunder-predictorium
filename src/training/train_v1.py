"""
Training script for v1 model (Improved GRU)

Usage:
    cd /Users/thomasmyles/dev/wunder-predictorium
    source venv/bin/activate
    python src/training/train_v1.py
"""

import os
import sys
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))
from src.models.improved_gru import ImprovedGRU


class LOBDataset(Dataset):
    """Dataset for LOB sequences."""
    
    def __init__(self, parquet_path, window_size=100):
        print(f"Loading data from {parquet_path}...")
        self.df = pd.read_parquet(parquet_path)
        self.window_size = window_size
        
        # Get feature and target columns
        self.feature_cols = [col for col in self.df.columns 
                            if col not in ['seq_ix', 'step_in_seq', 'need_prediction', 't0', 't1']]
        self.target_cols = ['t0', 't1']
        
        # Group by sequence
        self.sequences = []
        for seq_ix, group in self.df.groupby('seq_ix'):
            features = group[self.feature_cols].values
            targets = group[self.target_cols].values
            need_pred = group['need_prediction'].values
            
            self.sequences.append({
                'features': features,
                'targets': targets,
                'need_prediction': need_pred
            })
        
        print(f"Loaded {len(self.sequences)} sequences")
        print(f"Features: {len(self.feature_cols)}")
        print(f"Targets: {len(self.target_cols)}")
        
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        seq = self.sequences[idx]
        features = seq['features']  # (1000, 32)
        targets = seq['targets']    # (1000, 2)
        need_pred = seq['need_prediction']  # (1000,)
        
        # Get all prediction points (step 99-999)
        pred_indices = np.where(need_pred)[0]
        
        # Randomly sample one prediction point from this sequence
        pred_idx = np.random.choice(pred_indices)
        
        # Get window of 100 steps ending at pred_idx
        start_idx = max(0, pred_idx - self.window_size + 1)
        end_idx = pred_idx + 1
        
        window = features[start_idx:end_idx]
        
        # Pad if needed (shouldn't happen since pred starts at 99)
        if len(window) < self.window_size:
            padding = np.zeros((self.window_size - len(window), features.shape[1]))
            window = np.vstack([padding, window])
        
        target = targets[pred_idx]
        
        return (
            torch.FloatTensor(window),    # (100, 32)
            torch.FloatTensor(target)     # (2,)
        )


def train_epoch(model, dataloader, optimizer, criterion, device):
    """Train for one epoch."""
    model.train()
    total_loss = 0
    
    for features, targets in tqdm(dataloader, desc="Training"):
        features = features.to(device)
        targets = targets.to(device)
        
        # Forward pass
        optimizer.zero_grad()
        predictions = model(features)
        loss = criterion(predictions, targets)
        
        # Backward pass
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        
        total_loss += loss.item()
    
    return total_loss / len(dataloader)


def validate(model, dataloader, criterion, device):
    """Validate the model."""
    model.eval()
    total_loss = 0
    
    with torch.no_grad():
        for features, targets in tqdm(dataloader, desc="Validating"):
            features = features.to(device)
            targets = targets.to(device)
            
            predictions = model(features)
            loss = criterion(predictions, targets)
            
            total_loss += loss.item()
    
    return total_loss / len(dataloader)


def main():
    # Config
    TRAIN_PATH = "wnn_predictorium_starterpack/datasets/train.parquet"
    VALID_PATH = "wnn_predictorium_starterpack/datasets/valid.parquet"
    MODEL_SAVE_PATH = "models/v1_improved_gru.pt"
    
    BATCH_SIZE = 64
    EPOCHS = 10
    LEARNING_RATE = 1e-3
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    print(f"Device: {DEVICE}")
    print(f"Batch size: {BATCH_SIZE}")
    print(f"Epochs: {EPOCHS}")
    print(f"Learning rate: {LEARNING_RATE}")
    print()
    
    # Create datasets
    train_dataset = LOBDataset(TRAIN_PATH)
    valid_dataset = LOBDataset(VALID_PATH)
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=4,
        pin_memory=True if DEVICE.type == 'cuda' else False
    )
    
    valid_loader = DataLoader(
        valid_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=4,
        pin_memory=True if DEVICE.type == 'cuda' else False
    )
    
    # Create model
    model = ImprovedGRU().to(DEVICE)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    print(f"\nModel parameters: {total_params:,}")
    print()
    
    # Loss and optimizer
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=2
    )
    
    # Training loop
    best_val_loss = float('inf')
    
    for epoch in range(EPOCHS):
        print(f"\nEpoch {epoch+1}/{EPOCHS}")
        print("-" * 50)
        
        train_loss = train_epoch(model, train_loader, optimizer, criterion, DEVICE)
        val_loss = validate(model, valid_loader, criterion, DEVICE)
        
        print(f"Train Loss: {train_loss:.6f}")
        print(f"Valid Loss: {val_loss:.6f}")
        
        # Learning rate scheduling
        scheduler.step(val_loss)
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            os.makedirs(os.path.dirname(MODEL_SAVE_PATH), exist_ok=True)
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
                'train_loss': train_loss,
            }, MODEL_SAVE_PATH)
            print(f"✓ Saved best model (val_loss: {val_loss:.6f})")
    
    print(f"\nTraining complete!")
    print(f"Best validation loss: {best_val_loss:.6f}")
    print(f"Model saved to: {MODEL_SAVE_PATH}")


if __name__ == "__main__":
    main()

