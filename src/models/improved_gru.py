"""
v1 Model: Improved GRU Architecture

Changes from baseline:
- Hidden size: 128 → 256
- Layers: 1 → 2
- Added dropout: 0.3 (GRU) + 0.2 (Dense)
- Added Dense layer before output
- Loss: MSE (same as baseline)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ImprovedGRU(nn.Module):
    """
    Improved GRU model for LOB prediction.
    
    Architecture:
        Input (batch, 100, 32)
          ↓
        GRU (256 hidden, 2 layers, dropout=0.3)
          ↓
        Dense(256 → 128) + ReLU + Dropout(0.2)
          ↓
        Dense(128 → 2)
          ↓
        Output (batch, 2) [t0, t1]
    """
    
    def __init__(self, 
                 input_size=32,
                 hidden_size=256,
                 num_layers=2,
                 dropout=0.3,
                 fc_dropout=0.2):
        super(ImprovedGRU, self).__init__()
        
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # GRU layers
        self.gru = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            batch_first=True
        )
        
        # Dense layers
        self.fc1 = nn.Linear(hidden_size, 128)
        self.dropout = nn.Dropout(fc_dropout)
        self.fc2 = nn.Linear(128, 2)
        
    def forward(self, x):
        """
        Forward pass.
        
        Args:
            x: Input tensor of shape (batch, seq_len, features)
            
        Returns:
            Output tensor of shape (batch, 2) with predictions for [t0, t1]
        """
        # GRU forward pass
        # out: (batch, seq_len, hidden_size)
        # hidden: (num_layers, batch, hidden_size)
        out, hidden = self.gru(x)
        
        # Take last timestep output
        last = out[:, -1, :]  # (batch, hidden_size)
        
        # Dense layers
        x = F.relu(self.fc1(last))
        x = self.dropout(x)
        x = self.fc2(x)  # (batch, 2)
        
        return x


if __name__ == "__main__":
    # Test the model
    model = ImprovedGRU()
    
    # Create dummy input: batch=4, seq_len=100, features=32
    dummy_input = torch.randn(4, 100, 32)
    
    # Forward pass
    output = model(dummy_input)
    
    print("Model architecture test:")
    print(f"Input shape: {dummy_input.shape}")
    print(f"Output shape: {output.shape}")
    print(f"\nExpected: (4, 2)")
    print(f"Got: {output.shape}")
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"\nTotal parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    
    print("\n✓ Model test passed!")


