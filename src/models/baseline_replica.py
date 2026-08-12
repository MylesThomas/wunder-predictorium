"""
v3 Model: Baseline Replica

Goal: Replicate baseline's performance (0.2595 validation) to verify our training works.

Strategy: Try different simple GRU architectures with standard MSE loss.
Once we match baseline, we know our pipeline is correct.
"""

import torch
import torch.nn as nn

class BaselineReplicaGRU(nn.Module):
    """
    Simple GRU model - trying to match the baseline architecture.
    
    We'll try different hidden sizes to find what works:
    - 32, 64, 128 hidden units
    - 1 or 2 layers
    - Standard MSE loss
    """
    
    def __init__(self, hidden_size=32, num_layers=1, dropout=0.0):
        super().__init__()
        
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        self.gru = nn.GRU(
            input_size=32,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0.0,
            batch_first=True
        )
        
        # Simple MLP head
        self.fc1 = nn.Linear(hidden_size, hidden_size // 2)
        self.fc2 = nn.Linear(hidden_size // 2, 2)
        
    def forward(self, x):
        # x shape: (batch, 100, 32)
        out, _ = self.gru(x)
        
        # Take last timestep
        last = out[:, -1, :]  # (batch, hidden_size)
        
        # MLP head
        x = torch.relu(self.fc1(last))
        x = self.fc2(x)  # (batch, 2)
        
        return x

