"""
v2 Model: Baseline architecture + weighted loss

Key changes from v1:
- Go BACK to baseline's architecture (64 hidden, 1 layer)
- Add weighted loss (4x weight on t0 vs t1, matching WPCC formula)
- Keep it simple - v1 taught us bigger isn't better
"""

import torch
import torch.nn as nn

class WeightedGRU(nn.Module):
    """Same architecture as baseline, but we'll train with weighted loss"""
    
    def __init__(self):
        super().__init__()
        
        # Baseline architecture: simple and effective
        self.gru = nn.GRU(
            input_size=32,
            hidden_size=64,
            num_layers=1,
            batch_first=True
        )
        
        self.fc1 = nn.Linear(64, 32)
        self.dropout = nn.Dropout(0.1)  # Light dropout
        self.fc2 = nn.Linear(32, 2)
        
    def forward(self, x):
        # x shape: (batch, 100, 32)
        out, _ = self.gru(x)
        
        # Take last timestep
        last = out[:, -1, :]  # (batch, 64)
        
        # MLP head
        x = torch.relu(self.fc1(last))
        x = self.dropout(x)
        x = self.fc2(x)  # (batch, 2)
        
        return x


