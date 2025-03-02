import torch
import torch.nn as nn
import torch.nn.functional as F


class FeedForward(nn.Module):
    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1, activation: nn.Module = nn.GELU):
        
        """
        Args:
            `d_model`: the number of expected features in the input (required).
            `d_ff`: the number of expected features in the output (required).
            `dropout`: the dropout value (default=0.1).
            `activation`: the activation function of the feedforward layer (default=nn.ReLU).
        """
        super(FeedForward, self).__init__()
        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = activation()

    def forward(self, x: torch.FloatTensor) -> torch.FloatTensor:
        """
        Args:
            `x`: shape (batch_size, max_len, d_model)

        Returns:
            same shape as input x
        """
        return self.w_2(self.dropout(self.activation(self.w_1(x))))
