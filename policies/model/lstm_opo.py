import torch
import torch.nn as nn


class LSTMLoadPredictor(nn.Module):
    """Predicts next edge-server load from a window of past load observations.

    Input:  (batch, seq_len, n_nodes)  — idle CPU fraction per node
    Output: (batch, n_nodes)           — predicted idle CPU next step
    """

    def __init__(self, n_nodes: int, hidden_dim: int = 64, n_layers: int = 1):
        super().__init__()
        self.n_nodes = n_nodes
        self.lstm = nn.LSTM(
            input_size=n_nodes,
            hidden_size=hidden_dim,
            num_layers=n_layers,
            batch_first=True,
        )
        self.head = nn.Linear(hidden_dim, n_nodes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, seq_len, n_nodes)
        Returns:
            (batch, n_nodes)
        """
        out, _ = self.lstm(x)
        return self.head(out[:, -1, :])


class LSTMTaskPredictor(nn.Module):
    """Predicts next task feature vector from a window of past task features.

    Input:  (batch, seq_len, 4)  — [task_size, cycles_per_bit, trans_bit_rate, ddl]
    Output: (batch, 4)           — predicted next task features
    """

    TASK_DIM = 4

    def __init__(self, hidden_dim: int = 64, n_layers: int = 1):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=self.TASK_DIM,
            hidden_size=hidden_dim,
            num_layers=n_layers,
            batch_first=True,
        )
        self.head = nn.Linear(hidden_dim, self.TASK_DIM)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, seq_len, 4)
        Returns:
            (batch, 4)
        """
        out, _ = self.lstm(x)
        return self.head(out[:, -1, :])
