import torch
from torch import nn

class MultiLayerPerceptron(nn.Module):
    """Two fully connected layer."""

    def __init__(self, history_seq_len: int, prediction_seq_len: int, hidden_dim: int, input_dim: int, output_dim: int):
        super().__init__()
        self.fc1 = nn.Linear(history_seq_len* input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, prediction_seq_len*output_dim)
        self.act = nn.Tanh()
        self.prediction_seq_len = prediction_seq_len

    def forward(self, history_data: torch.Tensor, future_data: torch.Tensor, batch_seen: int, epoch: int, train: bool, **kwargs) -> torch.Tensor:
        """Feedforward function of AGCRN.

        Args:
            history_data (torch.Tensor): inputs with shape [B, L, N, C].

        Returns:
            torch.Tensor: outputs with shape [B, L, N, C]
        """
        history_data = history_data.transpose(1, 2)
        b, n = history_data.size(0), history_data.size(1)
        history_data = history_data.reshape(b, n, -1)     # B, N, L*C
        prediction = self.fc2(self.act(self.fc1(history_data)))     # B, N, L*C
        return prediction.view(b, n, self.prediction_seq_len, -1).transpose(1, 2)     # B, L, N, C