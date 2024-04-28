# -*- coding: utf-8 -*-
"""
@Time   : 2023/9/13

@Author : Shen Fang
"""
import torch
import torch.nn as nn

from model_utils import MLP
from main.main_arch.hypergraph import HyperMean, HyperMax, HyperMin, HyperLearn, HyperLearnFC


class LR(nn.Module):
    """Two fully connected layer."""

    def __init__(self, history_seq_len: int, prediction_seq_len: int, hidden_dim: int, input_dim: int, output_dim: int, norm_type: str=None, act_type: str="leaky_relu"):
        super().__init__()
        self.fc1 = MLP((history_seq_len * input_dim, prediction_seq_len * output_dim), act_type=act_type, norm_type=norm_type)
        self.prediction_seq_len = prediction_seq_len

    def forward(self, history_data: torch.Tensor, long_history_data: torch.Tensor, future_data: torch.Tensor, epoch: int, batch_seen: int, train: bool, **kwargs) -> torch.Tensor:
        """Feedforward function of MLP.

        Args:
            history_data (torch.Tensor): inputs with shape [B, L, N, C].
            long_history_data (torch.Tensor): inputs with shape [B, L * P, N, C].

        Returns:
            torch.Tensor: outputs with shape [B, L, N, C]
        """
        long_history_data = long_history_data.transpose(1, 2)
        b, n = long_history_data.size(0), long_history_data.size(1)
        long_history_data = long_history_data.reshape(b, n, -1)     # B, N, L*P*C
        prediction = self.fc1(long_history_data)     # B, N, L*C
        return prediction.view(b, n, self.prediction_seq_len, -1).transpose(1, 2)     # B, L, N, C
    

class LR_LN(nn.Module):
    def __init__(self, history_seq_len: int, prediction_seq_len: int, num_nodes: int, input_dim: int, output_dim: int, act_type: str=None, affine=True):
        super().__init__()
        self.fc1 = MLP((history_seq_len* input_dim, prediction_seq_len*output_dim), act_type=act_type)
        self.prediction_seq_len = prediction_seq_len
        self.norm = nn.LayerNorm([num_nodes, output_dim * prediction_seq_len], elementwise_affine=affine)

    def forward(self, history_data: torch.Tensor, long_history_data: torch.Tensor, future_data: torch.Tensor, epoch: int, batch_seen: int, train: bool, **kwargs) -> torch.Tensor:
        """
        LR + LayerNorm (across nodes and time steps)
        """
        long_history_data = long_history_data.transpose(1, 2)
        b, n = long_history_data.size(0), long_history_data.size(1)
        long_history_data = long_history_data.reshape(b, n, -1)   # B, N, L*P*C

        prediction = self.fc1(long_history_data)  # [B, N, L*C]
        prediction = self.norm(prediction)

        return prediction.view(b, n, self.prediction_seq_len, -1).transpose(1, 2)  # [B, L, N, C]


class LR_HyperMean(nn.Module):
    def __init__(self, history_seq_len: int, prediction_seq_len: int, num_nodes: int, input_dim: int, output_dim: int, act_type: str=None, affine=True):
        super().__init__()
        self.fc1 = MLP((history_seq_len * input_dim, prediction_seq_len * output_dim), act_type=act_type)
        self.prediction_seq_len = prediction_seq_len
        self.hyper = HyperMean(num_nodes, prediction_seq_len, output_dim, affine)

    def forward(self, history_data: torch.Tensor, long_history_data: torch.Tensor, future_data: torch.Tensor, epoch: int, batch_seen: int, train: bool, **kwargs) -> torch.Tensor:
        long_history_data = long_history_data.transpose(1, 2)
        b, n = long_history_data.size(0), long_history_data.size(1)
        long_history_data = long_history_data.reshape(b, n, -1)   # B, N, L*P*C

        prediction = self.fc1(long_history_data)  # [B, N, L*C]
        prediction = prediction.view(b, n, self.prediction_seq_len, -1).transpose(1, 2) # [B, L, N, C]

        prediction = self.hyper(prediction)
        return prediction


class LR_HyperMax(nn.Module):
    def __init__(self, history_seq_len: int, prediction_seq_len: int, num_nodes: int, input_dim: int, output_dim: int, act_type: str=None, affine=True):
        super().__init__()
        self.fc1 = MLP((history_seq_len * input_dim, prediction_seq_len * output_dim), act_type=act_type)
        self.prediction_seq_len = prediction_seq_len
        self.hyper = HyperMax(num_nodes, prediction_seq_len, output_dim, affine)

    def forward(self, history_data: torch.Tensor, long_history_data: torch.Tensor, future_data: torch.Tensor, epoch: int, batch_seen: int, train: bool, **kwargs) -> torch.Tensor:
        long_history_data = long_history_data.transpose(1, 2)
        b, n = long_history_data.size(0), long_history_data.size(1)
        long_history_data = long_history_data.reshape(b, n, -1)   # B, N, L*P*C

        prediction = self.fc1(long_history_data)  # [B, N, L*C]
        prediction = prediction.view(b, n, self.prediction_seq_len, -1).transpose(1, 2) # [B, L, N, C]

        prediction = self.hyper(prediction)
        return prediction


class LR_HyperMin(nn.Module):
    def __init__(self, history_seq_len: int, prediction_seq_len: int, num_nodes: int, input_dim: int, output_dim: int, act_type: str=None, affine=True):
        super().__init__()
        self.fc1 = MLP((history_seq_len * input_dim, prediction_seq_len * output_dim), act_type=act_type)
        self.prediction_seq_len = prediction_seq_len
        self.hyper = HyperMin(num_nodes, prediction_seq_len, output_dim, affine)

    def forward(self, history_data: torch.Tensor, long_history_data: torch.Tensor, future_data: torch.Tensor, epoch: int, batch_seen: int, train: bool, **kwargs) -> torch.Tensor:
        long_history_data = long_history_data.transpose(1, 2)
        b, n = long_history_data.size(0), long_history_data.size(1)
        long_history_data = long_history_data.reshape(b, n, -1)   # B, N, L*P*C

        prediction = self.fc1(long_history_data)  # [B, N, L*C]
        prediction = prediction.view(b, n, self.prediction_seq_len, -1).transpose(1, 2) # [B, L, N, C]

        prediction = self.hyper(prediction)
        return prediction


class LR_HyperLearn(nn.Module):
    def __init__(self, history_seq_len: int, prediction_seq_len: int, num_nodes: int, input_dim: int, output_dim: int, act_type: str=None, affine=True):
        super().__init__()
        self.fc1 = MLP((history_seq_len * input_dim, prediction_seq_len * output_dim), act_type=act_type)
        self.prediction_seq_len = prediction_seq_len
        self.hyper = HyperLearn(num_nodes, prediction_seq_len, output_dim, affine)

    def forward(self, history_data: torch.Tensor, long_history_data: torch.Tensor, future_data: torch.Tensor, epoch: int, batch_seen: int, train: bool, **kwargs) -> torch.Tensor:
        long_history_data = long_history_data.transpose(1, 2)
        b, n = long_history_data.size(0), long_history_data.size(1)
        long_history_data = long_history_data.reshape(b, n, -1)   # B, N, L*P*C

        prediction = self.fc1(long_history_data)  # [B, N, L*C]
        prediction = prediction.view(b, n, self.prediction_seq_len, -1).transpose(1, 2) # [B, L, N, C]

        prediction = self.hyper(prediction)
        return prediction


class LR_HyperLearnFC(nn.Module):
    def __init__(self, history_seq_len: int, prediction_seq_len: int, num_nodes: int, input_dim: int, output_dim: int, act_type: str=None, affine=True):
        super().__init__()
        self.fc1 = MLP((history_seq_len * input_dim, prediction_seq_len * output_dim), act_type=act_type)
        self.prediction_seq_len = prediction_seq_len
        self.hyper = HyperLearnFC(num_nodes, prediction_seq_len, output_dim, affine)

    def forward(self, history_data: torch.Tensor, long_history_data: torch.Tensor, future_data: torch.Tensor, epoch: int, batch_seen: int, train: bool, **kwargs) -> torch.Tensor:
        long_history_data = long_history_data.transpose(1, 2)
        b, n = long_history_data.size(0), long_history_data.size(1)
        long_history_data = long_history_data.reshape(b, n, -1)   # B, N, L*P*C

        prediction = self.fc1(long_history_data)  # [B, N, L*C]
        prediction = prediction.view(b, n, self.prediction_seq_len, -1).transpose(1, 2) # [B, L, N, C]

        prediction = self.hyper(prediction)
        return prediction


class LR_SLN(nn.Module):
    def __init__(self, history_seq_len: int, prediction_seq_len: int, num_nodes: int, input_dim: int, output_dim: int, act_type: str=None):
        super().__init__()
        self.fc1 = MLP((history_seq_len* input_dim, prediction_seq_len*output_dim), act_type=act_type)
        self.prediction_seq_len = prediction_seq_len
        self.norm = nn.LayerNorm(num_nodes)
    
    def forward(self, history_data: torch.Tensor, long_history_data: torch.Tensor, future_data: torch.Tensor, epoch: int, batch_seen: int, train: bool, **kwargs) -> torch.Tensor:
        """
        LR + Space + LayerNorm (across node steps)
        """
        long_history_data = long_history_data.transpose(1, 2)
        b, n = long_history_data.size(0), long_history_data.size(1)
        long_history_data = long_history_data.reshape(b, n, -1)   # B, N, L*P*C

        prediction = self.fc1(long_history_data)  # [B, N, L*C]
        prediction = prediction.transpose(1, 2)  # [B, L*C, N]
        prediction = self.norm(prediction)
        prediction = prediction.transpose(1, 2)  # [B, N, L*C]
        return prediction.view(b, n, self.prediction_seq_len, -1).transpose(1, 2)  # [B, L, N, C]


class LR_TLN(nn.Module):
    def __init__(self, history_seq_len: int, prediction_seq_len: int, num_nodes: int, input_dim: int, output_dim: int, act_type: str=None):
        super().__init__()
        self.fc1 = MLP((history_seq_len* input_dim, prediction_seq_len*output_dim), act_type=act_type)
        self.prediction_seq_len = prediction_seq_len
        self.norm = nn.LayerNorm(prediction_seq_len * output_dim)

    def forward(self, history_data: torch.Tensor, long_history_data: torch.Tensor, future_data: torch.Tensor, epoch: int, batch_seen: int, train: bool, **kwargs) -> torch.Tensor:
        """
        LR + Time + LayerNorm (across time steps)
        """
        long_history_data = long_history_data.transpose(1, 2)
        b, n = long_history_data.size(0), long_history_data.size(1)

        long_history_data = long_history_data.reshape(b, n, -1)  # [B, N, L*P*C]

        prediction = self.fc1(long_history_data)  # [B, N, L*C]
        prediction = self.norm(prediction)
        return prediction.view(b, n, self.prediction_seq_len, -1).transpose(1, 2)


class SLN_LR(nn.Module):
    def __init__(self, history_seq_len: int, prediction_seq_len: int, num_nodes: int, input_dim: int, output_dim: int, act_type: str=None):
        super().__init__()
        self.fc1 = MLP((history_seq_len* input_dim, prediction_seq_len*output_dim), act_type=act_type)
        self.prediction_seq_len = prediction_seq_len
        self.norm = nn.LayerNorm(num_nodes)
        self.bn = nn.BatchNorm1d(num_nodes)
    
    def forward(self, history_data: torch.Tensor, long_history_data: torch.Tensor, future_data: torch.Tensor, epoch: int, batch_seen: int, train: bool, **kwargs) -> torch.Tensor:
        """
        Space_LayerNorm + LR (across node steps)
        """
        # [B, T, N, C]

        long_history_data = long_history_data.transpose(1, 2)
        b, n = long_history_data.size(0), long_history_data.size(1)
        long_history_data = long_history_data.reshape(b, n, -1)   # B, N, T*C

        prediction = long_history_data.transpose(1, 2)  # [B, N, T*C] -> [B, T*C, N]
        prediction = self.norm(prediction)
        prediction = prediction.transpose(1, 2)  # [B, T*C, N] -> [B, N, T*C]

        prediction = self.fc1(prediction)  # [B, N, T*C]  -> [B, N, trg_len*C]
        prediction = self.bn(prediction)
        prediction = prediction.view(b, n, self.prediction_seq_len, -1).transpose(1, 2)  # [B, L, N, C]

        return prediction



