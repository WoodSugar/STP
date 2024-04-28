# -*- coding: utf-8 -*-
"""
@Time   : 2023/9/13

@Author : Shen Fang
"""
import torch
import torch.nn as nn

from model_utils import MLP
from main.main_arch.hypergraph import HyperNet


class MultiLayerPerceptron(nn.Module):
    """Two fully connected layer."""

    def __init__(self, history_seq_len: int, prediction_seq_len: int, hidden_dim: int, input_dim: int, output_dim: int, n_layers: int,
                 use_long_history: bool=True):
        super().__init__()
        assert n_layers >= 2, "number of layers should be greater than 2"
        dim_list = [history_seq_len * input_dim, prediction_seq_len * output_dim]

        for i in range(n_layers - 2):
            dim_list.insert(i + 1, hidden_dim)

        self.nn = MLP(dim_list, act_type="tanh")

        # self.fc1 = MLP((history_seq_len* input_dim, hidden_dim), act_type="linear")
        # self.fc2 = MLP((hidden_dim, prediction_seq_len*output_dim), act_type="linear")
        # self.act = nn.ReLU()
        self.prediction_seq_len = prediction_seq_len
        self.use_long_history = use_long_history

    def forward(self, history_data: torch.Tensor, long_history_data: torch.Tensor, future_data: torch.Tensor, epoch: int, batch_seen: int, train: bool, **kwargs) -> torch.Tensor:
        """Feedforward function of MLP.

        Args:
            history_data (torch.Tensor): inputs with shape [B, L, N, C].
            long_history_data (torch.Tensor): inputs with shape [B, L * P, N, C].

        Returns:
            torch.Tensor: outputs with shape [B, L, N, C]
        """
        if self.use_long_history:
            input_data = long_history_data.transpose(1, 2)
        else:
            input_data = history_data.transpose(1, 2)

        b, n = input_data.size(0), input_data.size(1)
        input_data = input_data.reshape(b, n, -1)     # B, N, L*P*C

        prediction = self.nn(input_data)
        # prediction = self.fc2(self.act(self.fc1(input_data)))     # B, N, L*C
        return prediction.view(b, n, self.prediction_seq_len, -1).transpose(1, 2)     # B, L, N, C
    

class MLP_LN(nn.Module):
    """Two fully connected layer."""

    def __init__(self, history_seq_len: int, prediction_seq_len: int, hidden_dim: int, input_dim: int, output_dim: int, n_layers: int, num_nodes: int,
                 use_long_history: bool=True):
        super().__init__()
        assert n_layers >= 2, "number of layers should be greater than 2"
        dim_list = [history_seq_len * input_dim, prediction_seq_len * output_dim]

        for i in range(n_layers - 2):
            dim_list.insert(i + 1, hidden_dim)

        self.nn = MLP(dim_list, act_type="tanh")

        # self.fc1 = MLP((history_seq_len* input_dim, hidden_dim), act_type="linear")
        # self.fc2 = MLP((hidden_dim, prediction_seq_len*output_dim), act_type="linear")
        # self.act = nn.ReLU()
        self.prediction_seq_len = prediction_seq_len
        self.use_long_history = use_long_history
        self.norm = nn.LayerNorm([num_nodes, output_dim * prediction_seq_len])

    def forward(self, history_data: torch.Tensor, long_history_data: torch.Tensor, future_data: torch.Tensor, epoch: int, batch_seen: int, train: bool, **kwargs) -> torch.Tensor:
        """Feedforward function of MLP.

        Args:
            history_data (torch.Tensor): inputs with shape [B, L, N, C].
            long_history_data (torch.Tensor): inputs with shape [B, L * P, N, C].

        Returns:
            torch.Tensor: outputs with shape [B, L, N, C]
        """
        if self.use_long_history:
            input_data = long_history_data.transpose(1, 2)
        else:
            input_data = history_data.transpose(1, 2)

        b, n = input_data.size(0), input_data.size(1)
        input_data = input_data.reshape(b, n, -1)     # B, N, L*P*C

        prediction = self.nn(input_data)
        prediction = self.norm(prediction)
        # prediction = self.fc2(self.act(self.fc1(input_data)))     # B, N, L*C
        return prediction.view(b, n, self.prediction_seq_len, -1).transpose(1, 2)     # B, L, N, C
    

class MLP_HyperNet(nn.Module):
    """Two fully connected layer."""

    def __init__(self, history_seq_len: int, prediction_seq_len: int, hidden_dim: int, input_dim: int, output_dim: int, n_layers: int, num_nodes: int,
                 s_cluster: int, t_cluster:int, affine: bool=True,
                 use_long_history: bool=True):
        super().__init__()
        assert n_layers >= 2, "number of layers should be greater than 2"
        dim_list = [history_seq_len * input_dim, prediction_seq_len * output_dim]

        for i in range(n_layers - 2):
            dim_list.insert(i + 1, hidden_dim)

        self.nn = MLP(dim_list, act_type="tanh")

        # self.fc1 = MLP((history_seq_len* input_dim, hidden_dim), act_type="linear")
        # self.fc2 = MLP((hidden_dim, prediction_seq_len*output_dim), act_type="linear")
        # self.act = nn.ReLU()
        self.prediction_seq_len = prediction_seq_len
        self.use_long_history = use_long_history
        self.hyper = HyperNet(num_nodes, prediction_seq_len, output_dim, s_cluster, t_cluster,affine )

    def forward(self, history_data: torch.Tensor, long_history_data: torch.Tensor, future_data: torch.Tensor, epoch: int, batch_seen: int, train: bool, **kwargs) -> torch.Tensor:
        """Feedforward function of MLP.

        Args:
            history_data (torch.Tensor): inputs with shape [B, L, N, C].
            long_history_data (torch.Tensor): inputs with shape [B, L * P, N, C].

        Returns:
            torch.Tensor: outputs with shape [B, L, N, C]
        """
        if self.use_long_history:
            input_data = long_history_data.transpose(1, 2)
        else:
            input_data = history_data.transpose(1, 2)

        b, n = input_data.size(0), input_data.size(1)
        input_data = input_data.reshape(b, n, -1)     # B, N, L*P*C

        prediction = self.nn(input_data)
        # prediction = self.fc2(self.act(self.fc1(input_data)))     # B, N, L*C
        prediction = prediction.view(b, n, self.prediction_seq_len, -1).transpose(1, 2)     # B, L, N, C
        prediction = self.hyper(prediction)

        return prediction


class MLP_SLN(nn.Module):
    """Two fully connected layer."""

    def __init__(self, history_seq_len: int, prediction_seq_len: int, hidden_dim: int, input_dim: int, output_dim: int, n_layers: int, num_nodes: int,
                 use_long_history: bool=True):
        super().__init__()
        assert n_layers >= 2, "number of layers should be greater than 2"
        dim_list = [history_seq_len * input_dim, prediction_seq_len * output_dim]
        for i in range(n_layers - 2):
            dim_list.insert(i + 1, hidden_dim)
        self.nn = MLP(dim_list, act_type="tanh")
        self.prediction_seq_len = prediction_seq_len
        self.use_long_history = use_long_history
        self.norm = nn.LayerNorm(num_nodes)
    def forward(self, history_data: torch.Tensor, long_history_data: torch.Tensor, future_data: torch.Tensor, epoch: int, batch_seen: int, train: bool, **kwargs) -> torch.Tensor:
        """Feedforward function of MLP.

        Args:
            history_data (torch.Tensor): inputs with shape [B, L, N, C].
            long_history_data (torch.Tensor): inputs with shape [B, L * P, N, C].

        Returns:
            torch.Tensor: outputs with shape [B, L, N, C]
        """
        if self.use_long_history:
            input_data = long_history_data.transpose(1, 2)
        else:
            input_data = history_data.transpose(1, 2)

        b, n = input_data.size(0), input_data.size(1)
        input_data = input_data.reshape(b, n, -1)     # B, N, L*P*C

        prediction = self.nn(input_data)
        prediction = self.norm(prediction.transpose(1, 2))  # B, L*P*C, N

        prediction = prediction.transpose(1, 2)  # B, N, L*P*C
    
        return prediction.view(b, n, self.prediction_seq_len, -1).transpose(1, 2)     # B, L, N, C


class MLP_TLN(nn.Module):
    def __init__(self, history_seq_len: int, prediction_seq_len: int, hidden_dim: int, input_dim: int, output_dim: int, n_layers: int, num_nodes: int,
                 use_long_history: bool=True):
        super().__init__()
        assert n_layers >= 2, "number of layers should be greater than 2"
        dim_list = [history_seq_len * input_dim, prediction_seq_len * output_dim]
        
        for i in range(n_layers - 2):
            dim_list.insert(i + 1, hidden_dim)
        
        self.nn = MLP(dim_list, act_type="tanh")

        self.prediction_seq_len = prediction_seq_len
        self.use_long_history = use_long_history
        
        self.norm = nn.LayerNorm(prediction_seq_len * output_dim)
    
    def forward(self, history_data: torch.Tensor, long_history_data: torch.Tensor, future_data: torch.Tensor, epoch: int, batch_seen: int, train: bool, **kwargs) -> torch.Tensor:
        """Feedforward function of MLP.

        Args:
            history_data (torch.Tensor): inputs with shape [B, L, N, C].
            long_history_data (torch.Tensor): inputs with shape [B, L * P, N, C].

        Returns:
            torch.Tensor: outputs with shape [B, L, N, C]
        """
        if self.use_long_history:
            input_data = long_history_data.transpose(1, 2)
        else:
            input_data = history_data.transpose(1, 2)

        b, n = input_data.size(0), input_data.size(1)
        input_data = input_data.reshape(b, n, -1)     # B, N, L*P*C
        
        prediction = self.nn(input_data)
        prediction = self.norm(prediction)

        prediction = prediction.view(b, n, self.prediction_seq_len, -1).transpose(1, 2)
        return prediction