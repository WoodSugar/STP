# -*- coding: utf-8 -*-
""" 
@Time   : 2023/11/05
 
@Author : Shen Fang
"""


from basic_ts.archs.arch_zoo.dcrnn_arch import DCRNN
import torch
import torch.nn as nn


class Seq2SeqAttrs:
    def __init__(self, adj_mx, **model_kwargs):
        self.adj_mx = adj_mx
        self.max_diffusion_step = int(
            model_kwargs.get("max_diffusion_step", 2))
        self.cl_decay_steps = int(model_kwargs.get("cl_decay_steps", 1000))
        self.filter_type = model_kwargs.get("filter_type", "laplacian")
        self.num_nodes = int(model_kwargs.get("num_nodes", 1))
        self.num_rnn_layers = int(model_kwargs.get("num_rnn_layers", 1))
        self.rnn_units = int(model_kwargs.get("rnn_units"))
        self.hidden_state_size = self.num_nodes * self.rnn_units
        self.use_gc_for_ru = model_kwargs.get("use_gc_for_ru")


class DCRNN_Seq2Seq(DCRNN, Seq2SeqAttrs):
    def __init__(self, adj_mx, **model_kwargs):
        super().__init__(adj_mx, **model_kwargs)
        self.use_long_history = model_kwargs.get("use_long_history", False)

    def forward(self, history_data: torch.Tensor, long_history_data: torch.Tensor, future_data: torch.Tensor, epoch: int, batch_seen: int, train: bool, **kwargs):
        """ Feedforward function for DCRNN.
            history_data (torch.Tensor): inputs with shape [B, L1, N, C].
            long_history_data (torch.Tensor): inputs with shape [B, L1 * P, N, C].
            future_data (torch.Tensor) : inputs with shape [B, L2, N, C].
        """
        # B, src_len, N, C = history_data.size()
        # history_data = history_data.transpose(1, 2)  # [L, B, N, C]
        if self.use_long_history:
            return super().forward(long_history_data, future_data, batch_seen=batch_seen, train=train, **kwargs)
        else:
            return super().forward(history_data, future_data, batch_seen=batch_seen, train=train, **kwargs)


class DCRNN_LN(DCRNN):
    def __init__(self, adj_mx, **model_kwargs):
        super().__init__(adj_mx, **model_kwargs)
        self.use_long_history = model_kwargs.get("use_long_history", False)
        self.norm = nn.LayerNorm([model_kwargs["num_nodes"]])
        
    def forward(self, history_data: torch.Tensor, long_history_data: torch.Tensor, future_data: torch.Tensor, epoch: int, batch_seen: int, train: bool, **kwargs):
        if self.use_long_history:
            prediction = super().forward(long_history_data, future_data, batch_seen=batch_seen, train=train, **kwargs)
        else:
            prediction = super().forward(history_data, future_data, batch_seen=batch_seen, train=train, **kwargs)

        prediction = prediction.transpose(-1, -2)
        prediction = self.norm(prediction)
        prediction = prediction.transpose(-1, -2)  # [b, t, n, c]

        return prediction