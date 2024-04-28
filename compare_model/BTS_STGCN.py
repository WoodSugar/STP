# -*- coding: utf-8 -*-
""" 
@Time   : 2024/03/15
 
@Author : Shen Fang
"""
import torch
from basic_ts.archs.arch_zoo.stgcn_arch import STGCN


class STGCN_Seq2Seq(STGCN):
    def __init__(self, Kt, Ks, blocks, T, n_vertex, act_func, graph_conv_type, gso, bias, droprate, **model_kwargs):
        super().__init__(Kt, Ks, blocks, T, n_vertex, act_func, graph_conv_type, gso, bias, droprate)
        self.use_long_history = model_kwargs.get("use_long_history", False)

    def forward(self, history_data: torch.Tensor, long_history_data: torch.Tensor, future_data: torch.Tensor, epoch: int, batch_seen: int, train: bool, **kwargs) -> torch.Tensor:
        if self.use_long_history:
            return super().forward(long_history_data, future_data, batch_seen=batch_seen, epoch=epoch, train=train, **kwargs)
        else:
            return super().forward(history_data, future_data, batch_seen=batch_seen, epoch=epoch, train=train, **kwargs)


class STGCN_LN(STGCN):
    def __init__(self, Kt, Ks, blocks, T, n_vertex, act_func, graph_conv_type, gso, bias, droprate, **model_kwargs):
        super().__init__(Kt, Ks, blocks, T, n_vertex, act_func, graph_conv_type, gso, bias, droprate)
        self.use_long_history = model_kwargs.get("use_long_history", False)

    def forward(self, history_data: torch.Tensor, long_history_data: torch.Tensor, future_data: torch.Tensor, epoch: int, batch_seen: int, train: bool, **kwargs) -> torch.Tensor:
        if self.use_long_history:
            return super().forward(long_history_data, future_data, batch_seen=batch_seen, epoch=epoch, train=train, **kwargs)
        else:
            return super().forward(history_data, future_data, batch_seen=batch_seen, epoch=epoch, train=train, **kwargs)