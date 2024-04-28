# -*- coding: utf-8 -*-
""" 
@Time   : 2023/11/05
 
@Author : Shen Fang
"""

from basic_ts.archs.arch_zoo.gwnet_arch import GraphWaveNet
import torch


class GWNet(GraphWaveNet):
    def __init__(self, num_nodes, dropout=0.3, supports=None, gcn_bool=True, addaptadj=True, aptinit=None, in_dim=2, out_dim=2*6, residual_channels=32, dilation_channels=32, skip_channels=128, end_channels=256, kernel_size=2, blocks=4, layers=2, use_long_history=False):
        super().__init__(num_nodes, dropout, supports, gcn_bool, addaptadj, aptinit, in_dim, out_dim, residual_channels, dilation_channels, skip_channels, end_channels, kernel_size, blocks, layers)
        self.use_long_history = use_long_history
        self.in_dim = in_dim
        self.trg_len = out_dim // in_dim
        self.num_nodes = num_nodes

    def forward(self, history_data: torch.Tensor, long_history_data: torch.Tensor, future_data: torch.Tensor, batch_seen: int, epoch: int, train: bool, **kwargs):
        if self.use_long_history:
            prediction = super().forward(long_history_data, future_data, batch_seen, epoch, train, **kwargs)
        else:
            prediction = super().forward(history_data, future_data, batch_seen, epoch, train, **kwargs) # [B, L2*inC, N, 1]
        B = prediction.size(0)
        prediction = prediction.squeeze().transpose(1, 2).view(B, self.num_nodes, self.trg_len, -1).transpose(1, 2)  # [B, L2, N, inC]
        return prediction