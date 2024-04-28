# -*- coding: utf-8 -*-
""" 
@Time   : 2023/11/06
 
@Author : Shen Fang
"""


from .stfgnn import STFGNN, construct_adj_fusion
import torch


class STFGNNSeq2Seq(STFGNN):
    def __init__(self, config, data_feature, use_long_history=False):
        super().__init__(config, data_feature)
        self.use_long_history = use_long_history
    
    def forward(self, history_data: torch.Tensor, long_history_data: torch.Tensor, future_data: torch.Tensor, epoch: int, batch_seen: int, train: bool, **kwargs) -> torch.Tensor:
        if self.use_long_history:
            input_data = long_history_data
        else:
            input_data = history_data
        
        x = torch.relu(self.First_FC(input_data))  # B, Tin, N, Cin

        for model in self.STSGCLS:
            x = model(x, self.mask)
        
        need_concat = []
        for i in range(self.horizon):
            out_step = self.predictLayer[i](x)  # (B, 1, N)
            need_concat.append(out_step)

        out = torch.cat(need_concat, dim=1)  # B, Tout, N, Cout

        del need_concat

        return out

        