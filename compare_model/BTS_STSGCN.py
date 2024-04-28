# -*- coding: utf-8 -*-
""" 
@Time   : 2023/11/06
 
@Author : Shen Fang
"""


from .stsgcn import STSGCN, construct_adj
import torch



class STSGCNSeq2Seq(STSGCN):
    def __init__(self, adj, history, num_of_vertices, in_dim, out_dim, hidden_dims, first_layer_embedding_size, out_layer_dim, activation='GLU', use_mask=True, temporal_emb=True, spatial_emb=True, horizon=12, strides=3, use_long_history=False):
        super().__init__(adj, history, num_of_vertices, in_dim, out_dim, hidden_dims, first_layer_embedding_size, out_layer_dim, activation, use_mask, temporal_emb, spatial_emb, horizon, strides)
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
