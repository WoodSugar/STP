from basic_ts.archs.arch_zoo.mtgnn_arch import MTGNN
import torch 
import torch.nn as nn


class MTGNNSeq2Seq(MTGNN):
    def __init__(self, gcn_true, buildA_true, gcn_depth, num_nodes, predefined_A=None, static_feat=None, dropout=0.3, subgraph_size=20, node_dim=40, dilation_exponential=1, conv_channels=32, residual_channels=32, skip_channels=64, end_channels=128, seq_length=12, in_dim=2, out_dim=12, layers=3, propalpha=0.05, tanhalpha=3, layer_norm_affline=True, use_long_history=False):
        

        super().__init__(gcn_true, buildA_true, gcn_depth, num_nodes, predefined_A, static_feat, dropout, subgraph_size, node_dim, dilation_exponential, conv_channels, residual_channels, skip_channels, end_channels, seq_length, in_dim, out_dim, layers, propalpha, tanhalpha, layer_norm_affline)
        
        self.use_long_history = use_long_history
        self.trg_len = out_dim // in_dim
        
        if not buildA_true:
            self.predefined_A = nn.Parameter(self.predefined_A, requires_grad=False) 

    def forward(self, history_data: torch.Tensor, long_history_data: torch.Tensor, future_data: torch.Tensor, epoch: int, batch_seen: int, train: bool, **kwargs):
        if "idx" in kwargs:
            node_idx = kwargs["idx"]
        else:
            raise ValueError("where is my idx data? ")
        
        if self.use_long_history:
            prediction = super().forward(history_data=long_history_data, idx=node_idx)
        else:
            prediction = super().forward(history_data=history_data, idx=node_idx)  # [B, L*C, N, 1]
        
        B = prediction.size(0)
        prediction = prediction.squeeze().transpose(1, 2).view(B, self.num_nodes, self.trg_len, -1).transpose(1, 2)  # [B, L2, N, inC]

        return prediction
        