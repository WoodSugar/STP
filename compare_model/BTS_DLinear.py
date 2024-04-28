# -*- coding: utf-8 -*-
""" 
@Time   : 2023/11/13
 
@Author : Shen Fang
"""


from basic_ts.archs.arch_zoo.linear_arch.dlinear import DLinear
import torch
import torch.nn as nn
from model_utils import MLP


class DLinearSeq2Seq(DLinear):
    def __init__(self, **model_args):
        model_args["seq_len"] = model_args["history_seq_len"]
        model_args["pred_len"] = model_args["prediction_seq_len"]
        model_args["enc_in"] = model_args["num_nodes"]
        model_args["individual"] = False

        super().__init__(**model_args)
        self.in_module = MLP((model_args["history_seq_len"] * model_args["input_dim"], model_args["history_seq_len"]), act_type=None)
        self.use_long_history = model_args["use_long_history"]
        self.out_module = MLP((model_args["prediction_seq_len"],  model_args["output_dim"] * model_args["prediction_seq_len"]), act_type=None)
        self.trg_len = model_args["prediction_seq_len"]

    def forward(self, history_data: torch.Tensor, long_history_data: torch.Tensor, future_data: torch.Tensor, epoch: int, batch_seen: int, train: bool, **kwargs) -> torch.Tensor:
        if self.use_long_history:
            input_data = long_history_data.transpose(1, 2)
        else:
            input_data = history_data.transpose(1, 2)
            
        B, N = input_data.size(0), input_data.size(1)
        input_data = input_data.reshape(B, N, -1)
        input_data = self.in_module(input_data)
        
        input_data = input_data.transpose(1,2).unsqueeze(-1)
            
        prediction = super().forward(input_data, future_data, epoch, batch_seen, train, **kwargs)  # [B, L, N, 1]

        prediction = prediction.transpose(1, 2)  # []
        B, N = prediction.size(0), prediction.size(1)
        prediction = prediction.view(B, N, -1)  
        prediction = self.out_module(prediction).view(B, N, self.trg_len, -1)
        prediction = prediction.transpose(1, 2)
        
        return prediction


class DLinear_LN(DLinearSeq2Seq):
    def __init__(self, **model_args):
        super().__init__(**model_args)
        self.norm = nn.LayerNorm([model_args["num_nodes"], model_args["prediction_seq_len"] * model_args["output_dim"]])

    def forward(self, history_data: torch.Tensor, long_history_data: torch.Tensor, future_data: torch.Tensor, epoch: int, batch_seen: int, train: bool, **kwargs) -> torch.Tensor:
        prediction = super().forward(history_data, long_history_data, future_data, epoch, batch_seen, train, **kwargs)  #[B, T, N, C]
        b, t, n, c = prediction.size()

        prediction = prediction.transpose(1, 2).view(b, n, t * c)

        prediction = self.norm(prediction)
        return prediction.view(b, n, t, c).transpose(1, 2)
    

class DLinear_SLN(DLinearSeq2Seq):
    def __init__(self, **model_args):
        super().__init__(**model_args)
        self.norm = nn.LayerNorm(model_args["num_nodes"])

    def forward(self, history_data: torch.Tensor, long_history_data: torch.Tensor, future_data: torch.Tensor, epoch: int, batch_seen: int, train: bool, **kwargs) -> torch.Tensor:
        prediction = super().forward(history_data, long_history_data, future_data, epoch, batch_seen, train, **kwargs)  #[B, T, N, C]
        b, t, n, c = prediction.size()

        prediction = prediction.transpose(1, 2).view(b, n, t * c)

        prediction = self.norm(prediction.transpose(1, 2))  # [b, t * c, n]
        return prediction.view(b, n, t, c).transpose(1, 2)


class DLinear_TLN(DLinearSeq2Seq):
    def __init__(self, **model_args):
        super().__init__(**model_args)
        self.norm = nn.LayerNorm(model_args["prediction_seq_len"] * model_args["output_dim"])

    def forward(self, history_data: torch.Tensor, long_history_data: torch.Tensor, future_data: torch.Tensor, epoch: int, batch_seen: int, train: bool, **kwargs) -> torch.Tensor:
        prediction = super().forward(history_data, long_history_data, future_data, epoch, batch_seen, train, **kwargs)  #[B, T, N, C]
        b, t, n, c = prediction.size()

        prediction = prediction.transpose(1, 2).view(b, n, t * c)

        prediction = self.norm(prediction)
        return prediction.view(b, n, t, c).transpose(1, 2)