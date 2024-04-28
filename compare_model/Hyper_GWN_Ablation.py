# -*- coding: utf-8 -*-
""" 
@Time   : 2024/04/10
 
@Author : Shen Fang
"""

import math
import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
from main.main_arch.hypergraph import HyperNet, HyperNetnFC, HyperNetnS, HyperNetnT


class HyperGCN(HyperNet):
    def __init__(self, num_nodes: int, T: int, c_in: int, c_out: int, Sk: int, Tk: int, affine: bool = True):
        super().__init__(num_nodes, T, c_in, Sk, Tk, affine)

        if c_in != c_out:
            self.c_transform = nn.Linear(c_in, c_out)
        else:
            self.c_transform = nn.Identity()

        self.reset_linear_parameters()

    def reset_linear_parameters(self):
        
        if hasattr(self.c_transform, "weight"):
            init.kaiming_uniform_(self.c_transform.weight, a=math.sqrt(5))
            if self.c_transform.bias is not None:
                fan_in, _ = init._calculate_fan_in_and_fan_out(self.c_transform.weight)
                bound = 1 / math.sqrt(fan_in)
                init.uniform_(self.c_transform.bias, -bound, bound)

    def forward(self, input_features: torch.Tensor) -> torch.Tensor:
        input_features = input_features.transpose(1, 3)
        
        output =  super().forward(input_features)
        output = self.c_transform(output)
        
        output = output.transpose(1, 3)
        return output


class HyperGCN_nFC(HyperNetnFC):
    def __init__(self, num_nodes: int, T: int, c_in: int, c_out: int, Sk: int, Tk: int, affine: bool = True):
        super().__init__(num_nodes, T, c_in, Sk, Tk, affine)

        if c_in != c_out:
            self.c_transform = nn.Linear(c_in, c_out)
        else:
            self.c_transform = nn.Identity()

        self.reset_linear_parameters()

    def reset_linear_parameters(self):
        
        if hasattr(self.c_transform, "weight"):
            init.kaiming_uniform_(self.c_transform.weight, a=math.sqrt(5))
            if self.c_transform.bias is not None:
                fan_in, _ = init._calculate_fan_in_and_fan_out(self.c_transform.weight)
                bound = 1 / math.sqrt(fan_in)
                init.uniform_(self.c_transform.bias, -bound, bound)

    def forward(self, input_features: torch.Tensor) -> torch.Tensor:
        input_features = input_features.transpose(1, 3)
        
        output =  super().forward(input_features)
        output = self.c_transform(output)
        
        output = output.transpose(1, 3)
        return output


class HyperGCN_nS(HyperNetnS):
    def __init__(self, num_nodes: int, T: int, c_in: int, c_out: int, Sk: int, Tk: int, affine: bool = True):
        super().__init__(num_nodes, T, c_in, Sk, Tk, affine)

        if c_in != c_out:
            self.c_transform = nn.Linear(c_in, c_out)
        else:
            self.c_transform = nn.Identity()

        self.reset_linear_parameters()

    def reset_linear_parameters(self):
        
        if hasattr(self.c_transform, "weight"):
            init.kaiming_uniform_(self.c_transform.weight, a=math.sqrt(5))
            if self.c_transform.bias is not None:
                fan_in, _ = init._calculate_fan_in_and_fan_out(self.c_transform.weight)
                bound = 1 / math.sqrt(fan_in)
                init.uniform_(self.c_transform.bias, -bound, bound)

    def forward(self, input_features: torch.Tensor) -> torch.Tensor:
        input_features = input_features.transpose(1, 3)
        
        output =  super().forward(input_features)
        output = self.c_transform(output)
        
        output = output.transpose(1, 3)
        return output


class HyperGCN_nT(HyperNetnT):
    def __init__(self, num_nodes: int, T: int, c_in: int, c_out: int, Sk: int, Tk: int, affine: bool = True):
        super().__init__(num_nodes, T, c_in, Sk, Tk, affine)

        if c_in != c_out:
            self.c_transform = nn.Linear(c_in, c_out)
        else:
            self.c_transform = nn.Identity()

        self.reset_linear_parameters()

    def reset_linear_parameters(self):
        
        if hasattr(self.c_transform, "weight"):
            init.kaiming_uniform_(self.c_transform.weight, a=math.sqrt(5))
            if self.c_transform.bias is not None:
                fan_in, _ = init._calculate_fan_in_and_fan_out(self.c_transform.weight)
                bound = 1 / math.sqrt(fan_in)
                init.uniform_(self.c_transform.bias, -bound, bound)

    def forward(self, input_features: torch.Tensor) -> torch.Tensor:
        input_features = input_features.transpose(1, 3)
        
        output =  super().forward(input_features)
        output = self.c_transform(output)
        
        output = output.transpose(1, 3)
        return output


class GWNHyperGConv(nn.Module):
    """
    Paper: Graph WaveNet for Deep Spatial-Temporal Graph Modeling
    Link: https://arxiv.org/abs/1906.00121
    Ref Official Code: https://github.com/nnzhan/Graph-WaveNet/blob/master/model.py
    """

    def __init__(self, num_nodes, dropout=0.3, supports=None,
                    gcn_bool=True, addaptadj=True, aptinit=None,
                    in_dim=2, out_dim=12, residual_channels=32,
                    dilation_channels=32, skip_channels=256, end_channels=512,
                    kernel_size=2, blocks=4, layers=2, Sk=20, Tk=1, affine=True, method="default"):
        super(GWNHyperGConv, self).__init__()
        self.dropout = dropout
        self.blocks = blocks
        self.layers = layers
        self.gcn_bool = gcn_bool
        self.addaptadj = addaptadj

        self.filter_convs = nn.ModuleList()
        self.gate_convs = nn.ModuleList()
        self.residual_convs = nn.ModuleList()
        self.skip_convs = nn.ModuleList()
        self.bn = nn.ModuleList()
        self.gconv = nn.ModuleList()

        self.start_conv = nn.Conv2d(in_channels=in_dim,
                                    out_channels=residual_channels,
                                    kernel_size=(1, 1))
        self.supports = supports

        receptive_field = 1

        self.supports_len = 0
        if supports is not None:
            self.supports_len += len(supports)

        if gcn_bool and addaptadj:
            if aptinit is None:
                if supports is None:
                    self.supports = []
                self.nodevec1 = nn.Parameter(
                    torch.randn(num_nodes, 10), requires_grad=True)
                self.nodevec2 = nn.Parameter(
                    torch.randn(10, num_nodes), requires_grad=True)
                self.supports_len += 1
            else:
                if supports is None:
                    self.supports = []
                m, p, n = torch.svd(aptinit)
                initemb1 = torch.mm(m[:, :10], torch.diag(p[:10] ** 0.5))
                initemb2 = torch.mm(torch.diag(p[:10] ** 0.5), n[:, :10].t())
                self.nodevec1 = nn.Parameter(initemb1, requires_grad=True)
                self.nodevec2 = nn.Parameter(initemb2, requires_grad=True)
                self.supports_len += 1
        
        for b in range(blocks):
            additional_scope = kernel_size - 1
            new_dilation = 1
            for i in range(layers):
                new_dilation *= 2
                receptive_field += additional_scope
                additional_scope *= 2
        self.receptive_field = receptive_field
        
        receptive_field = 1

        for b in range(blocks):  # blocks = 4
            additional_scope = kernel_size - 1
            new_dilation = 1
            for i in range(layers):  # layers = 2
                # dilated convolutions
                self.filter_convs.append(nn.Conv2d(in_channels=residual_channels,
                                                   out_channels=dilation_channels,
                                                   kernel_size=(1, kernel_size), dilation=new_dilation))

                self.gate_convs.append(nn.Conv2d(in_channels=residual_channels,
                                                 out_channels=dilation_channels,
                                                 kernel_size=(1, kernel_size), dilation=new_dilation))

                # 1x1 convolution for residual connection
                self.residual_convs.append(nn.Conv2d(in_channels=dilation_channels,
                                                     out_channels=residual_channels,
                                                     kernel_size=(1, 1)))

                # 1x1 convolution for skip connection
                self.skip_convs.append(nn.Conv2d(in_channels=dilation_channels,
                                                 out_channels=skip_channels,
                                                 kernel_size=(1, 1)))
                self.bn.append(nn.BatchNorm2d(residual_channels))
                new_dilation *= 2
                receptive_field += additional_scope
                additional_scope *= 2
                if self.gcn_bool:
                    # self.gconv.append(nn.Identity())
                    if method == "default":
                        self.gconv.append(
                        HyperGCN(num_nodes, self.receptive_field - 1 - i * kernel_size - b * (kernel_size + 1), 
                                 dilation_channels, residual_channels, Sk, Tk, affine))
                    elif method == "nFC":
                        self.gconv.append(
                            HyperGCN_nFC(num_nodes, self.receptive_field - 1 - i * kernel_size - b * (kernel_size + 1), 
                                 dilation_channels, residual_channels, Sk, Tk, affine))
                    elif method == "nS":
                        self.gconv.append(
                            HyperGCN_nS(num_nodes, self.receptive_field - 1 - i * kernel_size - b * (kernel_size + 1), 
                                 dilation_channels, residual_channels, Sk, Tk, affine))
                    elif method == "nT":
                        self.gconv.append(
                            HyperGCN_nT(num_nodes, self.receptive_field - 1 - i * kernel_size - b * (kernel_size + 1), 
                                     dilation_channels, residual_channels, Sk, Tk, affine))
                    elif method == "nWb":
                        self.gconv.append(
                            HyperGCN(num_nodes, self.receptive_field - 1 - i * kernel_size - b * (kernel_size + 1),  
                                     dilation_channels, residual_channels, Sk, Tk, False))
                    else:
                        raise ValueError("Invalid method: {}".format(method))
                    

        self.end_conv_1 = nn.Conv2d(in_channels=skip_channels,
                                    out_channels=end_channels,
                                    kernel_size=(1, 1),
                                    bias=True)

        self.end_conv_2 = nn.Conv2d(in_channels=end_channels,
                                    out_channels=out_dim,
                                    kernel_size=(1, 1),
                                    bias=True)

    def forward(self, history_data: torch.Tensor, future_data: torch.Tensor, batch_seen: int, epoch: int, train: bool, **kwargs) -> torch.Tensor:
        """Feedforward function of Graph WaveNet.

        Args:
            history_data (torch.Tensor): shape [B, L, N, C]

        Returns:
            torch.Tensor: [B, L, N, 1]
        """

        input = history_data.transpose(1, 3).contiguous()
        in_len = input.size(3)
        if in_len < self.receptive_field:
            x = nn.functional.pad(
                input, (self.receptive_field-in_len, 0, 0, 0))
        else:
            x = input
        x = self.start_conv(x)
        skip = 0

        # calculate the current adaptive adj matrix once per iteration
        new_supports = None
        if self.gcn_bool and self.addaptadj and self.supports is not None:
            adp = F.softmax(
                F.relu(torch.mm(self.nodevec1, self.nodevec2)), dim=1)
            new_supports = self.supports + [adp]

        # WaveNet layers
        for i in range(self.blocks * self.layers):

            #            |----------------------------------------|     *residual*
            #            |                                        |
            #            |    |-- conv -- tanh --|                |
            # -> dilate -|----|                  * ----|-- 1x1 -- + -->	*input*
            #                 |-- conv -- sigm --|     |
            #                                         1x1
            #                                          |
            # ---------------------------------------> + ------------->	*skip*

            #(dilation, init_dilation) = self.dilations[i]

            #residual = dilation_func(x, dilation, init_dilation, i)
            residual = x
            # dilated convolution
            filter = self.filter_convs[i](residual)
            filter = torch.tanh(filter)
            gate = self.gate_convs[i](residual)
            gate = torch.sigmoid(gate)
            x = filter * gate

            # parametrized skip connection

            s = x
            s = self.skip_convs[i](s)
            try:
                skip = skip[:, :, :,  -s.size(3):]
            except:
                skip = 0
            skip = s + skip

            if self.gcn_bool and self.supports is not None:
                # [B, C, N, T]
                x = self.gconv[i](x)
            else:
                x = self.residual_convs[i](x)

            x = x + residual[:, :, :, -x.size(3):]

            x = self.bn[i](x)

        x = F.relu(skip)
        x = F.relu(self.end_conv_1(x))
        x = self.end_conv_2(x)
        return x
    

class HyperGWN(GWNHyperGConv):
    def __init__(self, num_nodes, dropout=0.3, supports=None, gcn_bool=True, addaptadj=True, aptinit=None, in_dim=2, out_dim=12, residual_channels=32, dilation_channels=32, skip_channels=256, end_channels=512, kernel_size=2, blocks=4, layers=2, Sk=20, Tk=1, affine=True, **model_kwargs):
        method = model_kwargs.get("method", "default")
        super().__init__(num_nodes, dropout, supports, gcn_bool, addaptadj, aptinit, in_dim, out_dim, residual_channels, dilation_channels, skip_channels, end_channels, kernel_size, blocks, layers, Sk, Tk, affine, method)
        self.use_long_history = model_kwargs.get("use_long_history", False)
    def forward(self, history_data: torch.Tensor, long_history_data: torch.Tensor, future_data: torch.Tensor, epoch: int, batch_seen: int, train: bool, **kwargs) -> torch.Tensor:
        if self.use_long_history:
            return super().forward(long_history_data, future_data, batch_seen=batch_seen, epoch=epoch, train=train, **kwargs)
        else:
            return super().forward(history_data, future_data, batch_seen=batch_seen, epoch=epoch, train=train, **kwargs)