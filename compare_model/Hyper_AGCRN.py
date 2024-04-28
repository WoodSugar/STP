# -*- coding: utf-8 -*-
""" 
@Time   : 2024/04/12
 
@Author : Shen Fang
"""

import math
import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
from main.main_arch.hypergraph import HyperNet


class AVWHyperGCN(nn.Module):
    def __init__(self, dim_in, dim_out, cheb_k, embed_dim, num_nodes, temporal_length, Sk, Tk, affine):
        super().__init__()
        # self.cheb_k = cheb_k

        self.weights_pool = nn.Parameter(
            torch.FloatTensor(embed_dim, dim_in, dim_out))
        self.bias_pool = nn.Parameter(torch.FloatTensor(embed_dim, dim_out))

        self.c_transform = nn.Linear(dim_out + embed_dim, dim_out)
        self.gcn = HyperNet(num_nodes, temporal_length, dim_out, Sk, Tk, affine)

        # if dim_in+embed_dim != dim_out:
        #     self.c_transform = nn.Linear(dim_in + embed_dim, dim_out)
        # else:
        #     self.c_transform = nn.Identity()
        
        self.reset_linear_paratemers()

    def reset_linear_paratemers(self):
        if hasattr(self.c_transform, "weight"):
            init.kaiming_uniform_(self.c_transform.weight, a=math.sqrt(5))
            if self.c_transform.bias is not None:
                fan_in, _ = init._calculate_fan_in_and_fan_out(self.c_transform.weight)
                bound = 1 / math.sqrt(fan_in)
                init.uniform_(self.c_transform.bias, -bound, bound)
    
    def forward(self, x, node_embeddings, default_graph):
        """
            x shaped[B, N, C], node_embeddings shaped [N, D] -> supports shaped [N, N].
            node_embeddings: [N, D].

        """
        # node_num = node_embeddings.shape[0]
        # supports = F.softmax(F.relu(torch.mm(node_embeddings, node_embeddings.transpose(0, 1))), dim=1)
        # support_set = [torch.eye(node_num).to(supports.device), supports]
        
        # for k in range(2, self.cheb_k):
        #     support_set.append(torch.matmul(
        #         2 * supports, support_set[-1]) - support_set[-2])
        # supports = torch.stack(support_set, dim=0)  # cheb_k, N, N

        # weights = torch.einsum(
        #     'nd,dio->nio', node_embeddings, self.weights_pool)
        # bias = torch.matmul(node_embeddings, self.bias_pool)  # N, dim_out

        # x = x.unsqueeze(1)  # [B, T=1, N, C]
        # x = self.gcn(x)  # [B, 1, N, C]
        # x_g = x.squeeze(1)  # [B, N, C]

        # x_gconv = torch.einsum('bni,nio->bno', x_g,
        #                        weights) + bias  # b, N, dim_out
        # return x_gconv

        # ================================================================================================
        # node_num = node_embeddings.shape[0]
        # supports = F.softmax(F.relu(torch.mm(node_embeddings, node_embeddings.transpose(0, 1))), dim=1)
        # support_set = [torch.eye(node_num).to(supports.device), supports]
        # for k in range(2, self.cheb_k):
        #     support_set.append(torch.matmul(
        #         2 * supports, support_set[-1]) - support_set[-2])
        # supports = torch.stack(support_set, dim=0)

        weights = torch.einsum(
            'nd,dio->nio', node_embeddings, self.weights_pool)
        bias = torch.matmul(node_embeddings, self.bias_pool)  # N, dim_out

        x_emb = torch.einsum('bni,nio->bno', x, weights) + bias

        node_embeddings = node_embeddings.unsqueeze(0) # [1, N, D]
        node_embeddings = node_embeddings.repeat(x.shape[0], 1, 1)  # [B, N, D]
        x = torch.cat([x_emb, node_embeddings], dim=-1)  # [B, N, D + C]
        x = self.c_transform(x)  
        x = x.unsqueeze(1)  # [B, T=1, N, C]
        output = self.gcn(x)  # [B, T=1, N, C]
        # output = self.c_transform(x.squeeze(1))  # [B, N, C]
        return output.squeeze(1)


class AVWGCN(nn.Module):
    def __init__(self, dim_in, dim_out, cheb_k, embed_dim):
        super(AVWGCN, self).__init__()
        self.cheb_k = cheb_k
        self.weights_pool = nn.Parameter(
            torch.FloatTensor(embed_dim, cheb_k, dim_in, dim_out))
        self.bias_pool = nn.Parameter(torch.FloatTensor(embed_dim, dim_out))

    def forward(self, x, node_embeddings, default_graph):
        # x shaped[B, N, C], node_embeddings shaped [N, D] -> supports shaped [N, N]
        # output shape [B, N, C]
        node_num = node_embeddings.shape[0]
        if default_graph == "ori":
            supports = F.softmax(F.relu(torch.mm(node_embeddings, node_embeddings.transpose(0, 1))), dim=1)
        elif default_graph == "idn":
            supports = torch.eye(node_num).to(node_embeddings.device)
        elif default_graph == "uni":
            supports = torch.ones(node_num, node_num).to(node_embeddings.device) / node_num
        elif default_graph == "ran":
            supports = F.softmax(torch.rand(node_num, node_num).to(node_embeddings.device), dim=1)

        support_set = [torch.eye(node_num).to(supports.device), supports]
        # default cheb_k = 3
        for k in range(2, self.cheb_k):
            support_set.append(torch.matmul(
                2 * supports, support_set[-1]) - support_set[-2])
        supports = torch.stack(support_set, dim=0)
        # N, cheb_k, dim_in, dim_out
        weights = torch.einsum(
            'nd,dkio->nkio', node_embeddings, self.weights_pool)
        bias = torch.matmul(node_embeddings, self.bias_pool)  # N, dim_out
        x_g = torch.einsum("knm,bmc->bknc", supports,
                           x)  # B, cheb_k, N, dim_in
        x_g = x_g.permute(0, 2, 1, 3)  # B, N, cheb_k, dim_in
        x_gconv = torch.einsum('bnki,nkio->bno', x_g,
                               weights) + bias  # b, N, dim_out
        return x_gconv


class AGCRNCell(nn.Module):
    def __init__(self, node_num, dim_in, dim_out, cheb_k, embed_dim, temporal_length, Sk, Tk, affine):
        super(AGCRNCell, self).__init__()
        self.node_num = node_num
        self.hidden_dim = dim_out
        self.gate = AVWHyperGCN(dim_in+self.hidden_dim, 2 * dim_out, cheb_k, embed_dim, 
                                node_num, temporal_length, Sk, Tk, affine)
        
        self.update = AVWHyperGCN(dim_in+self.hidden_dim, dim_out, cheb_k, embed_dim,
                                  node_num, temporal_length, Sk, Tk, affine)

    def forward(self, x, state, node_embeddings, default_graph):
        # x: B, num_nodes, input_dim
        # state: B, num_nodes, hidden_dim
        state = state.to(x.device)
        input_and_state = torch.cat((x, state), dim=-1)
        z_r = torch.sigmoid(self.gate(input_and_state, node_embeddings, default_graph))
        z, r = torch.split(z_r, self.hidden_dim, dim=-1)
        candidate = torch.cat((x, z*state), dim=-1)
        hc = torch.tanh(self.update(candidate, node_embeddings, default_graph))
        h = r*state + (1-r)*hc
        return h

    def init_hidden_state(self, batch_size):
        return torch.zeros(batch_size, self.node_num, self.hidden_dim)


class AVWDCRNN(nn.Module):
    def __init__(self, node_num, dim_in, dim_out, cheb_k, embed_dim, temporal_length, Sk, Tk, affine, num_layers=1):
        super(AVWDCRNN, self).__init__()
        assert num_layers >= 1, 'At least one DCRNN layer in the Encoder.'
        self.node_num = node_num
        self.input_dim = dim_in
        self.num_layers = num_layers
        
        self.dcrnn_cells = nn.ModuleList()
        self.dcrnn_cells.append(
            AGCRNCell(node_num, dim_in, dim_out, cheb_k, embed_dim, 1, Sk, Tk, affine))
        # node_num, dim_in, dim_out, cheb_k, embed_dim, temporal_length, Sk, Tk, affine
        
        self.gcn_cells = nn.ModuleList()
        self.gcn_cells.append(
            HyperNet(node_num, temporal_length, dim_out, Sk, Tk, affine))
            # AVWHyperGCN (dim_in, dim_out, cheb_k, embed_dim, num_nodes, temporal_length, Sk, Tk, affine)

        for _ in range(1, num_layers):
            self.dcrnn_cells.append(
                AGCRNCell(node_num, dim_out, dim_out, cheb_k, embed_dim, 1, Sk, Tk, affine))
            self.gcn_cells.append(
                HyperNet(node_num, temporal_length, dim_out, Sk, Tk, affine))


    def forward(self, x, init_state, node_embeddings, default_graph):
        # shape of x: (B, T, N, D)
        # shape of init_state: (num_layers, B, N, hidden_dim)

        # x = self.gcn(x)
        assert x.shape[2] == self.node_num and x.shape[3] == self.input_dim
        
        seq_length = x.shape[1]
        current_inputs = x
        output_hidden = []
        for i in range(self.num_layers):
            state = init_state[i]
            inner_states = []

            for t in range(seq_length):
                state = self.dcrnn_cells[i](
                    current_inputs[:, t, :, :], state, node_embeddings, default_graph)
                inner_states.append(state)
            
            output_hidden.append(state)
            current_inputs = torch.stack(inner_states, dim=1)
            current_inputs = self.gcn_cells[i](current_inputs)
        # current_inputs: the outputs of last layer: (B, T, N, hidden_dim)
        # output_hidden: the last state for each layer: (num_layers, B, N, hidden_dim)
        #last_state: (B, N, hidden_dim)
        return current_inputs, output_hidden

    def init_hidden(self, batch_size):
        init_states = []
        for i in range(self.num_layers):
            init_states.append(
                self.dcrnn_cells[i].init_hidden_state(batch_size))
        # (num_layers, B, N, hidden_dim)
        return torch.stack(init_states, dim=0)


class AGCRN(nn.Module):
    """
    Paper: Adaptive Graph Convolutional Recurrent Network for Trafﬁc Forecasting
    Official Code: https://github.com/LeiBAI/AGCRN
    Link: https://arxiv.org/abs/2007.02842
    """

    def __init__(self, num_nodes, input_dim, rnn_units, output_dim, horizon, num_layers, default_graph, embed_dim, cheb_k, temporal_length, Sk, Tk, affine):
        super(AGCRN, self).__init__()
        self.num_node = num_nodes
        self.input_dim = input_dim
        self.hidden_dim = rnn_units
        self.output_dim = output_dim
        self.horizon = horizon
        self.num_layers = num_layers

        self.default_graph = default_graph
        self.node_embeddings = nn.Parameter(torch.randn(
            self.num_node, embed_dim), requires_grad=True)

        self.in_gcn = HyperNet(num_nodes, temporal_length, input_dim, Sk, Tk, affine)

        # AVWHyperGCN(dim_in, dim_out, cheb_k, embed_dim, num_nodes, temporal_length, Sk, Tk, affine)
        # self.in_gcn = AVWHyperGCN(input_dim, input_dim, cheb_k, embed_dim, num_nodes, temporal_length, Sk, Tk, affine)

        self.encoder = AVWDCRNN(num_nodes, input_dim, rnn_units, cheb_k,
                                embed_dim, temporal_length, Sk, Tk, affine, num_layers)
        
        self.out_gcn = HyperNet(num_nodes, temporal_length, input_dim, Sk, Tk, affine)

        # predictor
        self.end_conv = nn.Conv2d(
            1, horizon * self.output_dim, kernel_size=(1, self.hidden_dim), bias=True)

        self.init_param()

    def init_param(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
            else:
                nn.init.uniform_(p)

    def forward(self, history_data: torch.Tensor, future_data: torch.Tensor, batch_seen: int, epoch: int, train: bool, **kwargs) -> torch.Tensor:
        """Feedforward function of AGCRN.

        Args:
            history_data (torch.Tensor): inputs with shape [B, L, N, C].

        Returns:
            torch.Tensor: outputs with shape [B, L, N, C]
        """
        # history_data = self.in_gcn(history_data)

        init_state = self.encoder.init_hidden(history_data.shape[0])
        output, _ = self.encoder(
            history_data, init_state, self.node_embeddings, self.default_graph)  # B, T, N, hidden
        output = output[:, -1:, :, :]  # B, 1, N, hidden


        # CNN based predictor
        output = self.end_conv((output))  # B, T*C, N, 1
        output = output.squeeze(-1).reshape(-1, self.horizon,
                                            self.output_dim, self.num_node)
        output = output.permute(0, 1, 3, 2)  # B, T, N, C
        # output = self.out_gcn(output)
        return output + history_data


class HyperAGCRN(AGCRN):
    def __init__(self, num_nodes, input_dim, rnn_units, output_dim, horizon, num_layers, default_graph, embed_dim, cheb_k, temporal_length, Sk, Tk, affine, **model_kwargs):
        super().__init__(num_nodes, input_dim, rnn_units, output_dim, horizon, num_layers, default_graph, embed_dim, cheb_k, temporal_length, Sk, Tk, affine)
        self.use_long_history = model_kwargs.get("use_long_history", False)

    def forward(self, history_data: torch.Tensor, long_history_data: torch.Tensor, future_data: torch.Tensor, epoch: int, batch_seen: int, train: bool, **kwargs) -> torch.Tensor:
        if self.use_long_history:
            return super().forward(long_history_data, future_data, batch_seen=batch_seen, epoch=epoch, train=train, **kwargs)
        else:
            return super().forward(history_data, future_data, batch_seen=batch_seen, epoch=epoch, train=train, **kwargs)
