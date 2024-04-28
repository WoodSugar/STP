# -*- coding: utf-8 -*-
""" 
@Time   : 2024/04/08
 
@Author : Shen Fang
"""
import torch
import torch.nn as nn
from main.main_arch.hypergraph import HyperNet, HyperNetnFC, HyperNetnS, HyperNetnT 


class gcn_hyper(nn.Module):
    def __init__(self, adj, in_dim, out_dim, Sk, Tk, affine, activation='GLU', method="default"):
        """
        图卷积模块
        :param adj: 邻接图
        :param in_dim: 输入维度
        :param out_dim: 输出维度
        :param num_vertices: 节点数量
        :param activation: 激活方式 {'relu', 'GLU'}
        """
        super(gcn_hyper, self).__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        
        assert(adj.ndim == 2)
        if method == "default":
            self.gcn = HyperNet(adj.size(0), 1, in_dim, Sk, Tk, affine=affine)
        elif method == "nFC":
            self.gcn = HyperNetnFC(adj.size(0), 1, in_dim, Sk, Tk, affine=affine)
        elif method == "nS":
            self.gcn = HyperNetnS(adj.size(0), 1, in_dim, Sk, Tk, affine=affine)
        elif method == "nT":
            self.gcn = HyperNetnT(adj.size(0), 1, in_dim, Sk, Tk, affine=affine)
        elif method == "nWb":
            self.gcn = HyperNet(adj.size(0), 1, in_dim, Sk, Tk, affine=False)

        self.activation = activation

        assert self.activation in {'GLU', 'relu'}

        if self.activation == 'GLU':
            self.FC = nn.Linear(self.in_dim, 2 * self.out_dim, bias=True)
        else:
            self.FC = nn.Linear(self.in_dim, self.out_dim, bias=True)

    def forward(self, x, mask=None):
        """
        :param x: (3*N, B, Cin)
        :param mask:(3*N, 3*N)
        :return: (3*N, B, Cout)
        """
        x = x.transpose(0, 1)  # [B, 3N, Cin]
        x = x.unsqueeze(1)  # [B, 1, 3N, Cin]
        
        x = self.gcn(x)  # [B, 1, 3N, Cin]
        x = x.squeeze(1).transpose(0, 1) # [4N, B, Cin]

        if self.activation == 'GLU':
            lhs_rhs = self.FC(x)  # 4*N, B, 2*Cout
            lhs, rhs = torch.split(lhs_rhs, self.out_dim, dim=-1)  # 4*N, B, Cout

            out = lhs * torch.sigmoid(rhs)
            del lhs, rhs, lhs_rhs

            return out

        elif self.activation == 'relu':
            return torch.relu(self.FC(x))  # 3*N, B, Cout


class STSGCM(nn.Module):
    def __init__(self, adj, in_dim, out_dims, num_of_vertices, Sk, Tk, affine, activation='GLU', method="default"):
        """
        :param adj: 邻接矩阵
        :param in_dim: 输入维度
        :param out_dims: list 各个图卷积的输出维度
        :param num_of_vertices: 节点数量
        :param activation: 激活方式 {'relu', 'GLU'}
        """
        super(STSGCM, self).__init__()
        self.adj = adj
        self.in_dim = in_dim
        self.out_dims = out_dims
        self.num_of_vertices = num_of_vertices
        self.activation = activation

        self.gcn_operations = nn.ModuleList()

        self.gcn_operations.append(
            gcn_hyper(
                adj=self.adj,
                in_dim=self.in_dim,
                out_dim=self.out_dims[0],
                Sk=Sk, 
                Tk=Tk,
                affine=affine,
                activation=self.activation,
                method=method
            )
        )

        for i in range(1, len(self.out_dims)):
            self.gcn_operations.append(
                gcn_hyper(
                    adj=self.adj,
                    in_dim=self.out_dims[i-1],
                    out_dim=self.out_dims[i],
                    Sk=Sk, 
                    Tk=Tk,
                    affine=affine,
                    activation=self.activation,
                    method=method
                )
            )

    def forward(self, x, mask=None):
        """
        :param x: (3N, B, Cin)
        :param mask: (3N, 3N)
        :return: (N, B, Cout)
        """
        need_concat = []

        for i in range(len(self.out_dims)):
            x = self.gcn_operations[i](x, mask)
            need_concat.append(x)

        # shape of each element is (1, N, B, Cout)
        need_concat = [
            torch.unsqueeze(
                h[self.num_of_vertices: 2 * self.num_of_vertices], dim=0
            ) for h in need_concat
        ]

        out = torch.max(torch.cat(need_concat, dim=0), dim=0).values  # (N, B, Cout)

        del need_concat

        return out


class STSGCL(nn.Module):
    def __init__(self,
                 adj,
                 history,
                 num_of_vertices,
                 in_dim,
                 out_dims,
                 Sk, 
                 Tk, 
                 strides=4,
                 affine=True,
                 activation='GLU',
                 temporal_emb=True,
                 spatial_emb=True,
                 method="default"):
        """
        :param adj: 邻接矩阵
        :param history: 输入时间步长
        :param in_dim: 输入维度
        :param out_dims: list 各个图卷积的输出维度
        :param strides: 滑动窗口步长，local时空图使用几个时间步构建的，默认为3
        :param num_of_vertices: 节点数量
        :param activation: 激活方式 {'relu', 'GLU'}
        :param temporal_emb: 加入时间位置嵌入向量
        :param spatial_emb: 加入空间位置嵌入向量
        """
        super(STSGCL, self).__init__()
        self.adj = adj
        self.strides = strides
        self.history = history
        self.in_dim = in_dim
        self.out_dims = out_dims
        self.num_of_vertices = num_of_vertices

        self.activation = activation
        self.temporal_emb = temporal_emb
        self.spatial_emb = spatial_emb


        self.conv1 = nn.Conv2d(self.in_dim, self.out_dims[-1], kernel_size=(1, 2), stride=(1, 1), dilation=(1, 3))
        self.conv2 = nn.Conv2d(self.in_dim, self.out_dims[-1], kernel_size=(1, 2), stride=(1, 1), dilation=(1, 3))


        self.STSGCMS = nn.ModuleList()
        for i in range(self.history - self.strides + 1):
            self.STSGCMS.append(
                STSGCM(
                    adj=self.adj,
                    in_dim=self.in_dim,
                    out_dims=self.out_dims,
                    num_of_vertices=self.num_of_vertices,
                    Sk=Sk, 
                    Tk=Tk,
                    affine=affine,
                    activation=self.activation,
                    method=method
                )
            )

        if self.temporal_emb:
            self.temporal_embedding = nn.Parameter(torch.FloatTensor(1, self.history, 1, self.in_dim))
            # 1, T, 1, Cin

        if self.spatial_emb:
            self.spatial_embedding = nn.Parameter(torch.FloatTensor(1, 1, self.num_of_vertices, self.in_dim))
            # 1, 1, N, Cin

        self.reset()

    def reset(self):
        if self.temporal_emb:
            nn.init.xavier_normal_(self.temporal_embedding, gain=0.0003)

        if self.spatial_emb:
            nn.init.xavier_normal_(self.spatial_embedding, gain=0.0003)

    def forward(self, x, mask=None):
        """
        :param x: B, T, N, Cin
        :param mask: (N, N)
        :return: B, T-3, N, Cout
        """
        if self.temporal_emb:
            x = x + self.temporal_embedding

        if self.spatial_emb:
            x = x + self.spatial_embedding

        #############################################
        # shape is (B, C, N, T)
        data_temp = x.permute(0, 3, 2, 1)
        data_left = torch.sigmoid(self.conv1(data_temp))
        data_right = torch.tanh(self.conv2(data_temp))
        data_time_axis = data_left * data_right
        data_res = data_time_axis.permute(0, 3, 2, 1)
        # shape is (B, T-3, N, C)
        #############################################

        need_concat = []
        batch_size = x.shape[0]

        for i in range(self.history - self.strides + 1):
            t = x[:, i: i+self.strides, :, :]  # (B, 4, N, Cin)

            t = torch.reshape(t, shape=[batch_size, self.strides * self.num_of_vertices, self.in_dim])
            # (B, 4*N, Cin)

            t = self.STSGCMS[i](t.permute(1, 0, 2), mask)  # (4*N, B, Cin) -> (N, B, Cout)

            t = torch.unsqueeze(t.permute(1, 0, 2), dim=1)  # (N, B, Cout) -> (B, N, Cout) ->(B, 1, N, Cout)

            need_concat.append(t)

        mid_out = torch.cat(need_concat, dim=1)  # (B, T-3, N, Cout)
        out = mid_out + data_res

        del need_concat, batch_size
    
        return out


class output_layer(nn.Module):
    def __init__(self, num_of_vertices, history, in_dim, out_dim,
                 hidden_dim=128, horizon=12):
        """
        预测层，注意在作者的实验中是对每一个预测时间step做处理的，也即他会令horizon=1
        :param num_of_vertices:节点数
        :param history:输入时间步长
        :param in_dim: 输入维度
        :param hidden_dim:中间层维度
        :param horizon:预测时间步长
        """
        super(output_layer, self).__init__()
        self.num_of_vertices = num_of_vertices
        self.history = history
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.hidden_dim = hidden_dim
        self.horizon = horizon

        #print("#####################")
        #print(self.in_dim)
        #print(self.history)
        #print(self.hidden_dim)

        self.FC1 = nn.Linear(self.in_dim * self.history, self.hidden_dim, bias=True)

        #self.FC2 = nn.Linear(self.hidden_dim, self.horizon , bias=True)

        self.FC2 = nn.Linear(self.hidden_dim, self.horizon * self.out_dim, bias=True)

    def forward(self, x):
        """
        :param x: (B, Tin, N, Cin)
        :return: (B, Tout, N)
        """
        batch_size = x.shape[0]

        x = x.permute(0, 2, 1, 3)  # B, N, Tin, Cin

        out1 = torch.relu(self.FC1(x.reshape(batch_size, self.num_of_vertices, -1)))
        # (B, N, Tin, Cin) -> (B, N, Tin * Cin) -> (B, N, hidden)

        out2 = self.FC2(out1)  # (B, N, hidden) -> (B, N, horizon * 2)

        out2 = out2.reshape(batch_size, self.num_of_vertices, self.horizon, self.out_dim)

        del out1, batch_size

        return out2.permute(0, 2, 1, 3)  # B, horizon, N
        # return out2.permute(0, 2, 1)  # B, horizon, N


class STFGNN(nn.Module):
    def __init__(self, config, data_feature):
        """

        :param adj: local时空间矩阵
        :param history:输入时间步长
        :param num_of_vertices:节点数量
        :param in_dim:输入维度
        :param hidden_dims: lists, 中间各STSGCL层的卷积操作维度
        :param first_layer_embedding_size: 第一层输入层的维度
        :param out_layer_dim: 输出模块中间层维度
        :param activation: 激活函数 {relu, GlU}
        :param use_mask: 是否使用mask矩阵对adj进行优化
        :param temporal_emb:是否使用时间嵌入向量
        :param spatial_emb:是否使用空间嵌入向量
        :param horizon:预测时间步长
        :param strides:滑动窗口步长，local时空图使用几个时间步构建的，默认为4
        """
        super(STFGNN, self).__init__()

        self.config = config
        self.data_feature = data_feature
        self.scaler = data_feature["scaler"]
        self.num_batches = data_feature["num_batches"]

        adj = self.data_feature["adj_mx"]
        history = self.config.get("window", 6)
        num_of_vertices = self.config.get("num_nodes", None)
        in_dim = self.config.get("input_dim", 1)
        out_dim = self.config.get("output_dim", 1)
        hidden_dims = self.config.get("hidden_dims", None)
        first_layer_embedding_size = self.config.get("first_layer_embedding_size", None)
        out_layer_dim = self.config.get("out_layer_dim", None)
        activation = self.config.get("activation", "GLU")
        use_mask = self.config.get("mask")
        temporal_emb = self.config.get("temporal_emb", True)
        spatial_emb = self.config.get("spatial_emb", True)
        horizon = self.config.get("horizon", 12)
        strides = self.config.get("strides", 4)

        Sk = self.config.get("Sk", 20)
        Tk = self.config.get("Tk", 1)
        affine = self.config.get("affine", True)
        method = self.config.get("method", "default")

        self.adj = adj
        self.num_of_vertices = num_of_vertices
        self.hidden_dims = hidden_dims
        self.out_layer_dim = out_layer_dim
        self.activation = activation
        self.use_mask = use_mask

        self.temporal_emb = temporal_emb
        self.spatial_emb = spatial_emb
        self.horizon = horizon
        self.strides = strides

        self.First_FC = nn.Linear(in_dim, first_layer_embedding_size, bias=True)
        self.STSGCLS = nn.ModuleList()
        #print("____________________")
        #print(history)

        self.STSGCLS.append(
            STSGCL(
                adj=self.adj,
                history=history,
                num_of_vertices=self.num_of_vertices,
                in_dim=first_layer_embedding_size,
                out_dims=self.hidden_dims[0],
                strides=self.strides,
                Sk=Sk, 
                Tk=Tk,
                affine=affine,
                activation=self.activation,
                temporal_emb=self.temporal_emb,
                spatial_emb=self.spatial_emb,
                method=method
            )
        )

        in_dim = self.hidden_dims[0][-1]
        history -= (self.strides - 1)

        #print("!!!!!!!!!!!!!!!!!!!")
        #print(history)

        for idx, hidden_list in enumerate(self.hidden_dims):
            #print("?????? ", idx)
            if idx == 0:
                continue
            #print("---------", idx)
            self.STSGCLS.append(
                STSGCL(
                    adj=self.adj,
                    history=history,
                    num_of_vertices=self.num_of_vertices,
                    in_dim=in_dim,
                    out_dims=hidden_list,
                    strides=self.strides,
                    Sk=Sk, 
                    Tk=Tk,
                    affine=affine,
                    activation=self.activation,
                    temporal_emb=self.temporal_emb,
                    spatial_emb=self.spatial_emb,
                    method=method
                )
            )
            history -= (self.strides - 1)
            in_dim = hidden_list[-1]

        self.predictLayer = nn.ModuleList()
        #print("***********************")
        #print(history)
        for t in range(self.horizon):
            self.predictLayer.append(
                output_layer(
                    num_of_vertices=self.num_of_vertices,
                    history=history,
                    in_dim=in_dim,
                    out_dim = out_dim,
                    hidden_dim=out_layer_dim,
                    horizon=1
                )
            )

        if self.use_mask:
            mask = torch.zeros_like(self.adj)
            mask[self.adj != 0] = self.adj[self.adj != 0]
            self.mask = nn.Parameter(mask)
        else:
            self.mask = None

    def forward(self, input_data, **kwargs):
        """
        :param x: B, Tin, N, Cin)
        :return: B, Tout, N
        """
        device = kwargs["device"]
        x = input_data["flow_d0_x"].to(device)
        
        x = x.permute(0, 2, 1, 3)  # B, N, T, C -> B, T, N, C

        x = torch.relu(self.First_FC(x))  # B, Tin, N, Cin
        #print(1)

        for model in self.STSGCLS:
            x = model(x, self.mask)
        # (B, T - 8, N, Cout)
        #print(2)
        need_concat = []
        for i in range(self.horizon):
            out_step = self.predictLayer[i](x)  # (B, 1, N, 2)
            need_concat.append(out_step)
        #print(3)
        out = torch.cat(need_concat, dim=1)  # B, Tout, N, 2

        del need_concat
        
        out = out.permute(0, 2, 1, 3)  # B, T, N, C -> B, N, T, C
        target = input_data["flow_y"].to(device)
        
        return out, target
    

class HyperSTFGNN(STFGNN):
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