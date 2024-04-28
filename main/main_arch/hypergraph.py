# -*- coding: utf-8 -*-
""" 
@Time   : 2024/03/10
 
@Author : Shen Fang
"""

import torch
import torch.nn as nn
from model_utils import MLP


class ConcatModel(nn.Module):
    def __init__(self, predictor_model, predictor_args, 
                 backbone_model, backbone_args, backbone_path=None, backbone_freeze=True):
        super().__init__()
        self.backbone_path = backbone_path
        self.backbone_freeze = backbone_freeze

        self.load_backbone(backbone_model, backbone_args)
        self.load_predictor(predictor_model, predictor_args)
    def load_backbone(self, backbone_model: nn.Module, backbone_args: dict):
        self.backbone = backbone_model(**backbone_args)
        
        if self.backbone_path is not None:
            checkpoint_dict = torch.load(self.backbone_path)
            self.backbone.load_state_dict(checkpoint_dict["model_state_dict"])

            if self.backbone_freeze:
                for param in self.backbone.parameters():
                    param.requires_grad = False
    
    def load_predictor(self, predictor_model: nn.Module, predictor_args: dict):
        self.predictor = predictor_model(**predictor_args)
    
    def forward(self, history_data: torch.Tensor, long_history_data: torch.Tensor, future_data: torch.Tensor, epoch: int, batch_seen: int, train: bool, **kwargs):
        hidden_states = self.backbone(history_data=history_data, long_history_data=long_history_data, future_data=future_data, 
                                      epoch=epoch, batch_seen=batch_seen, train=train, **kwargs)
        
        prediction = self.predictor(hidden_states)
        return prediction


class HyperMean(nn.Module):
    def __init__(self, spatial_length: int, temporal_length: int, feature_dim: int, WB: True):
        super().__init__()
        self.WB = WB
        self.normalized_shape = (temporal_length, spatial_length, feature_dim)
        
        if self.WB:
            self.weight = nn.Parameter(torch.empty(self.normalized_shape, dtype=torch.float32))
            self.bias = nn.Parameter(torch.empty(self.normalized_shape, dtype=torch.float32))
        else:
            self.register_parameter('weight', None)
            self.register_parameter('bias', None)

        self.reset_parameters()

    def reset_parameters(self) -> None:
        if self.WB:
            nn.init.ones_(self.weight)
            nn.init.zeros_(self.bias)

    def forward(self, input_features: torch.Tensor):
        """
        input_features: [batch_size, t_length, num_nodes, feature_dim]
        """
        mean = input_features.mean(dim=(-3, -2, -1), keepdim=True)
        var = input_features.var(dim=(-3, -2, -1), keepdim=True)

        output_feature = (input_features - mean) / torch.sqrt(var + 1e-8)  # [batch_size, t_length, num_nodes, feature_dim]
        if self.WB:
            output_feature = output_feature * self.weight + self.bias
            
        return output_feature


class HyperMax(nn.Module):
    def __init__(self, spatial_length: int, temporal_length: int, feature_dim: int, WB: True):
        super().__init__()
        self.WB = WB
        self.normalized_shape = (temporal_length, spatial_length, feature_dim)

        if self.WB:
            self.weight = nn.Parameter(torch.empty(self.normalized_shape, dtype=torch.float32))
            self.bias = nn.Parameter(torch.empty(self.normalized_shape, dtype=torch.float32))
        else:
            self.register_parameter('weight', None)
            self.register_parameter('bias', None)
        
        self.reset_parameters()
    def reset_parameters(self) -> None:
        if self.WB:
            nn.init.ones_(self.weight)
            nn.init.zeros_(self.bias)

    def forward(self, input_features: torch.Tensor):
        max_value = input_features
        for max_dim in range(1,4):
            max_value = max_value.max(dim=max_dim, keepdim=True)
            max_value = max_value.values
    
        var = input_features.var(dim=(-3, -2, -1), keepdim=True)

        output_feature = (input_features - max_value) / torch.sqrt(var + 1e-8)  # [batch_size, t_length, num_nodes, feature_dim]
        if self.WB:
            output_feature = output_feature * self.weight + self.bias
            
        return output_feature


class HyperMin(nn.Module):
    def __init__(self, spatial_length: int, temporal_length: int, feature_dim: int, WB: True):
        super().__init__()
        self.WB = WB
        self.normalized_shape = (temporal_length, spatial_length, feature_dim)

        if self.WB:
            self.weight = nn.Parameter(torch.empty(self.normalized_shape, dtype=torch.float32))
            self.bias = nn.Parameter(torch.empty(self.normalized_shape, dtype=torch.float32))
        else:
            self.register_parameter('weight', None)
            self.register_parameter('bias', None)
        
        self.reset_parameters()

    def reset_parameters(self) -> None:
        if self.WB:
            nn.init.ones_(self.weight)
            nn.init.zeros_(self.bias)

    def forward(self, input_features: torch.Tensor):
        min_value = input_features
        for min_dim in range(1,4):
            min_value = min_value.min(dim=min_dim, keepdim=True)
            min_value = min_value.values
    
        var = input_features.var(dim=(-3, -2, -1), keepdim=True)

        output_feature = (input_features - min_value) / torch.sqrt(var + 1e-8)  # [batch_size, t_length, num_nodes, feature_dim]
        if self.WB:
            output_feature = output_feature * self.weight + self.bias

        return output_feature


class HyperLearn(nn.Module):
    def __init__(self, spatial_length: int, temporal_length: int, feature_dim: int, WB: True):
        super().__init__()
        
        self.WB = WB
        self.spatial_length = spatial_length
        self.temporal_length = temporal_length
        self.feature_dim = feature_dim

        self.normalized_shape = (temporal_length, spatial_length, feature_dim)

        if self.WB:
            self.weight = nn.Parameter(torch.empty(self.normalized_shape, dtype=torch.float32))
            self.bias = nn.Parameter(torch.empty(self.normalized_shape, dtype=torch.float32))
        else:
            self.register_parameter('weight', None)
            self.register_parameter('bias', None)
        self.Tk = 1
        self.Sk = 1

        self.st_linear = nn.Linear(spatial_length * temporal_length * feature_dim, self.Tk * self.Sk)
        
        self.reset_parameters()

    def reset_parameters(self) -> None:
        if self.WB:
            nn.init.ones_(self.weight)
            nn.init.zeros_(self.bias)
        nn.init.ones_(self.st_linear.weight)
        nn.init.zeros_(self.st_linear.bias)

    def forward(self, input_features: torch.Tensor):
        B, T, N, C = input_features.size()  
        
        learned_value = self.st_linear(input_features.reshape(B, -1)).unsqueeze(1).unsqueeze(1)  # [B, N*T*C] -> [B, 1, 1, 1]
        learned_value /= (self.spatial_length * self.temporal_length * self.feature_dim)

        var = input_features.var(dim=(-3, -2, -1), keepdim=True)

        # output_feature = (input_features - learned_value)  # [B, T, N, D]
        output_feature = (input_features - learned_value) / torch.sqrt(var + 1e-8)  # [B, T, N, D]
        
        if self.WB:
            output_feature = output_feature * self.weight + self.bias
        output_feature += input_features
        return output_feature


class HyperLearnFC(nn.Module):
    def __init__(self, spatial_length: int, temporal_length: int, feature_dim: int, WB: True):
        super().__init__()
        
        self.WB = WB
        self.spatial_length = spatial_length
        self.temporal_length = temporal_length
        self.feature_dim = feature_dim

        self.normalized_shape = (temporal_length, spatial_length, feature_dim)

        if self.WB:
            self.weight = nn.Parameter(torch.empty(self.normalized_shape, dtype=torch.float32))
            self.bias = nn.Parameter(torch.empty(self.normalized_shape, dtype=torch.float32))
        else:
            self.register_parameter('weight', None)
            self.register_parameter('bias', None)
        self.Tk = 1
        self.Sk = 1

        self.st_conv = nn.Linear(spatial_length * temporal_length * feature_dim, self.Tk * self.Sk)
        self.st_deconv = nn.Linear(self.Tk * self.Sk, spatial_length * temporal_length * feature_dim)
        
        self.reset_parameters()
    def reset_parameters(self) -> None:
        if self.WB:
            nn.init.ones_(self.weight)
            nn.init.zeros_(self.bias)

        nn.init.ones_(self.st_conv.weight)
        nn.init.zeros_(self.st_conv.bias)
        nn.init.ones_(self.st_deconv.weight)
        nn.init.zeros_(self.st_deconv.bias)
    def forward(self, input_features: torch.Tensor):
        B, T, N, C = input_features.size()  
        
        learned_value = self.st_conv(input_features.reshape(B, -1)).unsqueeze(1).unsqueeze(1)  # [B, N*T*C] -> [B, 1]
        learned_value /= (self.spatial_length * self.temporal_length * self.feature_dim)
        learned_value = self.st_deconv(learned_value)  # [B, 1] -> [B, N*T*C]
        learned_value = learned_value.reshape(B, T, N, C)

        var = input_features.var(dim=(-3, -2, -1), keepdim=True)

        output_feature = (input_features - learned_value) / torch.sqrt(var + 1e-8)  # [B, T, N, D]
        
        if self.WB:
            output_feature = output_feature * self.weight + self.bias
        output_feature += input_features
        return output_feature


class HyperNet(nn.Module):
    def __init__(self, num_nodes: int, temporal_length: int, input_dim: int, s_cluster: int, t_cluster: int , affine: bool=True):
        super().__init__()
        self.affine = affine
        self.num_nodes = num_nodes
        self.temporal_length = temporal_length
        self.input_dim = input_dim

        self.normalized_shape = (temporal_length, num_nodes, input_dim)
        
        if self.affine:
            self.weight = nn.Parameter(torch.empty(self.normalized_shape, dtype=torch.float32))
            self.bias = nn.Parameter(torch.empty(self.normalized_shape, dtype=torch.float32))
        else:
            self.register_parameter('weight', None)
            self.register_parameter('bias', None)
        self.Sk = s_cluster
        self.Tk = t_cluster
        
        self.st_conv = nn.Linear(num_nodes * temporal_length * input_dim, self.Tk * self.Sk)
        self.st_conv_act = nn.LeakyReLU()
        self.st_deconv = nn.Linear(self.Tk * self.Sk, num_nodes * temporal_length * input_dim)
        self.st_deconv_act = nn.LeakyReLU()
        self.reset_parameters()

    def reset_parameters(self) -> None:
        if self.affine:
            nn.init.ones_(self.weight)
            nn.init.zeros_(self.bias)

        nn.init.ones_(self.st_conv.weight)
        nn.init.zeros_(self.st_conv.bias)
        nn.init.ones_(self.st_deconv.weight)
        nn.init.zeros_(self.st_deconv.bias)

    def forward(self, input_features: torch.Tensor):
        B, T, N, C = input_features.size()  
        
        learned_value = self.st_conv(input_features.reshape(B, -1)).unsqueeze(1).unsqueeze(1)  # [B, N*T*C] -> [B, 1]
        learned_value /= (self.num_nodes * self.temporal_length * self.input_dim * self.Sk * self.Tk)
        learned_value = self.st_conv_act(learned_value)

        learned_value = self.st_deconv(learned_value)  # [B, 1] -> [B, N*T*C]
        learned_value = self.st_deconv_act(learned_value)
        
        learned_value = learned_value.reshape(B, T, N, C)

        var = input_features.var(dim=(-3, -2, -1), keepdim=True)

        output_feature = (input_features - learned_value) / torch.sqrt(var + 1e-8)  # [B, T, N, D]
        
        if self.affine:
            output_feature = output_feature * self.weight + self.bias
        output_feature += input_features
        return output_feature


class HyperNetnFC(nn.Module):
    def __init__(self, num_nodes: int, temporal_length: int, input_dim: int, s_cluster: int, t_cluster: int , affine: bool=True):
        super().__init__()
        self.affine = affine
        self.num_nodes = num_nodes
        self.temporal_length = temporal_length
        self.input_dim = input_dim

        self.normalized_shape = (temporal_length, num_nodes, input_dim)
        
        if self.affine:
            self.weight = nn.Parameter(torch.empty(self.normalized_shape, dtype=torch.float32))
            self.bias = nn.Parameter(torch.empty(self.normalized_shape, dtype=torch.float32))
        else:
            self.register_parameter('weight', None)
            self.register_parameter('bias', None)
        self.Sk = s_cluster
        self.Tk = t_cluster

        self.reset_parameters()

    def reset_parameters(self) -> None:
        if self.affine:
            nn.init.ones_(self.weight)
            nn.init.zeros_(self.bias)
    def forward(self, input_features: torch.Tensor):
        B, T, N, C = input_features.size()  
        
        mean_value = torch.mean(input_features, dim=(-3, -2, -1), keepdim=True) # [B, 1, 1, 1]
        var = input_features.var(dim=(-3, -2, -1), keepdim=True)

        output_feature = (input_features - mean_value) / torch.sqrt(var + 1e-8)  # [B, T, N, D]
        
        if self.affine:
            output_feature = output_feature * self.weight + self.bias
        output_feature += input_features
        return output_feature


class HyperNetnS(nn.Module):
    def __init__(self, num_nodes: int, temporal_length: int, input_dim: int, s_cluster: int, t_cluster: int , affine: bool=True):
        super().__init__()
        self.affine = affine
        self.num_nodes = num_nodes
        self.temporal_length = temporal_length
        self.input_dim = input_dim

        self.normalized_shape = (temporal_length, num_nodes, input_dim)
        
        if self.affine:
            self.weight = nn.Parameter(torch.empty(self.normalized_shape, dtype=torch.float32))
            self.bias = nn.Parameter(torch.empty(self.normalized_shape, dtype=torch.float32))
        else:
            self.register_parameter('weight', None)
            self.register_parameter('bias', None)
        self.Sk = s_cluster
        self.Tk = t_cluster
        
        self.st_conv = nn.Linear(temporal_length * input_dim, self.Tk * self.Sk)
        self.st_conv_act = nn.LeakyReLU()
        self.st_deconv = nn.Linear(self.Tk * self.Sk, temporal_length * input_dim)
        self.st_deconv_act = nn.LeakyReLU()
        self.reset_parameters()

    def reset_parameters(self) -> None:
        if self.affine:
            nn.init.ones_(self.weight)
            nn.init.zeros_(self.bias)

        nn.init.ones_(self.st_conv.weight)
        nn.init.zeros_(self.st_conv.bias)
        nn.init.ones_(self.st_deconv.weight)
        nn.init.zeros_(self.st_deconv.bias)

    def forward(self, input_features: torch.Tensor):
        B, T, N, C = input_features.size()  
        input_features = input_features.transpose(1, 2) # [B, N, T, C]

        learned_value = self.st_conv(input_features.reshape(B, N, -1)).unsqueeze(1)  # [B, N, T*C] -> [B, N, 1]
        learned_value /= (self.temporal_length * self.input_dim * self.Sk * self.Tk)
        learned_value = self.st_conv_act(learned_value)

        learned_value = self.st_deconv(learned_value)  # [B, N, 1] -> [B, N, T*C]
        learned_value = self.st_deconv_act(learned_value)
        
        learned_value = learned_value.reshape(B, N, T, C)

        var = input_features.var(dim=(-2, -1), keepdim=True)

        output_feature = (input_features - learned_value) / torch.sqrt(var + 1e-8)  # [B, N, T, D]
        
        output_feature = output_feature.transpose(1, 2)
        if self.affine:
            output_feature = output_feature * self.weight + self.bias
        output_feature += input_features.transpose(1, 2)
        return output_feature


class HyperNetnT(nn.Module):
    def __init__(self, num_nodes: int, temporal_length: int, input_dim: int, s_cluster: int, t_cluster: int , affine: bool=True):
        super().__init__()
        self.affine = affine
        self.num_nodes = num_nodes
        self.temporal_length = temporal_length
        self.input_dim = input_dim

        self.normalized_shape = (temporal_length, num_nodes, input_dim)
        
        if self.affine:
            self.weight = nn.Parameter(torch.empty(self.normalized_shape, dtype=torch.float32))
            self.bias = nn.Parameter(torch.empty(self.normalized_shape, dtype=torch.float32))
        else:
            self.register_parameter('weight', None)
            self.register_parameter('bias', None)
        self.Sk = s_cluster
        self.Tk = t_cluster
        
        self.st_conv = nn.Linear(num_nodes * input_dim, self.Tk * self.Sk)
        self.st_conv_act = nn.LeakyReLU()
        self.st_deconv = nn.Linear(self.Tk * self.Sk, num_nodes * input_dim)
        self.st_deconv_act = nn.LeakyReLU()
        self.reset_parameters()

    def reset_parameters(self) -> None:
        if self.affine:
            nn.init.ones_(self.weight)
            nn.init.zeros_(self.bias)

        nn.init.ones_(self.st_conv.weight)
        nn.init.zeros_(self.st_conv.bias)
        nn.init.ones_(self.st_deconv.weight)
        nn.init.zeros_(self.st_deconv.bias)

    def forward(self, input_features: torch.Tensor):
        B, T, N, C = input_features.size()  

        learned_value = self.st_conv(input_features.reshape(B, T, -1)).unsqueeze(1)  # [B, T, N*C] -> [B, T, 1]
        learned_value /= (self.num_nodes * self.input_dim * self.Sk * self.Tk)
        learned_value = self.st_conv_act(learned_value)

        learned_value = self.st_deconv(learned_value)  # [B, T, 1] -> [B, T, N*C]
        learned_value = self.st_deconv_act(learned_value)
        
        learned_value = learned_value.reshape(B, T, N, C)

        var = input_features.var(dim=(-2, -1), keepdim=True)

        output_feature = (input_features - learned_value) / torch.sqrt(var + 1e-8)  # [B, N, T, D]
        
        if self.affine:
            output_feature = output_feature * self.weight + self.bias
        output_feature += input_features
        return output_feature


if __name__ == "__main__":
    x = torch.randn(32, 6, 278, 2)
    module = HyperMean(278, 6, 2, True)

    y = module(x)

    print(y.shape)
