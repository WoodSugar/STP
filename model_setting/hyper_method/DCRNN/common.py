# -*- coding: utf-8 -*-
""" 
@Time   : 2024/03/18
 
@Author : Shen Fang
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


def get_uni_graph(num_nodes: int, eps=1e-8):
    return torch.ones(num_nodes, num_nodes, dtype=torch.float32) / num_nodes + eps
 
def get_ran_graph(num_nodes: int, eps=1e-8):
    graph = torch.rand(num_nodes, num_nodes, dtype=torch.float32)
    graph = F.softmax(graph, dim=1)
    return graph

def get_idn_graph(num_nodes: int, eps=1e-8):
    graph = torch.eye(num_nodes, dtype=torch.float32)
    return graph