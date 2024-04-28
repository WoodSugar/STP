# -*- coding: utf-8 -*-
"""
@Time   : 2023/9/15

@Author : Shen Fang
"""
from typing import Union, Dict
import numpy as np
import pandas as pd
import time
import argparse
import pickle
import json
import os
import torch
import csv
from tqdm import tqdm
from fastdtw import fastdtw
from argparse import ArgumentParser

from basic_ts.utils.serialization import load_pkl


import sys
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  
sys.path.append(BASE_DIR)

from data_loader import LoadData, FileStructure, DataStructure
from utils import CreateOption, create_option
from easydict import EasyDict


def gen_data(data, ntr, N):
    '''
    if flag:
        data=pd.read_csv(fname)
    else:
        data=pd.read_csv(fname,header=None)
    '''
    #data=data.as_matrix()
    data=np.reshape(data,[-1,288,N])
    return data[0:ntr]


def normalize(a):
    mu=np.mean(a,axis=1,keepdims=True)
    std=np.std(a,axis=1,keepdims=True)
    return (a-mu)/std


def compute_dtw(a,b,order=1,Ts=12,normal=True):
    if normal:
        a=normalize(a)
        b=normalize(b)
    T0=a.shape[1]
    d=np.reshape(a,[-1,1,T0])-np.reshape(b,[-1,T0,1])
    d=np.linalg.norm(d,axis=0,ord=order)
    D=np.zeros([T0,T0])
    for i in range(T0):
        for j in range(max(0,i-Ts),min(T0,i+Ts+1)):
            if (i==0) and (j==0):
                D[i,j]=d[i,j]**order
                continue
            if (i==0):
                D[i,j]=d[i,j]**order+D[i,j-1]
                continue
            if (j==0):
                D[i,j]=d[i,j]**order+D[i-1,j]
                continue
            if (j==i-Ts):
                D[i,j]=d[i,j]**order+min(D[i-1,j-1],D[i-1,j])
                continue
            if (j==i+Ts):
                D[i,j]=d[i,j]**order+min(D[i-1,j-1],D[i,j-1])
                continue
            D[i,j]=d[i,j]**order+min(D[i-1,j-1],D[i-1,j],D[i,j-1])
    return D[-1,-1]**(1.0/order)


def import_config(path: str) -> Dict:
    """Import config by path

    Examples:
        ```
        cfg = import_config('config/my_config.py')
        ```
        is equivalent to
        ```
        from config.my_config import CFG as cfg
        ```

    Args:
        path (str): Config path

    Returns:
        cfg (Dict): `CFG` in config file
    """

    if path.find('.py') != -1:
        path = path[:path.find('.py')].replace('/', '.').replace('\\', '.')
    cfg_name = path.split('.')[-1]
    cfg = __import__(path, fromlist=[cfg_name]).CFG

    return cfg


# 创建时间dtw-graph
def create_graph(dataset: str):
    model_hyper = json.load(open(os.path.join("model_setting", dataset + "_" + "stfgnn"), "r"))

    dataset_describe = json.load(open(os.path.join("dataset_describe", dataset), "r"))
    parser = argparse.ArgumentParser()
    # parser.add_argument("--dataset", type=str, default="PEMS08", help="Dataset path.")
    # parser.add_argument("--order", type=int, default=1, help="DTW order.")
    # parser.add_argument("--lag", type=int, default=12, help="Fast DTW search lag.")
    # parser.add_argument("--period", type=int, default=96, help="Time series periods.")  # 24 * (60/5) = 288; 16 * (60/10) = 96
    # parser.add_argument("--sparsity", type=float, default=0.01, help="sparsity of spatial graph")

    args = parser.parse_args()
    args.dataset = dataset

    args.order = model_hyper["dtw_order"]
    args.lag = model_hyper["search_lag"]
    args.period = model_hyper["one_day_length"]
    args.sparsity = model_hyper["sparsity"]

    df = np.load(os.path.join(dataset_describe["folder"], dataset_describe["flow_file"]))
    df = df.transpose(1, 0, 2)  # [T, N, D]

    num_samples, ndim, _ = df.shape  # [T, N, D]
    num_train = int(num_samples * 0.6)

    num_dtw = int(num_train/args.period)*args.period

    data = df[:num_dtw,:,:1].reshape([-1,args.period,ndim])

    d=np.zeros([ndim,ndim])

    for i in range(ndim):
        t1=time.time()
        for j in range(i+1,ndim):
            d[i,j]=compute_dtw(data[:,:,i],data[:,:,j],order=args.order,Ts=args.lag)
        t2=time.time()
        print('Line',i,'finished in',t2-t1,'seconds.')

    dtw=d+d.T

    np.save("./dtw_temp/"+ args.dataset+"-dtw-"+str(args.period)+'-'+str(args.order)+"-.npy",dtw)
    print("The calculation of time series is done!")

    adj = np.load("./dtw_temp/"+args.dataset+"-dtw-"+str(args.period)+'-'+str(args.order)+"-.npy")
    adj = adj+ adj.T

    w_adj = np.zeros([ndim,ndim])

    adj_percent = args.sparsity

    top = int(ndim * adj_percent)
    for i in range(adj.shape[0]):
        a = adj[i,:].argsort()[0:top]
        for j in range(top):
            w_adj[i, a[j]] = 1

    for i in range(ndim):
        for j in range(ndim):
            if (w_adj[i][j] != w_adj[j][i] and w_adj[i][j] ==0):
                w_adj[i][j] = 1
            if( i==j):
                w_adj[i][j] = 1

    print("Total route number: ", ndim)
    print("Sparsity of adj: ", len(w_adj.nonzero()[0])/(ndim*ndim))

    dtw_file_name = "./dtw_temp/" + args.dataset + "_graph_dtw.csv"
    pd.DataFrame(w_adj).to_csv(dtw_file_name, index = False, header=None)

    print("The weighted matrix of temporal graph is generated!")


def create_dtw_graph(cfg: Union[Dict, str]):
    cfg = import_config(cfg)
    
    # parser.add_argument("--dataset", type=str, default="PEMS08", help="Dataset path.")
    # parser.add_argument("--order", type=int, default=1, help="DTW order.")
    # parser.add_argument("--lag", type=int, default=12, help="Fast DTW search lag.")
    # parser.add_argument("--period", type=int, default=96, help="Time series periods.")  # 24 * (60/5) = 288; 16 * (60/10) = 96
    # parser.add_argument("--sparsity", type=float, default=0.01, help="sparsity of spatial graph")
    args = EasyDict()
    
    args.dataset = cfg.DATASET_NAME

    args.order = cfg.DTW.DTW_ORDER
    args.lag = cfg.DTW.SEARCH_LAG
    args.period = cfg.DTW.ONE_DAY_LENGTH
    args.sparsity = cfg.DTW.SPARSITY
    
    if cfg.DATASET_DESCRIBE.get("flow_file"):
        data_path = os.path.join(cfg.DATASET_DESCRIBE.get("folder"), cfg.DATASET_DESCRIBE.get("flow_file"))
    elif cfg.DATASET_DESCRIBE.get("speed_file"):
        data_path = os.path.join(cfg.DATASET_DESCRIBE.get("folder"), cfg.DATASET_DESCRIBE.get("speed_file"))

    if not os.path.exists(data_path):
        raise ValueError("Cannot find the correct data file!")
    
    if data_path.split('.')[-1] == 'npy':
        df = np.load(data_path)
        df = df.transpose(1, 0, 2)  # [T, N, D]

    elif data_path.split('.')[-1] == 'npz':
        df = np.load(data_path)["data"]
        
    elif data_path.split('.')[-1] == 'h5':
        df = pd.read_hdf(data_path) # [T, N, D]
        if df.ndim != 3:
            df = np.expand_dims(df.values, axis=-1)  # [T, N, D]   
        else:
            pass
    
    num_samples, ndim, _ = df.shape  # [T, N, D]
    num_train = int(num_samples * cfg.DATASET_DESCRIBE.get("divide_days")[0] / sum(cfg.DATASET_DESCRIBE.get("divide_days") ))

    num_dtw = int(num_train/args.period)*args.period

    data = df[:num_dtw,:,:1].reshape([-1,args.period,ndim])

    d=np.zeros([ndim,ndim])

    for i in range(ndim):
        t1=time.time()
        for j in range(i+1,ndim):
            d[i,j]=compute_dtw(data[:,:,i],data[:,:,j],order=args.order,Ts=args.lag)
        t2=time.time()
        print('Line',i,'finished in',t2-t1,'seconds.')

    dtw=d+d.T

    np.save("./dtw_temp/"+args.dataset+"-dtw-"+str(args.period)+'-'+str(args.order)+"-.npy",dtw)
    print("The calculation of time series is done!")

    adj = np.load("./dtw_temp/"+args.dataset+"-dtw-"+str(args.period)+'-'+str(args.order)+"-.npy")
    adj = adj+ adj.T

    w_adj = np.zeros([ndim,ndim])

    adj_percent = args.sparsity

    top = int(ndim * adj_percent)
    for i in range(adj.shape[0]):
        a = adj[i,:].argsort()[0:top]
        for j in range(top):
            w_adj[i, a[j]] = 1

    for i in range(ndim):
        for j in range(ndim):
            if (w_adj[i][j] != w_adj[j][i] and w_adj[i][j] ==0):
                w_adj[i][j] = 1
            if( i==j):
                w_adj[i][j] = 1

    print("Total route number: ", ndim)
    print("Sparsity of adj: ", len(w_adj.nonzero()[0])/(ndim*ndim))

    dtw_file_name = "./dtw_temp/" + args.dataset + "_graph_dtw.csv"
    pd.DataFrame(w_adj).to_csv(dtw_file_name, index = False, header=None)

    print("The weighted matrix of temporal graph is generated!")



# 创建st-graph
def construct_adj_fusion(A: np.array, A_dtw: np.array, steps:int=4):
    '''
    construct a bigger adjacency matrix using the given matrix

    Parameters
    ----------
    A: np.ndarray, adjacency matrix, shape is (N, N)

    steps: how many times of the does the new adj mx bigger than A

    Returns
    ----------
    new adjacency matrix: csr_matrix, shape is (N * steps, N * steps)

    ----------
    This is 4N_1 mode:

    [T, 1, 1, T
     1, S, 1, 1
     1, 1, S, 1
     T, 1, 1, T]

    '''
    N = len(A)
    adj = np.zeros([N * steps] * 2) # "steps" = 4 !!!

    for i in range(steps):
        if (i == 1) or (i == 2):
            adj[i * N: (i + 1) * N, i * N: (i + 1) * N] = A
        else:
            adj[i * N: (i + 1) * N, i * N: (i + 1) * N] = A_dtw
    #'''
    for i in range(N):
        for k in range(steps - 1):
            adj[k * N + i, (k + 1) * N + i] = 1
            adj[(k + 1) * N + i, k * N + i] = 1
    #'''
    adj[3 * N: 4 * N, 0:  N] = A_dtw #adj[0 * N : 1 * N, 1 * N : 2 * N]
    adj[0 : N, 3 * N: 4 * N] = A_dtw #adj[0 * N : 1 * N, 1 * N : 2 * N]

    adj[2 * N: 3 * N, 0 : N] = adj[0 * N : 1 * N, 1 * N : 2 * N]
    adj[0 : N, 2 * N: 3 * N] = adj[0 * N : 1 * N, 1 * N : 2 * N]
    adj[1 * N: 2 * N, 3 * N: 4 * N] = adj[0 * N : 1 * N, 1 * N : 2 * N]
    adj[3 * N: 4 * N, 1 * N: 2 * N] = adj[0 * N : 1 * N, 1 * N : 2 * N]


    for i in range(len(adj)):
        adj[i, i] = 1

    return adj



# ======== The following is for STGODE ========
def get_normalized_adj(A):
    """
    Returns a tensor, the degree normalized adjacency matrix.
    """
    alpha = 0.8
    D = np.array(np.sum(A, axis=1)).reshape((-1,))
    D[D <= 10e-5] = 10e-5    # Prevent infs
    diag = np.reciprocal(np.sqrt(D))
    A_wave = np.multiply(np.multiply(diag.reshape((-1, 1)), A),
                         diag.reshape((1, -1)))
    A_reg = alpha / 2 * (np.eye(A.shape[0]) + A_wave)
    return torch.from_numpy(A_reg.astype(np.float32))


def generate_dtw_spa_matrix(dataset_name, in_len, out_len, sigma1=0.1, thres1=0.6, sigma2=10, thres2=0.5):
    """read data, generate spatial adjacency matrix and semantic adjacency matrix by dtw

    Args:
        sigma1: float, default=0.1, sigma for the semantic matrix
        sigma2: float, default=10, sigma for the spatial matrix
        thres1: float, default=0.6, the threshold for the semantic matrix
        thres2: float, default=0.5, the threshold for the spatial matrix

    Returns:
        data: tensor, T * N * 1
        dtw_matrix: array, semantic adjacency matrix
        sp_matrix: array, spatial adjacency matrix
    """

    # original STGODE use the full time series to generate the matrices, which is not reasonable since the test set is not available in real world
    data_file = "../data/{0}/data_in{1}_out{2}.pkl".format(dataset_name, in_len, out_len)
    with open(data_file, 'rb') as f:
        data = pickle.load(f)["processed_data"]
    num_node = data.shape[1]
    if not os.path.exists('../data/{0}/dtw_distance_stgode.npy'.format(dataset_name)):
        print("generate dtw distance matrix")
        data_mean = np.mean([data[:, :, 0][24*12*i: 24*12*(i+1)] for i in range(data.shape[0]//(24*12))], axis=0)
        data_mean = data_mean.squeeze().T 
        dtw_distance = np.zeros((num_node, num_node))
        for i in tqdm(range(num_node)):
            for j in range(i, num_node):
                dtw_distance[i][j] = fastdtw(data_mean[i], data_mean[j], radius=6)[0]
        for i in range(num_node):
            for j in range(i):
                dtw_distance[i][j] = dtw_distance[j][i]
        np.save('../data/{0}/dtw_distance_stgode.npy'.format(dataset_name), dtw_distance)

    dist_matrix = np.load('../data/{0}/dtw_distance_stgode.npy'.format(dataset_name))

    mean = np.mean(dist_matrix)
    std = np.std(dist_matrix)
    dist_matrix = (dist_matrix - mean) / std
    sigma = sigma1
    dist_matrix = np.exp(-dist_matrix ** 2 / sigma ** 2)
    dtw_matrix = np.zeros_like(dist_matrix)
    dtw_matrix[dist_matrix > thres1] = 1

    # STGODE provides the scripts to generate spatial matrix for PEMS03, PEMS04, PEMS07, PEMS08
    # For other datasets, we use the original spatial matrix.    
    if dataset_name in ["PEMS03", "PEMS04", "PEMS07", "PEMS08"]:
        if not os.path.exists('{0}/{1}_spatial_distance.npy'.format(os.path.abspath(__file__ + "/.."), dataset_name)):
            graph_csv_file_path = "./datasets/raw_data/{0}/{0}.csv".format(dataset_name)
            with open(graph_csv_file_path, 'r') as fp:
                dist_matrix = np.zeros((num_node, num_node)) + np.float('inf')
                file = csv.reader(fp)
                for line in file:
                    break
                for line in file:
                    start = int(line[0])
                    end = int(line[1])
                    dist_matrix[start][end] = float(line[2])
                    dist_matrix[end][start] = float(line[2])
                np.save('{0}/{1}_spatial_distance.npy'.format(os.path.abspath(__file__ + "/.."), dataset_name), dist_matrix)

        dist_matrix = np.load('{0}/{1}_spatial_distance.npy'.format(os.path.abspath(__file__ + "/.."), dataset_name))
        # normalization
        std = np.std(dist_matrix[dist_matrix != np.float('inf')])
        mean = np.mean(dist_matrix[dist_matrix != np.float('inf')])
        dist_matrix = (dist_matrix - mean) / std
        sigma = sigma2
        sp_matrix = np.exp(- dist_matrix**2 / sigma**2)
        sp_matrix[sp_matrix < thres2] = 0 
    else:
        spatial_distance_file = "./datasets/{0}/adj_mx.pkl".format(dataset_name)
        sp_matrix = load_pkl(spatial_distance_file)[-1]

    print(f'average degree of spatial graph is {np.sum(sp_matrix > 0)/2/num_node}')
    print(f'average degree of semantic graph is {np.sum(dtw_matrix > 0)/2/num_node}')
    # normalize
    dtw_matrix = get_normalized_adj(dtw_matrix)
    sp_matrix = get_normalized_adj(sp_matrix)
    return dtw_matrix, sp_matrix


def create_args():
    parser = ArgumentParser(description="creating the dtw graph according to the config file")
    parser.add_argument("-c", "--cfg", default="model_setting/compare_method/STFGNN/PeMS04/STFGNN_Ori.py", help="creating config file path.")

    return parser.parse_args()


if __name__ == "__main__":
    # create_graph(dataset="MyTaxi")
    # create_graph(dataset="MyBus")

    # create_dtw_graph(cfg="model_setting/compare_method/STFGNN_MyBus.py")
    # create_dtw_graph(cfg="model_setting/compare_method/STFGNN_PeMS03.py")

    # x = pd.read_csv("newdataset/adj_dtw_MySubway.csv", header=None).to_numpy()
    # a = np.load('dataset/MySubway/graph.npy')
    # adj = construct_adj_fusion(a, x)

    args = create_args()
    create_dtw_graph(cfg=args.cfg)
    # "model_setting/compare_method/STFGNN/SD/STFGNN_Ori.py"