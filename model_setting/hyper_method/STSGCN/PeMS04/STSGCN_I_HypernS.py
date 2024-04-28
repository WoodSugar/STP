# -*- coding: utf-8 -*-
""" 
@Time   : 2023/11/06
 
@Author : Shen Fang
"""


import os
import sys
import json

# TODO: remove it when basicts can be installed by pip
sys.path.append(os.path.abspath(__file__ + "/../../.."))
import torch
from easydict import EasyDict

from basic_ts.losses import masked_mae, masked_rmse
from basic_ts.utils import load_adj

from main.main_data.forecast_dataset import ForecastDataset
from main.main_runner.forecast_runner import ForecastRunner

from compare_model.BTS_STSGCN import STSGCNSeq2Seq, construct_adj
from ..common import get_uni_graph, get_ran_graph, get_idn_graph
from main.main_arch.hypergraph import HyperNet, ConcatModel, HyperNetnS


CFG = EasyDict()

# ================= general ================= #
CFG.DESCRIPTION = "STSGCN model configuration"
CFG.RUNNER = ForecastRunner
CFG.DATASET_CLS = ForecastDataset
CFG.DATASET_NAME = "PeMS04New"
CFG.DATASET_TYPE = "Traffic flow"
CFG.DATASET_DESCRIBE = EasyDict(json.load(open(os.path.join("dataset_describe", CFG.DATASET_NAME), "r")))

CFG.DATASET_INPUT_LEN = CFG.DATASET_DESCRIBE.get("src_len")
CFG.DATASET_OUTPUT_LEN = CFG.DATASET_DESCRIBE.get("trg_len")
CFG.GPU_NUM = 1
CFG.NULL_VAL = 0.0

CFG.DATASET_ARGS = {
    "seq_len": CFG.DATASET_DESCRIBE.get("pretrain_src_len")
}

CFG.GRAPH_PATH = os.path.join(CFG.DATASET_DESCRIBE.get("folder"), "adj_mx.pkl")

# ================= environment ================= #
CFG.ENV = EasyDict()
CFG.ENV.SEED = 1
CFG.ENV.CUDNN = EasyDict()
CFG.ENV.CUDNN.ENABLED = True


# ================= model ================= #
CFG.MODEL = EasyDict()
CFG.MODEL.NAME = "STSGCN_I_HypernS"
CFG.MODEL.ARCH = ConcatModel
CFG.MODEL.FORWARD_FEATURES = [0]
CFG.MODEL.TARGET_FEATURES = [0]
CFG.MODEL.INPUT_DIM = len(CFG.MODEL.FORWARD_FEATURES)
CFG.MODEL.OUTPUT_DIM = len(CFG.MODEL.TARGET_FEATURES)

CFG.LOCAL_STEP = 3

_, adj_mx = load_adj(CFG.GRAPH_PATH, "original")

local_adj = get_idn_graph(num_nodes=CFG.DATASET_DESCRIBE.get("num_nodes") * CFG.LOCAL_STEP)


CFG.MODEL.PARAM = {
    "predictor_model": HyperNetnS,
    "predictor_args": {
        "num_nodes": CFG.DATASET_DESCRIBE.get("num_nodes"), 
        "temporal_length": CFG.DATASET_DESCRIBE.get("trg_len"),
        "input_dim": CFG.MODEL.OUTPUT_DIM, 
        "s_cluster": 20, 
        "t_cluster": 1 ,
        "affine": True,
    },

    "backbone_model": STSGCNSeq2Seq,
    "backbone_args" :{
        "adj": local_adj, 
        "history": CFG.DATASET_INPUT_LEN, 
        "num_of_vertices": CFG.DATASET_DESCRIBE.get("num_nodes"), 
        "in_dim": CFG.MODEL.INPUT_DIM, 
        "out_dim": CFG.MODEL.OUTPUT_DIM, 
        "hidden_dims": [[32, 32]], 
        "first_layer_embedding_size": 32, 
        "out_layer_dim": 32, 
        "activation": 'GLU', 
        "use_mask": True, 
        "temporal_emb": True, 
        "spatial_emb": True, 
        "horizon": CFG.DATASET_OUTPUT_LEN, 
        "strides": 3, 
        "use_long_history": False}
}


# ================= optim ================= #
CFG.TRAIN = EasyDict()
CFG.TRAIN.LOSS = masked_mae
CFG.TRAIN.OPTIM = EasyDict()
CFG.TRAIN.OPTIM.TYPE = "Adam"
CFG.TRAIN.OPTIM.PARAM = {
    "lr": 0.002,
    "weight_decay": 0,
    "eps": 1.0e-8
}

CFG.TRAIN.LR_SCHEDULER = EasyDict()
CFG.TRAIN.LR_SCHEDULER.TYPE = "MultiStepLR"
CFG.TRAIN.LR_SCHEDULER.PARAM = {
    "milestones": [50],
    "gamma": 0.5
}

# ================= train ================= #
CFG.TRAIN.CLIP_GRAD_PARAM = {
    "max_norm": 3.0
}
CFG.TRAIN.NUM_EPOCHS = 300
CFG.TRAIN.CKPT_SAVE_DIR = os.path.join(
    "result/hyper_model/STSGCN/PeMS04",
    "_".join([CFG.MODEL.NAME, CFG.DATASET_NAME, str(CFG.TRAIN.NUM_EPOCHS)])
)

# train data
CFG.TRAIN.DATA = EasyDict()
# read data
CFG.TRAIN.DATA.DIR = CFG.DATASET_DESCRIBE.get("folder")
# dataloader args, optional
CFG.TRAIN.DATA.NUM_WORKERS = 8
CFG.TRAIN.DATA.BATCH_SIZE = 32
CFG.TRAIN.DATA.PREFETCH = False
CFG.TRAIN.DATA.SHUFFLE = True
CFG.TRAIN.DATA.PIN_MEMORY = True

# ================= validate ================= #
CFG.VAL = EasyDict()
CFG.VAL.INTERVAL = 1
# validating data
CFG.VAL.DATA = EasyDict()
# read data
CFG.VAL.DATA.DIR = CFG.TRAIN.DATA.DIR
# dataloader args, optional
CFG.VAL.DATA.BATCH_SIZE = 32
CFG.VAL.DATA.PREFETCH = False
CFG.VAL.DATA.SHUFFLE = False
CFG.VAL.DATA.NUM_WORKERS = 8
CFG.VAL.DATA.PIN_MEMORY = True

# ================= test ================= #
CFG.TEST = EasyDict()
CFG.TEST.INTERVAL = 1
# test data
CFG.TEST.DATA = EasyDict()
# read data
CFG.TEST.DATA.DIR = CFG.TRAIN.DATA.DIR
# dataloader args, optional
CFG.TEST.DATA.BATCH_SIZE = 16
CFG.TEST.DATA.PREFETCH = False
CFG.TEST.DATA.SHUFFLE = False
CFG.TEST.DATA.NUM_WORKERS = 8
CFG.TEST.DATA.PIN_MEMORY = True


# ================= eval ================= #
CFG.EVAL = EasyDict()
CFG.EVAL.HORIZONS = range(1, CFG.DATASET_OUTPUT_LEN+1)