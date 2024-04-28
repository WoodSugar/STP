# -*- coding: utf-8 -*-
""" 
@Time   : 2023/11/06
 
@Author : Shen Fang
"""


import os
import sys
import json
import pandas as pd

# TODO: remove it when basicts can be installed by pip
sys.path.append(os.path.abspath(__file__ + "/../../.."))
import torch
from easydict import EasyDict

from basic_ts.losses import masked_mae, masked_rmse
from basic_ts.utils import load_adj

from main.main_data.forecast_dataset import ForecastDataset
from main.main_runner.forecast_runner import ForecastRunner

from ..common import get_uni_graph, get_ran_graph, get_idn_graph
from compare_model.Hyper_STFGNN import HyperSTFGNN

CFG = EasyDict()

# ================= general ================= #
CFG.DESCRIPTION = "STFGNN model configuration"
CFG.RUNNER = ForecastRunner
CFG.DATASET_CLS = ForecastDataset
CFG.DATASET_NAME = "METR_LA"
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

CFG.DTW = EasyDict()

CFG.DTW.DTW_ORDER = 1
CFG.DTW.SEARCH_LAG = 12
CFG.DTW.ONE_DAY_LENGTH = 96
CFG.DTW.SPARSITY = 0.01

# ================= environment ================= #
CFG.ENV = EasyDict()
CFG.ENV.SEED = 1
CFG.ENV.CUDNN = EasyDict()
CFG.ENV.CUDNN.ENABLED = True


# ================= model ================= #
CFG.MODEL = EasyDict()
CFG.MODEL.NAME = "HyperSTFGNN"
CFG.MODEL.ARCH = HyperSTFGNN
CFG.MODEL.FORWARD_FEATURES = [0]
CFG.MODEL.TARGET_FEATURES = [0]
CFG.MODEL.INPUT_DIM = len(CFG.MODEL.FORWARD_FEATURES)
CFG.MODEL.OUTPUT_DIM = len(CFG.MODEL.TARGET_FEATURES)

CFG.LOCAL_STEP = 4

local_adj = get_idn_graph(CFG.DATASET_DESCRIBE.get("num_nodes") * CFG.LOCAL_STEP)


CFG.MODEL.PARAM = {
    "config": {
        "window": CFG.DATASET_INPUT_LEN,
        "num_nodes": CFG.DATASET_DESCRIBE.get("num_nodes"),
        "input_dim": CFG.MODEL.INPUT_DIM,
        "output_dim": CFG.MODEL.OUTPUT_DIM, 
        "hidden_dims": [[64, 64]], 
        "first_layer_embedding_size": 64,
        "out_layer_dim": 64,
        "Sk": 20,
        "Tk": 1,
        "affine": True,
        "activation": "GLU",
        "use_mask": 1,
        "temporal_emb": 1,
        "spatial_emb": 1,
        "horizon": CFG.DATASET_OUTPUT_LEN,
        "strides": CFG.LOCAL_STEP,
    }, 
    "data_feature": {
        "adj_mx": local_adj,
        "scaler": 1, 
        "num_batches": 32,
    }, 
    "use_long_history": False,
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
    "result/compare_model/STFGNN/METR",
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