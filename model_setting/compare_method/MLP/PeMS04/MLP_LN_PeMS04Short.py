import os
import sys
import json

# TODO: remove it when basicts can be installed by pip
sys.path.append(os.path.abspath(__file__ + "/../../.."))
import torch
from easydict import EasyDict

from basic_ts.losses import masked_mae, masked_rmse

from main.main_data.forecast_dataset import ForecastDataset
from main.main_runner.forecast_runner import ForecastRunner

from compare_model.BTS_MLP import MultiLayerPerceptron, MLP_LN

CFG = EasyDict()

# ================= general ================= #
CFG.DESCRIPTION = "MLP model configuration"
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

# ================= environment ================= #
CFG.ENV = EasyDict()
CFG.ENV.SEED = 1
CFG.ENV.CUDNN = EasyDict()
CFG.ENV.CUDNN.ENABLED = True

# ================= model ================= #
CFG.MODEL = EasyDict()
CFG.MODEL.NAME = "MLP_LN_Short"
CFG.MODEL.ARCH = MLP_LN
CFG.MODEL.FORWARD_FEATURES = [0]
CFG.MODEL.TARGET_FEATURES = [0]
CFG.MODEL.INPUT_DIM = len(CFG.MODEL.FORWARD_FEATURES)
CFG.MODEL.OUTPUT_DIM = len(CFG.MODEL.TARGET_FEATURES)

CFG.MODEL.PARAM = {
    "history_seq_len": CFG.DATASET_DESCRIBE.get("src_len"),
    "prediction_seq_len": CFG.DATASET_DESCRIBE.get("trg_len"),
    "hidden_dim": 64,
    "input_dim": CFG.MODEL.INPUT_DIM,
    "output_dim": CFG.MODEL.OUTPUT_DIM,
    "n_layers": 4,
    "num_nodes": CFG.DATASET_DESCRIBE.get("num_nodes"),
    "use_long_history":False
}


# ================= optim ================= #
CFG.TRAIN = EasyDict()
CFG.TRAIN.LOSS = masked_mae
CFG.TRAIN.OPTIM = EasyDict()
CFG.TRAIN.OPTIM.TYPE = "Adam"
CFG.TRAIN.OPTIM.PARAM = {
    "lr": 0.001,
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
CFG.TRAIN.NUM_EPOCHS = 500
CFG.TRAIN.CKPT_SAVE_DIR = os.path.join(
    "result/compare_model/MLP/PeMS04",
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