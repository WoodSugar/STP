# -*- coding: utf-8 -*-
"""
@Time   : 2023/9/26

@Author : Shen Fang
"""
import os
import sys
sys.path.append(os.path.abspath(__file__ + "/../../"))

from argparse import ArgumentParser
import torch
from basic_ts import launch_training
from easytorch import launch_runner, Runner
from utils import get_best_model_file_path, str2bool

torch.set_num_threads(4)  # aviod high cpu avg usage


def create_args():
    parser = ArgumentParser(description="Run time series forecasting model in BasicTS framework!")
    parser.add_argument("-c", "--cfg", default='model_setting/hyper_method/DCRNN/PeMS04/HyperDCRNN.py', help="training config file path.")
    parser.add_argument("-g", "--gpus", default="0", help="visible gpus id.")
    parser.add_argument("-t", "--test", default=1, help="run the best val model when training finished.")
    return parser.parse_args()


def inference(cfg: dict, runner: Runner, ckpt_path: str):
    runner.init_logger(logger_name='easytorch-inference', log_file_name='validate_result')

    # runner.model.eval() 
    runner.setup_graph(cfg=cfg, train=False) # On when DCRNN
    
    runner.load_model(ckpt_path=ckpt_path)

    runner.test_process(cfg)


if __name__ == "__main__":
    args = create_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpus

    launch_training(args.cfg, args.gpus)

    auto_test = args.test if isinstance(args.test, int) else str2bool(args.test)
    
    if auto_test:
        model_file = get_best_model_file_path(args.cfg)
        
        if model_file:
            launch_runner(args.cfg, inference, (model_file,), devices=args.gpus)
            print("Load best val model from:", model_file)
        else:
            print("When searching the path, no valid best model found!")
            exit(0)
    else:
        pass
