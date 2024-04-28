# -*- coding: utf-8 -*-
"""
@Time   : 2023/9/26

@Author : Shen Fang
"""
import os 
import sys
sys.path.append(os.path.abspath(__file__ + "/../../"))
from easytorch import launch_runner, Runner
from argparse import ArgumentParser
from utils import get_best_model_file_path, str2bool

def create_args():
    parser =ArgumentParser(description="EasyTorch for time series forecasting test!")
    parser.add_argument("-c", "--cfg", default='model_setting/compare_method/LR/LR_PeMS03NewShort.py', help="training config file path.")
    parser.add_argument("-a", "--auto", default=1, help="automatically search the best ckpt by the cfg file (when turning on, the ck cfg is not used).")
    parser.add_argument("-ck", "--ckpt", default=None, help="the checkpoint file path. if None, load default ckpt in ckpt save dir.")
    parser.add_argument("-g", "--gpus", default="0", help="gpu ids.")

    return parser.parse_args()


def inference(cfg: dict, runner: Runner, ckpt_path: str):
    runner.init_logger(logger_name='easytorch-inference', log_file_name='validate_result')
    
    # runner.model.eval() 
    runner.setup_graph(cfg=cfg, train=False) # On when DCRNN
    
    runner.load_model(ckpt_path=ckpt_path)
    
    runner.test_process(cfg)


if __name__ == "__main__":
    args = create_args()
    auto = args.auto if isinstance(args.auto, int) else str2bool(args.auto)
    
    if not auto:
        if args.ckpt:
            launch_runner(args.cfg, inference, (args.ckpt,), devices=args.gpus)
        else:
            raise ValueError("Ckpt file is not specified, please supply the ckpt file!")
    else:
        model_file = get_best_model_file_path(args.cfg)

        if model_file:
            launch_runner(args.cfg, inference, (model_file,), devices=args.gpus)
        else:
            print("When searching the path, no valid best model found!")
            exit(0)