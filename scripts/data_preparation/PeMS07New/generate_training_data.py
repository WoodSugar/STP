# -*- coding: utf-8 -*-
""" 
@Time   : 2024/03/06
 
@Author : Shen Fang
"""

import os
import sys
import shutil
import pickle
import argparse
import json
import numpy as np

from generate_adj_mx import generate_adj_pems07
# TODO: remove it when basicts can be installed by pip
sys.path.append(os.path.abspath(__file__ + "/../../../.."))
from basic_ts.data.transform import standard_transform
from utils import str2bool, sum_byindex, choose_data_one_day
from easydict import EasyDict


def generate_data_all(args: argparse.Namespace):
    """Preprocess and generate train/valid/test datasets.

    Args:
        args (argparse): configurations of preprocessing
    """
    data_info_folder = args.data_info_folder
    data_info = json.load(open(os.path.join(data_info_folder, "PeMS07New"), "r"))
    data_info = EasyDict(data_info)

    src_len = args.src_len if args.src_len is not None else data_info.src_len if not args.pre_train else data_info.pretrain_src_len
    trg_len = args.trg_len if args.trg_len is not None else data_info.trg_len if not args.pre_train else data_info.pretrain_trg_len

    output_dir = args.output_dir

    add_time_of_day = args.add_time_of_day
    add_day_of_week = args.add_day_of_week

    merge_number = data_info.merge_num
    time_interval = data_info.time_interval * merge_number

    flow_file = os.path.join(data_info.folder, data_info.flow_file)

    flow_data = np.load(flow_file)["data"]  # [T, N, C]

    merge_index = []
    for i in range(0, flow_data.shape[0], merge_number):  # [0, 2, 4, ... , 98]  
        merge_index.append([i + tau for tau in range(merge_number)])
    print("T length / merger number = ", flow_data.shape[0] / merge_number)

    flow_data = sum_byindex(flow_data, merge_index, axis=0)

    print("Raw data shape: {0}".format(flow_data.shape))
    total_days = data_info.divide_days
    flow_data = choose_data_one_day(flow_data, time_interval, data_info.one_day_range, sum(total_days))
    print("Choosed data shape: {0}".format(flow_data.shape))

    num_samples = flow_data.shape[0] - (src_len + trg_len) + 1
    num_nodes = flow_data.shape[1]

    total_days = data_info.divide_days
    train_num = round(num_samples * data_info.divide_days[0] / sum(total_days))
    valid_num = round(num_samples * data_info.divide_days[1] / sum(total_days))
    test_num = num_samples - train_num - valid_num

    print("number of training samples:{0}".format(train_num))
    print("number of validation samples:{0}".format(valid_num))
    print("number of test samples:{0}".format(test_num))

    index_list = []

    for t in range(src_len, num_samples + src_len):
        index = (t - src_len, t, t + trg_len)
        index_list.append(index)

    train_index = index_list[:train_num]
    valid_index = index_list[train_num: train_num + valid_num]
    test_index = index_list[train_num +
                            valid_num: train_num + valid_num + test_num]

    scaler = standard_transform

    data_norm = scaler(flow_data, output_dir, train_index, src_len, trg_len,
                       norm_each_channel=False)

    feature_list = [data_norm]
    steps_per_day = int((data_info.one_day_range[1] - data_info.one_day_range[0]) * 60 / time_interval)

    if add_time_of_day:
        tod = [i % steps_per_day /
               steps_per_day for i in range(data_norm.shape[0])]
        tod = np.array(tod)
        tod_tiled = np.tile(tod, [1, data_info.num_nodes, 1]).transpose((2, 1, 0))
        feature_list.append(tod_tiled)

    if add_day_of_week:
        pass
        dow = [(i // steps_per_day) % 7 / 7 for i in range(data_norm.shape[0])]
        dow = np.array(dow)
        dow_tiled = np.tile(dow, [1, data_info.num_nodes, 1]).transpose((2, 1, 0))
        feature_list.append(dow_tiled)

    processed_data = np.concatenate(feature_list, axis=-1)

    index = {}
    index["train"] = train_index
    index["valid"] = valid_index
    index["test"] = test_index
    with open(output_dir + "/index_in{0}_out{1}.pkl".format(src_len, trg_len), "wb") as f:
        pickle.dump(index, f)

    data = {}
    data["processed_data"] = processed_data
    with open(output_dir + "/data_in{0}_out{1}.pkl".format(src_len, trg_len), "wb") as f:
        pickle.dump(data, f)


    ori_graph_flie = os.path.join(data_info.folder, data_info.ori_graph_file)
    out_graph_flie = os.path.join(data_info.folder, data_info.graph_file)

    with open(ori_graph_flie, "rb") as f:
        graph = pickle.load(f)

    with open(output_dir + "/adj_mx.pkl", "wb") as f:
        pickle.dump(graph, f)

    if type(graph) == list:
        if type(graph[-1]) == np.ndarray:
            np.save(out_graph_flie, graph[-1])
        else:
            print("graph type error and the save failed, type: {}, please check the graph file".format(type(graph)))
    elif type(graph) == np.ndarray:
            np.save(out_graph_flie, graph)
    else:
        print("graph type error and the save failed, type: {}, please check the graph file".format(type(graph)))


def generate_data(args: argparse.Namespace):
    """Preprocess and generate train/valid/test datasets.

    Args:
        args (argparse): configurations of preprocessing
    """

    target_channel = args.target_channel
    future_seq_len = args.future_seq_len
    history_seq_len = args.history_seq_len
    add_time_of_day = args.tod
    add_day_of_week = args.dow
    output_dir = args.output_dir
    train_ratio = args.train_ratio
    valid_ratio = args.valid_ratio
    data_file_path = args.data_file_path
    graph_file_path = args.graph_file_path
    steps_per_day = args.steps_per_day
    norm_each_channel = args.norm_each_channel
    if_rescale = not norm_each_channel # if evaluate on rescaled data. see `basicts.runner.base_tsf_runner.BaseTimeSeriesForecastingRunner.build_train_dataset` for details.

    # read data
    data = np.load(data_file_path)["data"]
    data = data[..., target_channel]
    print("raw time series shape: {0}".format(data.shape))

    # split data
    l, n, f = data.shape
    num_samples = l - (history_seq_len + future_seq_len) + 1
    train_num = round(num_samples * train_ratio)
    valid_num = round(num_samples * valid_ratio)
    test_num = num_samples - train_num - valid_num
    print("number of training samples:{0}".format(train_num))
    print("number of validation samples:{0}".format(valid_num))
    print("number of test samples:{0}".format(test_num))

    index_list = []
    for t in range(history_seq_len, num_samples + history_seq_len):
        index = (t-history_seq_len, t, t+future_seq_len)
        index_list.append(index)

    train_index = index_list[:train_num]
    valid_index = index_list[train_num: train_num + valid_num]
    test_index = index_list[train_num +
                            valid_num: train_num + valid_num + test_num]

    # normalize data
    scaler = standard_transform
    data_norm = scaler(data, output_dir, train_index, history_seq_len, future_seq_len, norm_each_channel=norm_each_channel)

    # add temporal feature
    feature_list = [data_norm]
    if add_time_of_day:
        # numerical time_of_day
        tod = [i % steps_per_day /
               steps_per_day for i in range(data_norm.shape[0])]
        tod = np.array(tod)
        tod_tiled = np.tile(tod, [1, n, 1]).transpose((2, 1, 0))
        feature_list.append(tod_tiled)

    if add_day_of_week:
        # numerical day_of_week
        dow = [(i // steps_per_day) % 7 / 7 for i in range(data_norm.shape[0])]
        dow = np.array(dow)
        dow_tiled = np.tile(dow, [1, n, 1]).transpose((2, 1, 0))
        feature_list.append(dow_tiled)

    processed_data = np.concatenate(feature_list, axis=-1)

    # save data
    index = {}
    index["train"] = train_index
    index["valid"] = valid_index
    index["test"] = test_index
    with open(output_dir + "/index_in_{0}_out_{1}_rescale_{2}.pkl".format(history_seq_len, future_seq_len, if_rescale), "wb") as f:
        pickle.dump(index, f)

    data = {}
    data["processed_data"] = processed_data
    with open(output_dir + "/data_in_{0}_out_{1}_rescale_{2}.pkl".format(history_seq_len, future_seq_len, if_rescale), "wb") as f:
        pickle.dump(data, f)
    # copy adj
    if os.path.exists(args.graph_file_path):
        # copy
        shutil.copyfile(args.graph_file_path, output_dir + "/adj_mx.pkl")
    else:
        # generate and copy
        generate_adj_pems07()
        shutil.copyfile(args.graph_file_path, output_dir + "/adj_mx.pkl")


if __name__ == "__main__":
    OUTPUT_DIR = "dataset/PeMS07New"
    DATA_INFO_DIR = "dataset_describe"
    
    parser = argparse.ArgumentParser()

    parser.add_argument("--output_dir", type=str, default=OUTPUT_DIR, help="Output directory.")
    parser.add_argument("--data_info_folder", type=str, default=DATA_INFO_DIR, help=" dataset describe directory.")

    parser.add_argument("--src_len", type=int, default=None, help="Source length.")
    parser.add_argument("--trg_len", type=int, default=None, help="Target length.")

    parser.add_argument("--add_time_of_day", type=bool, default=False, help="Add time of day.")
    parser.add_argument("--add_day_of_week", type=bool, default=False, help="Add day of week.")

    parser.add_argument("--pre_train", type=str2bool, default=True, help="whether to generate pre-train dataset or not.")

    args = parser.parse_args()

    generate_data_all(args)


"""
if __name__ == "__main__":
    # sliding window size for generating history sequence and target sequence
    HISTORY_SEQ_LEN = 12
    FUTURE_SEQ_LEN = 12

    TRAIN_RATIO = 0.6
    VALID_RATIO = 0.2
    TARGET_CHANNEL = [0]                   # target channel(s)
    STEPS_PER_DAY = 288

    DATASET_NAME = "PEMS07"
    TOD = True                  # if add time_of_day feature
    DOW = True                  # if add day_of_week feature

    OUTPUT_DIR = "datasets/" + DATASET_NAME
    DATA_FILE_PATH = "datasets/raw_data/{0}/{0}.npz".format(DATASET_NAME)
    GRAPH_FILE_PATH = "datasets/raw_data/{0}/adj_{0}.pkl".format(DATASET_NAME)

    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", type=str,
                        default=OUTPUT_DIR, help="Output directory.")
    parser.add_argument("--data_file_path", type=str,
                        default=DATA_FILE_PATH, help="Raw traffic readings.")
    parser.add_argument("--graph_file_path", type=str,
                        default=GRAPH_FILE_PATH, help="Raw traffic readings.")
    parser.add_argument("--history_seq_len", type=int,
                        default=HISTORY_SEQ_LEN, help="Sequence Length.")
    parser.add_argument("--future_seq_len", type=int,
                        default=FUTURE_SEQ_LEN, help="Sequence Length.")
    parser.add_argument("--steps_per_day", type=int,
                        default=STEPS_PER_DAY, help="Sequence Length.")
    parser.add_argument("--tod", type=bool, default=TOD,
                        help="Add feature time_of_day.")
    parser.add_argument("--dow", type=bool, default=DOW,
                        help="Add feature day_of_week.")
    parser.add_argument("--target_channel", type=list,
                        default=TARGET_CHANNEL, help="Selected channels.")
    parser.add_argument("--train_ratio", type=float,
                        default=TRAIN_RATIO, help="Train ratio")
    parser.add_argument("--valid_ratio", type=float,
                        default=VALID_RATIO, help="Validate ratio.")
    parser.add_argument("--norm_each_channel", type=float, help="Validate ratio.")
    args = parser.parse_args()

    # print args
    print("-"*(20+45+5))
    for key, value in sorted(vars(args).items()):
        print("|{0:>20} = {1:<45}|".format(key, str(value)))
    print("-"*(20+45+5))

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    args.norm_each_channel = True
    generate_data(args)
    args.norm_each_channel = False
    generate_data(args)
"""