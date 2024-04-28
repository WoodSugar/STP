# -*- coding: utf-8 -*-
"""
@Time   : 2020/7/2

@Author : Shen Fang
"""
import os
import csv
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader


class ReadData:
    @staticmethod
    def read_data(flow_file):
        return np.load(flow_file)


class FileStructure:
    def __init__(self, data_folder, **kwargs):
        self.flow_file = os.path.join(data_folder, kwargs["flow_file"])

        self.graph_file = os.path.join(data_folder, kwargs["graph_file"])

        self.poi_file = os.path.join(data_folder, kwargs["poi_file"]) if kwargs["poi_file"] else None

        self.wea_file = os.path.join(data_folder, kwargs["wea_file"]) if kwargs["wea_file"] else None

        self.time_file = os.path.join(data_folder, kwargs["time_file"]) if kwargs["time_file"] else None


class DataStructure:
    def __init__(self, num_nodes, divide_days, one_day_range, time_interval, merge_number, src_length, trg_length):
        self.num_nodes = num_nodes
        self.divide_days = divide_days
        self.one_day_range = one_day_range
        self.time_interval = time_interval
        self.merge_number = merge_number
        self.src_length = src_length
        self.trg_length = trg_length


class LoadData(Dataset):
    def __init__(self, file_structure: FileStructure, data_structure: DataStructure, mode: str):
        self.file_structure = file_structure
        self.data_structure = data_structure
        self.mode = mode

        self.graph_data = ReadData.read_data(file_structure.graph_file)

        self.flow_data = LoadData.merge_data(ReadData.read_data(file_structure.flow_file), 1,
                                             self.data_structure.merge_number, "sum")

        if file_structure.poi_file:
            self.poi_data = ReadData.read_data(file_structure.poi_file)
        else:
            self.poi_data = None

        if file_structure.wea_file:
            self.wea_data = ReadData.read_data(file_structure.wea_file)
        else:
            self.wea_data = None

        if file_structure.time_file:
            self.time_data = ReadData.read_data(file_structure.time_file)
        else:
            self.time_data = None

        self.flow_norm, self.flow_data = self.pre_process_data(self.flow_data, temporal_dim=1, norm_dim=1)

        if self.poi_data is not None:
            self.poi_norm, self.poi_data = self.pre_process_data(self.poi_data, temporal_dim=None, norm_dim=0)
        if self.wea_data is not None:

            self.wea_norm, self.wea_data = self.pre_process_data(self.wea_data, temporal_dim=0, norm_dim=0)
        if self.time_data is not None:
            self.time_norm, self.time_data = self.pre_process_data(self.time_data, temporal_dim=0, norm_dim=0)

    def __len__(self):
        one_day_range = (self.data_structure.one_day_range[1] - self.data_structure.one_day_range[0]) \
                        * int(60 / (self.data_structure.time_interval * self.data_structure.merge_number))

        train_sub_number = self.data_structure.src_length + self.data_structure.trg_length + 2 * one_day_range
        valid_sub_number = self.data_structure.trg_length

        if self.mode == "train":
            return self.data_structure.divide_days[0] * one_day_range - train_sub_number
        elif self.mode == "valid":
            return self.data_structure.divide_days[1] * one_day_range - valid_sub_number
        elif self.mode == "test":
            return self.data_structure.divide_days[2] * one_day_range - valid_sub_number
        else:
            raise ValueError("train model is not defined.")

    def __getitem__(self, index):
        one_day_length = (self.data_structure.one_day_range[1] - self.data_structure.one_day_range[0]) \
                         * int(60 / (self.data_structure.time_interval * self.data_structure.merge_number))

        if self.mode == "train":
            index = index
        elif self.mode == "valid":
            index += self.data_structure.divide_days[0] * one_day_length
        elif self.mode == "test":
            index += (self.data_structure.divide_days[0] + self.data_structure.divide_days[1]) * one_day_length
        else:
            raise ValueError("train model is not defined.")

        flow_d0_x, flow_d1_x, flow_d2_x, flow_y = self.slice_sample(self.flow_data, self.data_structure.src_length,
                                                                    self.data_structure.trg_length, index, 1, self.mode)

        data_dict = {"flow_d0_x": LoadData.to_tensor(flow_d0_x),    # [N, SRC_len, C]
                     "flow_d1_x": LoadData.to_tensor(flow_d1_x),    # [N, SRC_len, C]
                     "flow_d2_x": LoadData.to_tensor(flow_d2_x),    # [N, SRC_len, C]
                     "flow_y":    LoadData.to_tensor(flow_y),       # [N, TRG_len, C]

                     "graph": LoadData.to_tensor(self.graph_data)}  # [N, N]

        if self.poi_data is not None:
            data_dict["poi"] = LoadData.to_tensor(self.poi_data)  # [N, C]

        if self.wea_data is not None:
            wea_d0_x, wea_d1_x, wea_d2_x, wea_y = self.slice_sample(self.wea_data,
                                                                    self.data_structure.src_length,
                                                                    self.data_structure.trg_length, index, 0, self.mode)

            data_dict["wea_d0_x"] = LoadData.to_tensor(wea_d0_x)  # [SRC_len, C]
            data_dict["wea_d1_x"] = LoadData.to_tensor(wea_d1_x)  # [SRC_len, C]
            data_dict["wea_d2_x"] = LoadData.to_tensor(wea_d2_x)  # [SRC_len, C]

            data_dict["wea_y"] = LoadData.to_tensor(wea_y)        # [TRG_len, C]

        if self.time_data is not None:
            time_d0_x, time_d1_x, time_d2_x, time_y = self.slice_sample(self.time_data,
                                                                        self.data_structure.src_length,
                                                                        self.data_structure.trg_length, index, 0, self.mode)

            data_dict["time_d0_x"] = LoadData.to_tensor(time_d0_x)  # [SRC_len, C]
            data_dict["time_d1_x"] = LoadData.to_tensor(time_d1_x)  # [SRC_len, C]
            data_dict["time_d2_x"] = LoadData.to_tensor(time_d2_x)  # [SRC_len, C]

            data_dict["time_y"] = LoadData.to_tensor(time_y)        # [TRG_len, C]

        return data_dict

    def select_each_day(self, data, temporal_dim):
        """
        Select the data in [one_day_range[0], one_day_range[1]] during each day.

        :param data: np.array, [*, T, *].
        :param temporal_dim: int,
        :return:
            np.array, [*, T1:T2, *]
        """
        records_each_hour = int(60 / (self.data_structure.time_interval * self.data_structure.merge_number))
        one_day_range = self.data_structure.one_day_range
        total_days = sum(self.data_structure.divide_days)

        result_data = None

        if temporal_dim == 0:
            list_result = [data[(day * 24 + one_day_range[0]) * records_each_hour:
                                (day * 24 + one_day_range[1]) * records_each_hour] for day in range(total_days)]
            result_data = np.concatenate(list_result, axis=0)

        if temporal_dim == 1:
            list_result = [data[:, (day * 24 + one_day_range[0]) * records_each_hour:
                                   (day * 24 + one_day_range[1]) * records_each_hour] for day in range(total_days)]
            result_data = np.concatenate(list_result, axis=1)

        if temporal_dim == 2:
            list_result = [data[:, :, (day * 24 + one_day_range[0]) * records_each_hour:
                                      (day * 24 + one_day_range[1]) * records_each_hour] for day in range(total_days)]
            result_data = np.concatenate(list_result, axis=2)

        if temporal_dim not in [0, 1, 2]:
            raise ValueError("Temporal dimension is not correct.")

        return result_data

    def slice_data(self, data, history_length, future_length, index, temporal_dim, train_mode, time_mode):
        one_day_length = (self.data_structure.one_day_range[1] - self.data_structure.one_day_range[0]) \
                         * int(60 / (self.data_structure.time_interval * self.data_structure.merge_number))

        if train_mode == "train":
            if time_mode == "day_2":
                index = index + history_length
            elif time_mode == "day_1":
                index = index + one_day_length + history_length
            elif time_mode == "day_0":
                index = index + 2 * one_day_length
            else:
                raise ValueError("Time mode is not defined.")

        elif train_mode in ["valid", "test"]:
            if time_mode == "day_2":
                index = index - 2 * one_day_length
            elif time_mode == "day_1":
                index = index - one_day_length
            elif time_mode == "day_0":
                index = index - history_length
            else:
                raise ValueError("Time mode is not defined")

        else:
            raise ValueError("train model is not defined.")

        start_index = index
        if time_mode in ["day_1", "day_2"]:
            end_index = start_index + future_length
        else:
            end_index = start_index + history_length

        if temporal_dim == 0:
            data_x = data[start_index: end_index]
        elif temporal_dim == 1:
            data_x = data[:, start_index: end_index]
        elif temporal_dim == 2:
            data_x = data[:, :, start_index: end_index]
        else:
            raise ValueError("temporal dimension is not correct.")
        return data_x

    def slice_target(self, data, history_length, future_length, index, temporal_dim, train_mode):
        one_day_length = (self.data_structure.one_day_range[1] - self.data_structure.one_day_range[0]) \
                         * int(60 / (self.data_structure.time_interval * self.data_structure.merge_number))

        if train_mode == "train":
            index = index + history_length + 2 * one_day_length
        elif train_mode in ["valid", "test"]:
            index = index
        else:
            raise ValueError("train model is not defined.")

        start_index = index
        end_index = start_index + future_length

        if temporal_dim == 0:
            data_y = data[start_index: end_index]
        elif temporal_dim == 1:
            data_y = data[:, start_index: end_index]
        elif temporal_dim == 2:
            data_y = data[:, :, start_index: end_index]
        else:
            raise NotImplementedError("temporal dimension  is not correct.")

        return data_y

    def slice_sample(self, data, history_length, future_length, index, temporal_dim, train_mode):
        day_0_data = self.slice_data(data, history_length, future_length, index, temporal_dim,
                                     time_mode="day_0", train_mode=train_mode)

        day_1_data = self.slice_data(data, history_length, future_length, index, temporal_dim,
                                     time_mode="day_1", train_mode=train_mode)

        day_2_data = self.slice_data(data, history_length, future_length, index, temporal_dim,
                                     time_mode="day_2", train_mode=train_mode)

        target_data = self.slice_target(data, history_length, future_length, index, temporal_dim, train_mode)

        return day_0_data, day_1_data, day_2_data, target_data

    def pre_process_data(self, data, temporal_dim, norm_dim):
        """
        :param data: np.array, original traffic data without normalization.
        :param temporal_dim: int, temporal dimension.
        :param norm_dim: int, normalization dimension.
        :return:
            norm_base: list, [max_data, min_data], data of normalization base.
            norm_data: np.array, normalized traffic data.
        """
        if temporal_dim is not None:
            data = self.select_each_day(data, temporal_dim)

        norm_base = LoadData.normalize_base(data, norm_dim)  # find the normalize base
        norm_data = LoadData.normalize_data(norm_base[0], norm_base[1], data)  # normalize data

        return norm_base, norm_data

    @staticmethod
    def merge_data(data, temporal_dim, merge_num, compute_mode):
        """
        Merge traffic data with every merge_num records.

        :param data: np.array, [*, T, *]
        :param temporal_dim: int,
        :param merge_num: int,
        :param compute_mode: str, ["sum", "average"]
        :return:
            np.array, [*, T/merge_num , *]
        """
        T = data.shape[temporal_dim]

        result_data = None

        if temporal_dim == 0:
            list_result = [np.sum(data[i: i + merge_num], axis=0, keepdims=True) if compute_mode == "sum" else
                           np.mean(data[i: i + merge_num], axis=0, keepdims=True) for i in range(0, T, merge_num)]
            result_data = np.concatenate(list_result, axis=0)

        if temporal_dim == 1:
            list_result = [np.sum(data[:, i: i + merge_num], axis=1, keepdims=True) if compute_mode == "sum" else
                           np.mean(data[:, i: i + merge_num], axis=1, keepdims=True) for i in range(0, T, merge_num)]
            result_data = np.concatenate(list_result, axis=1)

        if temporal_dim == 2:
            list_result = [np.sum(data[:, :, i: i + merge_num], axis=2, keepdims=True) if compute_mode == "sum" else
                           np.mean(data[:, :, i: i + merge_num], axis=2, keepdims=True) for i in range(0, T, merge_num)]
            result_data = np.concatenate(list_result, axis=2)

        if temporal_dim not in [0, 1, 2]:
            raise ValueError("Temporal dimension is not correct.")

        return result_data

    @staticmethod
    def normalize_base(data, norm_dim):
        """
        :param data: np.array, original traffic data without normalization.
        :param norm_dim: int, normalization dimension.
        :return:
            max_data: np.array
            min_data: np.array
        """
        max_data = np.max(data, norm_dim, keepdims=True)  # [N, T, D] , norm_dim=1, [N, 1, D]
        min_data = np.min(data, norm_dim, keepdims=True)
        return max_data, min_data

    @staticmethod
    def normalize_data(max_data, min_data, data):
        """
        :param max_data: np.array, max data.
        :param min_data: np.array, min data.
        :param data: np.array, original traffic data without normalization.
        :return:
            np.array, normalized traffic data.
        """
        mid = min_data
        base = max_data - min_data
        normalized_data = (data - mid) / base

        return normalized_data

    @staticmethod
    def recover_data(max_data, min_data, data):
        """
        :param max_data: np.array, max data.
        :param min_data: np.array, min data.
        :param data: np.array, normalized data.
        :return:
            recovered_data: np.array, recovered data.
        """
        if data.ndim != max_data.ndim:
            max_data = max_data[np.newaxis]
            min_data = min_data[np.newaxis]

        mid = min_data
        base = max_data - min_data

        recovered_data = data * base + mid

        return recovered_data

    @staticmethod
    def to_tensor(data):
        return torch.tensor(data, dtype=torch.float)


if __name__ == '__main__':
    file_opt = FileStructure(data_folder="dataset/MySubway", flow_file="flow.npy", graph_file="graph.npy",
                             poi_file="poi.npy",
                             wea_file="wea.npy",
                             time_file="time.npy")

    data_opt = DataStructure(num_nodes=278,
                             divide_days=[15, 7, 7], one_day_range=[6, 22],
                             time_interval=10, merge_number=1,
                             src_length=6, trg_length=6)
    dataset = LoadData(file_opt, data_opt, "valid")

    dataloader = DataLoader(dataset, batch_size=128, shuffle=False, num_workers=32)

    for data in dataloader:
        print(data.keys())