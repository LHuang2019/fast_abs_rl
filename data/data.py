""" CNN/DM dataset"""
import json
import re
import os
from os.path import join

from torch.utils.data import Dataset


class CnnDmDataset(Dataset):
    def __init__(self, data_list) -> None:
        self._n_data = len(data_list)
        self._data_list = data_list

    def __len__(self) -> int:
        return self._n_data

    def __getitem__(self, i: int):
        return self._data_list[i]
