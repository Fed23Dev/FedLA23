import re
import random
import pandas as pd
import numpy as np
from typing import List

from utils.VContainer import VContainer
from env.running_env import *
species = ['VGG', 'ResNet']


def random_list(length=100):
    random_int_list = []
    for i in range(length):
        random_int_list.append(random.randint(0, 10))
    return random_int_list

def random_label(length=100):
    random_int_list = []
    for i in range(length):
        random_int_list.append(random.choice(species))
    return random_int_list

def get_ori_lists():
    return [range(100), random_list(), random_list(), random_label()]

def get_ori_dict():
    return {'index': np.array(range(100)),
            'FLOPs': np.array(random_list()),
            'Acc': np.array(random_list()),
            'class': np.array(random_label())}

def get_lists():
    return list(zip(range(100), random_list(), random_list(), random_label()))


class Extractor:
    ERROR_MESS1 = "The length cannot exceed 4 characters."
    ERROR_MESS2 = "The mode str contains not defined character."

    CLASS_COL_NAME = "class"
    CURRENT_CLASS_NAME = "VGG16"
    # log_path = 'logs/hrankFL.log'
    log_path = 'sim.out'
    KEYS = ['FLOPs', 'Acc', 'Interval', 'Rate']
    REG_PATTERNS = [r'(?<=#FLOPs:)[^\#]+(?=#)',
                    r'(?<=#Acc:)[^\#]+(?=#)',
                    r'(?<=#Interval:)[^\#]+(?=#)',
                    r'(?<=#Rate:)[^\#]+(?=#)']
    CSV_FILE_NAME = "vital"

    def __init__(self):
        self.info_vars = VContainer()
        self.rows = 0
        self.data_frame = None

    def parse(self, indices: list):
        with open(self.log_path, "rb") as f:
            for line in f:
                for idx in indices:
                    matches = re.finditer(self.REG_PATTERNS[idx], str(line), re.M)
                    for ma in matches:
                        if ma.group().find('%') != -1:
                            self.info_vars.flash(self.KEYS[idx], float(ma.group()[:-1]))
                        else:
                            self.info_vars.flash(self.KEYS[idx], float(ma.group()))
                        # self.info_vars.flash(self.CLASS_COL_NAME, self.CURRENT_CLASS_NAME)
                        self.rows += 1

    # +-------+-------+-----+----------+------+-------+
    # | index | FLOPs | Acc | Interval | Rate | class |
    # +-------+-------+-----+----------+------+-------+
    # | 0     | 1.1M  | 98% | Round:100| 0.9  | VGG16 |
    # | 1     | 1.1M  | 50% | Round:50 | 0.8  | VGG16 |
    # +-------+-------+-----+----------+------+-------+
    def toDataFrame(self) -> pd.DataFrame:
        return self.data_frame

    # def map_vars(self, key_indices: List[int]) -> pd.DataFrame:
    #     self.parse(key_indices)
    #     self.data_frame.df.to_csv(file_repo.visual(name=self.CSV_FILE_NAME))
    #     self.data_frame = pd.DataFrame(self.info_vars.container)
    #     return self.data_frame
    def map_vars(self, key_indices: List[int]) -> pd.DataFrame:
        self.parse(key_indices)
        self.data_frame = pd.DataFrame(self.info_vars.container)
        return self.data_frame

    def clear_container(self):
        self.info_vars.container.clear()
        self.rows = 0
