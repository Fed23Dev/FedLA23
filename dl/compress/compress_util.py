import re
import sys

import numpy as np
import torch
from copy import deepcopy

zero_value = 0.001
data_key = 'VRank_COO_INDEX'
size_key = 'VRank_COO_SIZE'
threshold_zero = 1e-5


# Calculate memory size of the give tensor - units:B
def get_size(tensor: torch.Tensor) -> int:
    nums_element = tensor.nelement()
    regex = r"(\d+)$"
    test_str = str(tensor.dtype)
    match = next(re.finditer(regex, test_str, re.MULTILINE))
    bits = int(match.group())
    return bits * nums_element // 8


# 仅支持纬度数据
# only support single dimension
def max_min_normalization(x: np.ndarray) -> np.ndarray:
    return (x - np.min(x)) / (np.max(x) - np.min(x) + zero_value)


# only support single dimension
def mean_var_normalization(x: np.ndarray) -> np.ndarray:
    return (x - np.mean(x)) / (np.std(x) + zero_value)


def arrays_normalization(arrays: list, scales: list, way=mean_var_normalization) -> list:
    assert len(scales) == len(arrays), 'The length of arrays must be equal to the length of scale.'
    norm_list = []
    for x, coff in zip(arrays, scales):
        norm_list.append(coff*way(x))
    return norm_list


def calculate_average_value(ndarray_list: list) -> list:
    ret_list = []
    for array in ndarray_list:
        ret_list.append(np.mean(array))
    return ret_list


def list_sub(max_l, min_l) -> torch.Tensor:
    c = deepcopy(max_l)
    b = deepcopy(min_l)
    while len(b) > 0:
        if b[0] in c:
            c.remove(b[0])
            b.remove(b[0])
        else:
            print("Error: b is not a sub set of a")
            exit(1)
    return torch.tensor(c)


def mem_usage(obj):
    print(sys.getsizeof(obj) / 1024 / 1024, 'MB')


def dict_coo_express(model_dict: dict) -> dict:
    global data_key
    global size_key
    data_indices = []
    ori_sizes = []
    ret_dict = deepcopy(model_dict)
    # k: str, v: torch.Tensor
    for k, v in model_dict.items():
        if k.find('weight') != -1 and k.find('conv') != -1:
            zero_index = []
            first_dim = v.size()[0]
            ori_sizes.append(first_dim)
            all_index = list(range(first_dim))
            for filter_idx in range(first_dim):
                filter_weight = v[filter_idx:filter_idx+1]
                if filter_weight.sum() == 0:
                    zero_index.append(filter_idx)
            data_index = list_sub(all_index, zero_index)
            assert data_index != torch.Size([0]), "Can't prune all filters out."
            ret_dict[k] = torch.index_select(v, dim=0, index=data_index)
            data_indices.append(data_index)
    ret_dict[data_key] = data_indices
    ret_dict[size_key] = ori_sizes
    return ret_dict


# only support dim=0
def fill_tensor_zero(t: torch.Tensor, index: list, dim0_size: int) -> torch.Tensor:
    new_size = list(t.size())
    new_size[0] = dim0_size
    zero_tensor = torch.zeros(new_size)
    curt_fill = 0
    for ind in index:
        zero_tensor[ind] = t[curt_fill]
        curt_fill += 1
    return zero_tensor


def dict_coo_recover(model_dict: dict) -> dict:
    global data_key
    global size_key
    assert data_key in model_dict.keys(), "Not supported model_dict."
    assert size_key in model_dict.keys(), "Not supported model_dict."
    data_indices = model_dict[data_key]
    sizes = model_dict[size_key]
    curt_index = 0
    ret_dict = deepcopy(model_dict)
    for k, v in model_dict.items():
        if k.find('weight') != -1 and k.find('conv') != -1:
            data_index = data_indices[curt_index]
            size = sizes[curt_index]
            ret_dict[k] = fill_tensor_zero(v, data_index, size)
            curt_index += 1
    ret_dict.pop(data_key)
    ret_dict.pop(size_key)
    return ret_dict
