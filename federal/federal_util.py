from typing import Union, Generator
import numpy as np
from collections import Counter

import torch

from dl.data.dataProvider import get_data_loaders, get_data_loader
from dl.data.datasets import get_data
from dl.data.samplers import dataset_user_indices
from env.running_env import args


def deepcopy_dict(ori_dict: Union[dict, Generator]):
    generator = ori_dict.items() if isinstance(ori_dict, dict) else ori_dict
    copy_dict = dict()
    for key, param in generator:
        copy_dict[key] = param.clone()
    return copy_dict


def disp_num_params(model):
    total_param_in_use = 0
    total_all_param = 0
    for layer, layer_prefix in zip(model.prunable_layers, model.prunable_layer_prefixes):
        layer_param_in_use = layer.num_weight
        layer_all_param = layer.mask.nelement()
        total_param_in_use += layer_param_in_use
        total_all_param += layer_all_param
        print("{} remaining: {}/{} = {}".format(layer_prefix, layer_param_in_use, layer_all_param,
                                                layer_param_in_use / layer_all_param))
    print("Total: {}/{} = {}".format(total_param_in_use, total_all_param, total_param_in_use / total_all_param))
    return total_param_in_use / total_all_param


# 计算总数据集的各分类分布
def dataset_dist() -> (np.ndarray, torch.Tensor):
    targets = np.array(get_data(args.dataset, "train").targets)
    total_num = len(targets)
    total_cnt = Counter(targets)
    global_dist = torch.tensor([total_cnt[cls] / total_num if cls in total_cnt else 0.00
                                for cls in range(args.num_classes)])
    return targets, global_dist


def max_class(client_cnt: Counter) -> int:
    base_class = 0
    max_cnt = -1
    for cls in range(args.num_classes):
        if client_cnt[cls] > max_cnt:
            base_class = cls
            max_cnt = client_cnt[cls]
    return base_class


def simulation_federal_process():
    user_dict = dataset_user_indices(args.dataset, args.workers, args.non_iid)
    workers_loaders = get_data_loaders(args.dataset, data_type="train", batch_size=args.batch_size,
                                       users_indices=user_dict, num_workers=0, pin_memory=False)
    test_loader = get_data_loader(args.dataset, data_type="test", batch_size=args.batch_size,
                                  shuffle=True, num_workers=0, pin_memory=False)
    return test_loader, workers_loaders, user_dict


def get_data_ratio(user_dict: dict):
    ratios_list = []
    sorted_cid = sorted(user_dict.keys())
    targets, global_dist = dataset_dist()

    for client_id in sorted_cid:
        indices = user_dict[client_id]
        client_targets = targets[indices]
        client_sample_num = len(indices)
        client_target_cnt = Counter(client_targets)

        ratio = torch.tensor([client_target_cnt[cls] / client_sample_num if cls in client_target_cnt else 0.00
                              for cls in range(args.num_classes)])
        ratios_list.append(ratio)
    return global_dist, ratios_list
