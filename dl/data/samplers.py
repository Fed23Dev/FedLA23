import random

import numpy as np
from fedlab.utils.dataset.partition import CIFAR10Partitioner, CIFAR100Partitioner, FMNISTPartitioner
import math
import functools
from sklearn.utils import shuffle

from dl.data.datasets import get_data
from env.running_env import global_logger, global_file_repo
from env.support_config import VDataSet, VNonIID
from utils.TimeCost import timeit
from utils.objectIO import check_file_exists, pickle_load, pickle_mkdir_save

# todo
dir_alpha = 0.2


def fetch_shards(dataset_type: VDataSet, num_slices: int) -> int:
    if dataset_type == VDataSet.CIFAR10:
        return 2 * num_slices
    elif dataset_type == VDataSet.CIFAR100:
        return 2000
    elif dataset_type == VDataSet.FMNIST:
        return 2
    elif dataset_type == VDataSet.TinyImageNet:
        return 4000
    elif dataset_type == VDataSet.EMNIST:
        return 496
    else:
        global_logger.error("Not supported dataset type.")
        return 0


def iid(targets, num_clients: int) -> dict:
    client_dict = dict()
    total_data_amount = len(targets)
    each_amount = total_data_amount // num_clients
    data_indices = list(range(total_data_amount))
    random.shuffle(data_indices)
    data_indices = np.array(data_indices, dtype='int64')
    for client_index in range(num_clients):
        start = client_index * each_amount
        end = start + each_amount
        client_dict[client_index] = data_indices[start:end].copy()
    return client_dict


def get_partition_name(part_type: str, num_shards: int, alpha: float) -> str:
    if part_type == VNonIID.Hetero.value:
        assert alpha is not None, "If hetero, alpha must be not None."
        return f"hetero{alpha}"
    elif part_type == VNonIID.Shards.value:
        assert num_shards is not None, "If shards, num_shards must be not None."
        return f"shards{num_shards}"
    else:
        global_logger.error("Not supported partition type, using iid.")
        return f"iid"


def dataset_user_indices(dataset_type: VDataSet, num_slices, non_iid: str, seed: int = 2022) -> dict:
    dataset = get_data(dataset_type, data_type="train")
    if non_iid == 'iid':
        assert isinstance(dataset_type, VDataSet), "Not supported dataset type."
        client_dict = iid(dataset.targets, num_slices)
    else:
        num_shards = fetch_shards(dataset_type, num_slices)
        part_cache = global_file_repo.new_non_iid(dataset_type.value, num_slices,
                                                  get_partition_name(non_iid,
                                                                     num_shards,
                                                                     dir_alpha))[0]
        if check_file_exists(part_cache):
            global_logger.info("Using non-iid partition cache...")
            return pickle_load(part_cache)

        if dataset_type == VDataSet.CIFAR10:
            if non_iid == 'hetero':
                client_dict = CIFAR10Partitioner(dataset.targets, num_slices,
                                                 balance=None, partition="dirichlet",
                                                 dir_alpha=dir_alpha, seed=seed).client_dict
            else:
                client_dict = CIFAR10Partitioner(dataset.targets, num_slices,
                                                 balance=None, partition="shards",
                                                 num_shards=num_shards, seed=seed).client_dict
        elif dataset_type == VDataSet.CIFAR100:
            if non_iid == 'hetero':
                client_dict = CIFAR100Partitioner(dataset.targets, num_slices,
                                                  balance=None, partition="dirichlet",
                                                  dir_alpha=dir_alpha, seed=seed).client_dict
            else:
                client_dict = CIFAR100Partitioner(dataset.targets, num_slices,
                                                  balance=None, partition="shards",
                                                  num_shards=num_shards, seed=seed).client_dict
        elif dataset_type == VDataSet.FMNIST:
            if non_iid == 'hetero':
                client_dict = FMNISTPartitioner(dataset.targets, num_slices,
                                                partition="noniid-labeldir",
                                                dir_alpha=dir_alpha, seed=seed).client_dict
            else:
                client_dict = FMNISTPartitioner(dataset.targets, num_slices,
                                                partition="noniid-#label",
                                                major_classes_num=num_shards).client_dict
        elif dataset_type == VDataSet.TinyImageNet:
            if non_iid == 'hetero':
                client_dict = _hetero_non_iid(dataset.targets, 200,
                                              num_slices, dir_alpha, seed=seed)
            else:
                client_dict = _shards_non_iid(dataset.targets, num_shards,
                                              num_slices, seed, 200)
        elif dataset_type == VDataSet.EMNIST:
            if non_iid == 'hetero':
                client_dict = _hetero_non_iid(dataset.targets, 62,
                                              num_slices, dir_alpha, seed=seed)
            else:
                client_dict = _shards_non_iid(dataset.targets, num_shards,
                                              num_slices, seed, 62)
        else:
            global_logger.error("Not supported dataset type.")
            client_dict = None
            exit(1)
        global_logger.info("Saving non-iid partition cache...")
        pickle_mkdir_save(client_dict, part_cache)
        # pickle_mkdir_save(dataset.targets, f"{dataset_type.value}.targets")
    return client_dict


# dict {int: ndarray[dtype(int64)]}
def _shards_non_iid(targets: list, nums_shards: int, num_clients: int,
                    seed: int, num_classes: int) -> dict:
    assert nums_shards % num_classes == 0, "num_shards must be times of num_classes."
    assert nums_shards % num_clients == 0, "num_shards must be times of num_clients."
    # groups = _distribute_indices(targets, num_groups=nums_shards)
    # client_dict = _random_group_pairs(groups, n=nums_shards // num_clients, seed=seed)
    # return client_dict
    return _shards_non_iid_la(targets, nums_shards, num_clients, seed, num_classes)


# from Wang Huan
# [[1,2,3,411,534..],[]]
def _hetero_non_iid(targets: list, num_classes: int, num_clients: int, non_iid_alpha: float, seed=None):
    # all_class_index = []
    # for i in range(num_classes):
    #     curt_idx = []
    #     for idx, label in enumerate(targets):
    #         if label == i:
    #             curt_idx.append(idx)
    #     all_class_index.append(curt_idx)
    #
    # index2label = []  # 每一个样本和对应的label
    # for label, label_index in enumerate(all_class_index):
    #     for idx in label_index:
    #         index2label.append((idx, label))
    #
    # batch_indices = _build_non_iid_by_dirichlet(
    #     seed=seed,
    #     indices2targets=index2label,
    #     non_iid_alpha=non_iid_alpha,
    #     num_classes=num_classes,
    #     num_indices=len(index2label),
    #     n_workers=num_clients
    # )
    # # functools定义高阶函数或操作
    # index_dirichlet = functools.reduce(lambda x, y: x + y, batch_indices)
    # client2index = _partition_balance(index_dirichlet, num_clients)
    #
    # client_dict = dict()
    # for i in range(num_clients):
    #     client_dict[i] = client2index[i]
    # return client_dict
    return _hetero_non_iid_la(targets, num_classes, num_clients, non_iid_alpha, seed)


# from Wang Huan
def _partition_balance(idxs, num_split: int):
    # 所有index分成num_clients份,每份是num_per_part; 剩下没分完的是r
    num_per_part, r = len(idxs) // num_split, len(idxs) % num_split
    parts = []
    i, r_used = 0, 0
    # 把剩下没分完的r,给前r份每个都加1
    while i < len(idxs):
        if r_used < r:
            parts.append(idxs[i:(i + num_per_part + 1)])
            i += num_per_part + 1
            r_used += 1
        else:
            parts.append(idxs[i:(i + num_per_part)])
            i += num_per_part

    return parts


def _hetero_non_iid_la(targets: list, num_classes: int, num_clients: int, non_iid_alpha: float, seed=None):
    if seed is not None:
        np.random.seed(seed)

    targets = np.array(targets)
    client_data = {i: [] for i in range(num_clients)}

    # 对每个类别进行处理
    for c in range(num_classes):
        idx = np.where(targets == c)[0]
        idx = shuffle(idx, random_state=seed)

        # 使用Dirichlet分布来生成分布
        proportions = np.random.dirichlet(np.repeat(non_iid_alpha, num_clients))

        # 按照生成的比例将数据分配给每个客户端
        proportions = (np.cumsum(proportions) * len(idx)).astype(int)[:-1]
        client_idx_split = np.split(idx, proportions)

        for client_idx, data in enumerate(client_idx_split):
            client_data[client_idx] += data.tolist()

    return client_data


def _shards_non_iid_la(targets: list, nums_shards: int, num_clients: int, seed: int, num_classes: int) -> dict:
    """
    将目标数据按碎片（shards）方式进行非独立同分布（non-IID）划分。

    Args:
        targets (list): 数据的目标标签列表。
        nums_shards (int): 将数据集划分成的碎片数。
        num_clients (int): 客户端数量。
        seed (int): 随机种子。
        num_classes (int): 数据集分类数。

    Returns:
        client_data (dict): 每个客户端的数据索引字典，键为客户端ID，值为数据索引列表。
    """
    np.random.seed(seed)

    targets = np.array(targets)
    indices = np.arange(len(targets))

    # 每个类别分成的碎片数
    shards_per_class = nums_shards // num_classes
    # 每个客户端分到的碎片数
    shards_per_client = nums_shards // num_clients

    class_indices = [indices[targets == i] for i in range(num_classes)]

    shards = []
    for class_idx in class_indices:
        class_idx = shuffle(class_idx, random_state=seed)
        shard_size = len(class_idx) // shards_per_class
        shards += [class_idx[i * shard_size: (i + 1) * shard_size] for i in range(shards_per_class)]

    shards = shuffle(shards, random_state=seed)
    client_data = {i: [] for i in range(num_clients)}

    for i in range(num_clients):
        client_shards = shards[i * shards_per_client: (i + 1) * shards_per_client]
        client_data[i] = np.concatenate(client_shards).tolist()

    return client_data

# from Wang Huan
def _build_non_iid_by_dirichlet(seed, indices2targets, non_iid_alpha, num_classes, num_indices, n_workers):
    random_state = np.random.RandomState(seed)  # 设置伪随机数字序列
    n_auxi_workers = 10
    assert n_auxi_workers <= n_workers

    # 随机打乱重排数据[(data, label)]
    random_state.shuffle(indices2targets)
    # 分区索引
    from_index = 0
    splitted_targets = []

    num_splits = math.ceil(n_workers / n_auxi_workers)
    # 需要分成多少组,10个一次来分,如果不够那就第二次把剩下的全分完
    split_n_workers = [
        n_auxi_workers
        if idx < num_splits - 1
        else n_workers - n_auxi_workers * (num_splits - 1)
        for idx in range(num_splits)
    ]
    # 每次分的比率,占全部客户端的比例
    split_ratios = [_n_workers / n_workers for _n_workers in split_n_workers]

    # 按照比例来对全部数据样本进行non-IID划分
    for idx, ratio in enumerate(split_ratios):
        to_index = from_index + int(n_auxi_workers / n_workers * num_indices)
        splitted_targets.append(
            # 最后一组就是把剩下的全部数据放进去
            indices2targets[from_index: (num_indices if idx == num_splits - 1 else to_index)]
        )
        from_index = to_index

    # 此时数据被划分为一组一组的(splitted_targets)
    idx_batch = []
    for _targets in splitted_targets:
        _targets = np.array(_targets)
        _targets_size = len(_targets)

        # 考虑n_auxi_workers, n_workers其中最小的
        _n_workers = min(n_auxi_workers, n_workers)
        # 然后n_workers剩下的
        n_workers = n_workers - n_auxi_workers

        # 获取对应的idx_batch
        min_size = 0
        _idx_batch = None
        while min_size < int(0.50 * _targets_size / _n_workers):
            _idx_batch = [[] for _ in range(_n_workers)]
            for _class in range(num_classes):
                # 取出对应label的数据样本idx
                idx_class = np.where(_targets[:, 1] == _class)[0]  # 判断规则得到元组
                idx_class = _targets[idx_class, 0]  # 取出数据idx

                # 采样数据
                try:
                    proportions = random_state.dirichlet(
                        np.repeat(non_iid_alpha, _n_workers)
                    )
                    # balance
                    proportions = np.array(
                        [
                            p * (len(idx_j) < _targets_size / _n_workers)
                            for p, idx_j in zip(proportions, _idx_batch)
                        ]
                    )
                    proportions = proportions / proportions.sum()
                    proportions = (np.cumsum(proportions) * len(idx_class)).astype(int)[
                                  :-1
                                  ]
                    _idx_batch = [
                        idx_j + idx.tolist()
                        for idx_j, idx in zip(
                            _idx_batch, np.split(idx_class, proportions)
                        )
                    ]
                    sizes = [len(idx_j) for idx_j in _idx_batch]
                    min_size = min([_size for _size in sizes])
                except ZeroDivisionError:
                    pass
        if _idx_batch is not None:
            idx_batch += _idx_batch

    return idx_batch


def _distribute_indices(array, num_groups):
    unique_elements = np.unique(array)
    groups = [[] for _ in range(num_groups)]
    group_index = 0

    for element in unique_elements:
        indices = np.where(array == element)[0]
        half = len(indices) // 2
        groups[group_index].extend(indices[:half])
        group_index += 1
        groups[group_index].extend(indices[half:])
        group_index += 1
    return groups


def _random_group_pairs(groups, n=2, seed=2024):
    np.random.seed(seed)  # 确保随机组合是可重现的
    np.random.shuffle(groups)  # 随机打散所有组
    combined_groups = [sum(groups[i:i + n], []) for i in range(0, len(groups), n)]
    groups_dict = {i: combined_groups[i] for i in range(len(combined_groups))}
    return groups_dict
