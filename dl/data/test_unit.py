import pickle

import pandas as pd
import torch
from fedlab.utils.dataset import CIFAR10Partitioner
from fedlab.utils.functional import partition_report
import matplotlib.pyplot as plt

from dl.data.datasets import get_data
from dl.data.samplers import dataset_user_indices
from dl.data.dataProvider import get_data_loader, get_data_loaders
from env.running_env import args
from env.support_config import VDataSet
from dl.data.datasets import download_datasets


def data_size():
    loader = get_data_loader(VDataSet.UCM, batch_size=16, data_type="test")
    for x, y in loader:
        print(f"data:{x.size()}")
        print(f"label:{y.size()}")
        break


def test_non_iid():
    user_dict = dataset_user_indices(VDataSet.UCM, 100, non_iid="shards")
    a_dict = dataset_user_indices(VDataSet.CIFAR10, 100, non_iid="shards")
    print("here")


def test_loaders():
    user_dict = dataset_user_indices(args.dataset, args.workers)

    workers_loaders = get_data_loaders(args.dataset, data_type="train", batch_size=args.batch_size,
                                       users_indices=user_dict, num_workers=0, pin_memory=False)
    test_loader = get_data_loader(args.dataset, data_type="test", batch_size=args.batch_size,
                                  shuffle=True, num_workers=0, pin_memory=False)
    loaders = list(workers_loaders.values())
    print("here")


def null2client_dict2csv():
    dataset = get_data(VDataSet.CIFAR10, data_type="train")
    client_dict = CIFAR10Partitioner(dataset.targets, 100,
                                     balance=None, partition="dirichlet",
                                     dir_alpha=0.3, seed=2023).client_dict
    csv_file = "hetero.csv"
    partition_report(dataset.targets, client_dict,
                     class_num=10, verbose=False, file=csv_file)
    client_dict2 = CIFAR10Partitioner(dataset.targets, 100,
                                      balance=None, partition="shards",
                                      num_shards=200, seed=2023).client_dict

    csv_file = "shards.csv"
    partition_report(dataset.targets, client_dict2,
                     class_num=10, verbose=False, file=csv_file)


def client_dict2png(trainset, client_dict, num_classes: int):
    csv_file = f"res/non-iid.csv"
    out_file = f"res/non-iid.png"

    partition_report(trainset.targets, client_dict,
                     class_num=num_classes,
                     verbose=False, file=csv_file)

    hetero_dir_part_df = pd.read_csv(csv_file, header=1)
    hetero_dir_part_df = hetero_dir_part_df.set_index('client')
    col_names = [f"class{i}" for i in range(num_classes)]
    for col in col_names:
        hetero_dir_part_df[col] = (hetero_dir_part_df[col] * hetero_dir_part_df['Amount']).astype(int)

    hetero_dir_part_df[col_names].iloc[:10].plot.barh(stacked=True)
    plt.tight_layout()
    plt.xlabel('sample num')
    plt.savefig(out_file, dpi=400)


def test_tiny_imagenet():
    tinyimagenet = get_data(VDataSet.TinyImageNet, data_type="train")
    data0, label0 = tinyimagenet[0]
    print(data0.size())
    print(label0.size())


def test_loader_label():
    test_loader = get_data_loader(args.dataset, data_type="test", batch_size=args.batch_size,
                                  shuffle=True, num_workers=0, pin_memory=False)
    for batch_idx, (inputs, targets) in enumerate(test_loader):
        if batch_idx > 0:
            break
        label = torch.argmax(targets, -1)
        print(inputs.size())
        print(targets.size())
        print(targets)
        print(label)

        labels = torch.argmax(targets, -1)
        _labels, _cnt = torch.unique(labels, return_counts=True)
        labels_cnt = torch.zeros(args.num_classes, dtype=torch.int64).scatter_(dim=0, index=_labels, src=_cnt)
        print(_labels)
        print(_cnt)
        print(labels_cnt)


def main():
    test_loader_label()


if __name__ == '__main__':
    download_datasets()
