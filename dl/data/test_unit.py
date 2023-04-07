import pickle
from fedlab.utils.dataset import CIFAR10Partitioner
from fedlab.utils.functional import partition_report

from dl.data.samplers import dataset_user_indices
from dl.data.dataProvider import get_data_loader, get_data_loaders, DataLoader
from env.running_env import args
from env.support_config import VDataSet


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


# def dataset():
#     user_dict = dataset_user_indices(args.dataset, args.workers)
#     workers_loaders = get_data_loaders(args.dataset, data_type="train", batch_size=args.batch_size,
#                                        users_indices=user_dict, num_workers=0, pin_memory=False)
#     test_loader = get_data_loader(args.dataset, data_type="test", batch_size=args.batch_size,
#                                   shuffle=True, num_workers=0, pin_memory=False)
#     loaders = list(workers_loaders.values())
#     print("here")


def test_ucm():
    cifar10 = get_data(VDataSet.CIFAR10, data_type="test")
    ucm = get_data(VDataSet.UCM, data_type="test")
    a = cifar10[0]
    b = ucm[0]

    len1 = len(cifar10)
    len2 = len(ucm)

    test_loader = get_data_loader(VDataSet.UCM, data_type="test", batch_size=16,
                                  shuffle=True, num_workers=0, pin_memory=False)
    # for batch in test_loader:
    #     x = batch["image"]
    #     y = batch["label"]
    #     print(x)
    #     print(y)
    #     break

    for x, y in test_loader:
        print(x.size())
        print(y.size())
        break


def test_cifar10_gan():
    cifar10_gan = get_fake_data(VDataSet.CIFAR10)
    a = cifar10_gan[0]
    len1 = len(cifar10_gan)

    cifar10 = get_data(VDataSet.CIFAR10, "train")
    b = cifar10[0]

    print(a[0].size(), a[1].size())
    print(b[0].size(), b[1].size())
    exit(0)
    len2 = len(cifar10)

    test_loader = DataLoader(cifar10_gan, batch_size=32, shuffle=True,
                             num_workers=8,
                             pin_memory=False, drop_last=True)

    # for batch in test_loader:
    #     x = batch["image"]
    #     y = batch["label"]
    #     print(x)
    #     print(y)
    #     break

    for x, y in test_loader:
        print(x.size())
        print(y.size())
        break

    test_loader1 = get_data_loader(VDataSet.CIFAR10, "train", batch_size=32)
    for x, y in test_loader1:
        print(x.size())
        print(y.size())
        break

def client_dict2csv():
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


# 各个阶段返回的数据类型
