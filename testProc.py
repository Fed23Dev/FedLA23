import torch
import random


def tensor(some: int):
    data_dist = [torch.tensor([1, 1]), torch.tensor([1, 1])]
    curt_dist = torch.tensor([0.] * len(data_dist[0]))
    curt_dist[random.randrange(len(curt_dist))] = 1.
    print(curt_dist)


def inner(a, **kwargs):
    print(a)
    print(kwargs['b'])


def outter(**kwargs):
    inner(**kwargs)


if __name__ == "__main__":
    # from dl.data.test_unit import main
    # main()
    # from federal.test_unit import test_master
    # test_master()
    # from utils.test_unit import kl_and_js
    # kl_and_js()
    # tensor()
    outter(a=1, b=2)
    print("----------------------")
