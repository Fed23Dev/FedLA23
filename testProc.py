import numpy as np
import torch


def tes():
    all = torch.tensor([[2, 2, 2], [3, 3, 3], [4, 4, 4]])
    zero = torch.tensor([[0, 0, 0], [1, 0.2, 1], [0, 0, 0]])

    mask = ~zero.bool()
    print(mask * all + zero)


if __name__ == "__main__":
    from dl.model.test_unit import test_shunet

    test_shunet()

    print("----------------------")
