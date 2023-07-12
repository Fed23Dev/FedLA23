import torch
import random


def _get_gt_mask(logits, target):
    target = target.reshape(-1)
    mask = torch.zeros_like(logits).scatter_(1, target.unsqueeze(1), 1).bool()
    return mask


def _get_other_mask(logits, target):
    target = target.reshape(-1)
    mask = torch.ones_like(logits).scatter_(1, target.unsqueeze(1), 0).bool()
    return mask


if __name__ == "__main__":
    # from dl.test_unit import main
    # main()
    a = torch.ones(10)
    b = torch.div(a, 10)
    print(a)
    print(b)
    print("----------------------")
