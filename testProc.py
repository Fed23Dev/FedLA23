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
    batch_size = 32
    classes = 10
    logits = torch.randn(batch_size, classes)
    target = torch.randint(low=0, high=classes, size=(batch_size, 1))

    a = _get_gt_mask(logits, target)
    b = _get_other_mask(logits, target)
    print(logits)
    print(target)
    print(a*1000)
    print("----------------------")
