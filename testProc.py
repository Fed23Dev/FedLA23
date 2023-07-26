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
    loss_func = torch.nn.CrossEntropyLoss(reduction="none")
    pre = torch.tensor([[0.1, 0.5, 0.3, 0.4],
                        [0.1, 0.5, 0.3, 0.4],
                        [0.1, 0.5, 0.3, 0.4]], dtype=torch.float)
    tgt = torch.tensor([[1, 0, 0, 0],
                        [0, 1, 0, 0],
                        [0, 0, 1, 0]], dtype=torch.float)
    print("手动计算:")
    print("1.softmax")
    print(torch.softmax(pre, dim=-1))
    print("2.取对数")
    print(torch.log(torch.softmax(pre, dim=-1)))
    print("3.与真实值相乘")
    print(-torch.sum(torch.mul(torch.log(torch.softmax(pre, dim=-1)), tgt), dim=-1))
    print()
    print("调用损失函数:")
    loss = loss_func(pre, tgt)
    print(loss)
    print("----------------------")
