import torch.nn.functional as F
from torch import Tensor


# KL (y || x)
# Approved, Right
def _kl_divergence(x: Tensor, y: Tensor) -> Tensor:
    kl = F.kl_div(x.log(), y, reduction='sum')
    return kl


# KL (x || y)
def kl_divergence(x: Tensor, y: Tensor) -> Tensor:
    kl = F.kl_div(y.softmax(dim=-1).log(), x.softmax(dim=-1), reduction='sum')
    return kl


def js_divergence(x: Tensor, y: Tensor) -> Tensor:
    avg = ((x.softmax(dim=-1) + y.softmax(dim=-1)) / 2).log()
    return (kl_divergence(x, avg) + kl_divergence(y, avg)) / 2
