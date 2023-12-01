import numpy as np
import torch.nn.functional as F
from torch import Tensor
from sklearn.manifold import TSNE


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


# t-SNE 2 dimension
def t_sne(data: np.ndarray) -> np.ndarray:
    tsne = TSNE(n_components=2)
    transformed_data = tsne.fit_transform(data)
    return transformed_data
