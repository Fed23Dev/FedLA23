import torch


class FedProx:
    def __init__(self, mu: float):
        self.mu = mu

    def loss_prox(self, curt_weight: dict, union_weight: dict) -> torch.Tensor:
        l2_norm = torch.tensor(0.0)
        for _, v1, _, v2 in zip(curt_weight.items(), union_weight.items()):
            l2_norm += torch.norm(v1 - v2) ** 2
        prox_info = torch.Tensor(self.mu / 2 * l2_norm)
        return prox_info
