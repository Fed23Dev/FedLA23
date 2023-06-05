import copy
from typing import List, Tuple

import torch
from torch import optim
from torch.nn.functional import binary_cross_entropy_with_logits

from federal.aggregation.FedAvg import FedAvg


class FedLA(FedAvg):
    ERROR_MESS2 = "clients_ids must be not null."

    def __init__(self, init_model: torch.nn.Module, workers: int,
                 specification: Tuple[torch.Size], num_classes: int,
                 epoch: int = 10, batch_limit: int = 5):
        super().__init__(init_model.state_dict())
        self.merge_weight = torch.ones(workers)
        self.specification = specification
        self.classes = num_classes

        self.epoch = epoch
        self.batch_limit = batch_limit

        self.model = init_model
        self.optim = optim.SGD([self.merge_weight], lr=0.01, momentum=0.9, weight_decay=1e-4)
        self.loss_func = binary_cross_entropy_with_logits

    def merge_dict(self, clients_dicts: List[dict], clients_ids: List[int] = None) -> dict:
        assert clients_ids is not None, self.ERROR_MESS2
        if self.epoch == 0 and self.batch_limit == 0:
            super(FedLA, self).merge_dict(clients_dicts)
            pass
        curt_weight = self.merge_weight.index_select(0, torch.tensor(clients_ids))
        for i in range(self.epoch):
            self.union_dict.clear()
            batch_cnt = 0
            while batch_cnt < self.batch_limit:
                # right现为一个标量，后可以改为张量
                for right, dic in zip(curt_weight, clients_dicts):
                    for k, v in dic.items():
                        if k in self.union_dict.keys():
                            self.union_dict[k] += v * right
                        else:
                            self.union_dict[k] = v * right

                device = next(self.model.parameters()).device
                inputs = torch.randn(self.specification[0]).to(device)
                logits = (torch.ones(self.specification[1]) / torch.tensor(self.classes)).to(device)

                pred = self.model(inputs)
                loss = self.loss_func(pred, logits)
                self.optim.zero_grad()
                loss.backward()
                self.optim.step()

                self.model.load_state_dict(self.union_dict)
                batch_cnt += 1

        clients_dicts.clear()
        return copy.deepcopy(self.model.state_dict())
