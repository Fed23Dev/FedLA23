import copy
from typing import List, Tuple

import torch
from torch import optim
from torch.nn.functional import binary_cross_entropy_with_logits
import torch.utils.data as tdata

from federal.aggregation.FedAvg import FedAvg


class FedLA(FedAvg):
    ERROR_MESS2 = "clients_ids must be not null."

    def __init__(self, init_model: torch.nn.Module,
                 specification: Tuple[torch.Size], num_classes: int):
        super().__init__(init_model.state_dict())
        self.specification = specification
        self.classes = num_classes

