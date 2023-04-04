import math
from torch import optim
from torch.optim import Optimizer


class WarmUPCosineLR(optim.lr_scheduler.LambdaLR):
    def __init__(self, optimizer: Optimizer, warm_up_steps: int = 5, T_max: int = 50, lr_min: float = 1e-5):
        self.optimizer = optimizer
        self.warm_up_steps = warm_up_steps
        self.T_max = T_max
        self.lr_max = optimizer.state_dict()['param_groups'][0]['lr']
        self.lr_min = lr_min
        self.const_coff = 0.5 * (self.lr_max - self.lr_min)

        super().__init__(optimizer, self.lr_lambda)

    def lr_lambda(self, curt_step: int):
        if curt_step < self.warm_up_steps:
            return float(curt_step) / float(max(1, self.warm_up_steps))
        else:
            return (self.lr_min +
                    self.const_coff * (1.0 + math.cos((curt_step - self.warm_up_steps) /
                                                      (self.T_max - self.warm_up_steps) * math.pi))) / self.lr_max
