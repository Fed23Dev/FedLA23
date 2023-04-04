from torch import optim
from torch.optim import Optimizer


class WarmUPStepLR(optim.lr_scheduler.LambdaLR):
    def __init__(self, optimizer: Optimizer, warm_up_steps: int = 5,
                 step_size: int = 1, gamma: float = 0.5 ** (1 / 100)):
        self.warm_up_steps = warm_up_steps
        self.step_size = step_size
        self.gamma = gamma
        super().__init__(optimizer, self.lr_lambda)

    def lr_lambda(self, curt_step: int):
        if curt_step < self.warm_up_steps:
            return float(curt_step) / float(max(1, self.warm_up_steps))
        else:
            return self.gamma**((curt_step - self.warm_up_steps) // self.step_size)
