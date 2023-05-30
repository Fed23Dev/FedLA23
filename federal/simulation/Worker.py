from typing import Iterator, List, Any

import torch
from torch.nn import Parameter
import torch.utils.data as tdata
from yacs.config import CfgNode

from dl.SingleCell import SingleCell
from dl.compress.DKD import DKD
from env.running_env import global_logger, args
from federal.simulation.FLnodes import FLWorker


# CIFAR VGG
class FedAvgWorker(FLWorker):
    def __init__(self, worker_id: int, worker_cell: SingleCell):
        super().__init__(worker_id, worker_cell)

    def local_train(self):
        global_logger.info(f'------Train from device: {self.id}------')
        self.cell.run_model(train=True)


class FedProxWorker(FLWorker):
    def __init__(self, worker_id: int, worker_cell: SingleCell):
        super().__init__(worker_id, worker_cell)

    def local_train(self, global_params: Iterator):
        """
        :param global_params: Iterator[Parameter]
        :return:
        """
        global_logger.info(f'------Train from device: {self.id}------')
        self.cell.run_model(train=True, pre_params=global_params)


class FedLAWorker(FLWorker):
    def __init__(self, worker_id: int, worker_cell: SingleCell):
        super().__init__(worker_id, worker_cell)

    def local_train(self):
        global_logger.info(f'------Train from device: {self.id}------')
        self.cell.run_model(train=True)

    def local_distill(self, teacher_model: torch.nn.Module):
        self.cell.wrapper.dkd_loss_optim(teacher_model)
