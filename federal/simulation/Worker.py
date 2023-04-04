from typing import Iterator, List, Any

import torch
from torch.nn import Parameter
import torch.utils.data as tdata

from dl.SingleCell import SingleCell
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
        :param blank:
        :param global_params: Iterator[Parameter]
        :return:
        """
        global_logger.info(f'------Train from device: {self.id}------')
        self.cell.run_model(train=True, pre_params=global_params)


class FedIRWorker(FLWorker):
    def __init__(self, worker_id: int, worker_cell: SingleCell, dist_ratio: torch.Tensor):
        super().__init__(worker_id, worker_cell)
        self.loss_weight = dist_ratio

    def local_train(self):
        global_logger.info(f'------Train from device: {self.id}------')
        self.cell.run_model(train=True, loss_weight=self.loss_weight)


class HRankFLWorker(FLWorker):
    def __init__(self, worker_id: int, worker_cell: SingleCell):
        super().__init__(worker_id, worker_cell)

    def local_train(self):
        global_logger.info(f'------Train from device: {self.id}------')
        self.cell.run_model(train=True)


class CALIMFLWorker(FLWorker):
    ERROR_MESS1 = "When prune, must provide grads_container."

    def __init__(self, worker_id: int, worker_cell: SingleCell,
                 loader: tdata.DataLoader):
        super().__init__(worker_id, worker_cell)
        self.en_loader = loader

    def local_train(self, grads_container: list):
        global_logger.info(f'------Train from device: {self.id}------')
        self.cell.run_model(train=True)
        assert grads_container is not None, self.ERROR_MESS1
        grads_container.append(self.cell.get_latest_grads())

    def enhance_local_train(self):
        if self.en_loader is None:
            return
        global_logger.info(f'------Enhance Train from device: {self.id}------')
        for i in range(args.local_epoch):
            self.cell.wrapper.step_run(args.batch_limit,
                                       train=True, loader=self.en_loader)
