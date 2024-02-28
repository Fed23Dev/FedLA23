from copy import deepcopy
from typing import Iterator, List, Any

import torch

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
        :param global_params: Iterator[Parameter]
        :return:
        """
        global_logger.info(f'------Train from device: {self.id}------')
        self.cell.run_model(train=True, pre_params=global_params)


class FedLAWorker(FLWorker):
    def __init__(self, worker_id: int, worker_cell: SingleCell):
        super().__init__(worker_id, worker_cell)

    def local_train(self, info_matrix: torch.Tensor):
        global_logger.info(f'------Train from device: {self.id}------')
        self.cell.run_model(train=True, info_matrix=info_matrix)

    def local_distill(self, teacher_model: torch.nn.Module):
        self.cell.wrapper.dkd_loss_optim(teacher_model)


class ScaffoldWorker(FLWorker):
    def __init__(self, worker_id: int, worker_cell: SingleCell):
        super().__init__(worker_id, worker_cell)
        self.control, self.delta_control, self.delta_y = dict(), dict(), dict()
        for k, v in self.cell.access_model().named_parameters():
            self.control[k] = torch.zeros_like(v.data)
            self.delta_control[k] = torch.zeros_like(v.data)
            self.delta_y[k] = torch.zeros_like(v.data)
        self.train_before = deepcopy(self.cell.access_model())
        self.train_before_c = deepcopy(self.control)

        self.prev_batch = 0
        self.prev_epoch = 0

    def local_train(self, server_controls: dict):
        global_logger.info(f'------Train from device: {self.id}------')
        self.train_before = deepcopy(self.cell.access_model())
        self.train_before_c = deepcopy(self.control)
        self.cell.run_model(train=True, server_controls=server_controls, self_controls=self.control)

    def update_control(self, server_controls: dict):
        local_steps = self.cell.latest_feed_amount // args.batch_size
        lr = self.cell.wrapper.get_lr()
        temp = {}
        for k, v in self.cell.access_model().named_parameters():
            temp[k] = v.data.clone()

        for k, v in self.train_before.named_parameters():
            self.control[k] = self.control[k] - server_controls[k] + (v.data - temp[k]) / (local_steps * lr)
            self.delta_y[k] = temp[k] - v.data
            self.delta_control[k] = self.control[k] - self.train_before_c[k]


class MoonWorker(FLWorker):

    def __init__(self, worker_id: int, worker_cell: SingleCell):
        super().__init__(worker_id, worker_cell)
        self.prev_model = deepcopy(self.cell.access_model())

    def local_train(self, global_model: torch.nn.Module, mu: float, T: int):
        """
        :param T:
        :param mu:
        :param global_model: Iterator[Parameter]
        :return:
        """
        global_logger.info(f'------Train from device: {self.id}------')
        self.prev_model = deepcopy(self.cell.access_model())
        self.cell.run_model(train=True, global_model=global_model,
                            prev_model=self.prev_model,
                            mu=mu,
                            T=T)


class CriticalFLWorker(FLWorker):
    def __init__(self, worker_id: int, worker_cell: SingleCell):
        super().__init__(worker_id, worker_cell)

    def local_train(self, *_args, **kwargs):
        global_logger.info(f'------Train from device: {self.id}------')
        self.cell.run_model(train=True)

    def get_latest_grad(self) -> List[torch.Tensor]:
        return self.cell.latest_grad


class IFCAWorker(FLWorker):
    def __init__(self, worker_id: int, worker_cell: SingleCell):
        super().__init__(worker_id, worker_cell)

    def local_train(self, *_args, **kwargs):
        global_logger.info(f'------Train from device: {self.id}------')
        self.cell.run_model(train=True)

    def get_latest_grad(self) -> List[torch.Tensor]:
        return self.cell.latest_grad

    def get_group_loss(self, group_models: List[torch.nn.Module]) -> List[torch.Tensor]:
        return self.cell.wrapper.get_models_loss(group_models)
