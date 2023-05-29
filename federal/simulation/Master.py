import random

import numpy as np
import torch
import torch.utils.data as tdata

from dl.SingleCell import SingleCell

from federal.simulation.FLnodes import FLMaster
from federal.simulation.Worker import FedAvgWorker, FedProxWorker, FedLAWorker
from utils.MathTools import js_divergence


class FedAvgMaster(FLMaster):

    def __init__(self, workers: int, activists: int, local_epoch: int,
                 loader: tdata.dataloader, workers_loaders: dict):
        """

        :param workers:
        :param activists:
        :param local_epoch:
        :param loader: *only in simulation*
        :param workers_loaders: *only in simulation*
        """
        master_cell = SingleCell(loader)
        super().__init__(workers, activists, local_epoch, master_cell)

        workers_cells = [SingleCell(loader) for loader in list(workers_loaders.values())]
        self.workers_nodes = [FedAvgWorker(index, cell) for index, cell in enumerate(workers_cells)]

    def drive_workers(self):
        for index in self.curt_selected:
            self.workers_nodes[index].local_train()


class FedProxMaster(FLMaster):
    def __init__(self, workers: int, activists: int, local_epoch: int,
                 loader: tdata.dataloader, workers_loaders: dict):
        """

        :param workers:
        :param activists:
        :param local_epoch:
        :param loader: *only in simulation*
        :param workers_loaders: *only in simulation*
        """
        master_cell = SingleCell(loader)
        super().__init__(workers, activists, local_epoch, master_cell)

        workers_cells = [SingleCell(loader) for loader in list(workers_loaders.values())]
        self.workers_nodes = [FedProxWorker(index, cell) for index, cell in enumerate(workers_cells)]

    def drive_workers(self):
        for index in self.curt_selected:
            self.workers_nodes[index].local_train(self.cell.access_model().parameters())


class FedLAMaster(FLMaster):
    def __init__(self, workers: int, activists: int, local_epoch: int,
                 loader: tdata.dataloader, workers_loaders: dict, data_dist: list):

        master_cell = SingleCell(loader)
        super().__init__(workers, activists, local_epoch, master_cell)

        workers_cells = [SingleCell(loader) for loader in list(workers_loaders.values())]

        self.workers_nodes = [FedLAWorker(index, cell) for index, cell in enumerate(workers_cells)]

        self.dataset_dist = data_dist
        self.curt_dist = torch.tensor([0.] * len(data_dist[0]))
        self.curt_dist[random.randrange(len(self.curt_dist))] = 1.

    def schedule_strategy(self):
        js_distance = []
        for dist in self.dataset_dist:
            js_distance.append(js_divergence(self.curt_dist, dist))

        sort_rank = np.argsort(np.array(js_distance))
        self.curt_selected = sort_rank[:self.plan]

        for ind in self.curt_selected:
            self.curt_dist += self.dataset_dist[ind]

    def drive_workers(self, *_args, **kwargs):
        for index in self.curt_selected:
            self.workers_nodes[index].local_train(self.cell.access_model().parameters())
        self.asymmetric_distillation()

    def asymmetric_distillation(self):
        pass
