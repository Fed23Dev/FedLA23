import random

import numpy as np
import torch
import torch.utils.data as tdata

from dl.SingleCell import SingleCell
from dl.wrapper.Wrapper import ProxWrapper, LAWrapper
from federal.aggregation.FedLA import FedLA

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
        master_cell = SingleCell(loader, Wrapper=ProxWrapper)
        super().__init__(workers, activists, local_epoch, master_cell)

        workers_cells = [SingleCell(loader) for loader in list(workers_loaders.values())]
        self.workers_nodes = [FedProxWorker(index, cell) for index, cell in enumerate(workers_cells)]

    def drive_workers(self):
        for index in self.curt_selected:
            self.workers_nodes[index].local_train(self.cell.access_model().parameters())


class FedLAMaster(FLMaster):
    def __init__(self, workers: int, activists: int, local_epoch: int,
                 loader: tdata.dataloader, workers_loaders: dict, data_dist: list,
                 num_classes: int, mb: int, me: int):

        master_cell = SingleCell(loader)
        super().__init__(workers, activists, local_epoch, master_cell)

        specification = master_cell.wrapper.running_scale()
        self.merge = FedLA(master_cell.access_model(), workers, specification, num_classes, me, mb)

        workers_cells = [SingleCell(loader, Wrapper=LAWrapper) for loader in list(workers_loaders.values())]
        self.workers_nodes = [FedLAWorker(index, cell) for index, cell in enumerate(workers_cells)]

        self.dataset_dist = data_dist
        self.curt_dist = torch.tensor([0.] * len(data_dist[0]))
        self.curt_dist[random.randrange(len(self.curt_dist))] = 1.

    def info_aggregation(self):
        workers_dict = []
        part_selected = self.curt_selected[:len(self.curt_selected)]
        for index in part_selected:
            workers_dict.append(self.workers_nodes[index].cell.access_model().state_dict())
        self.merge.merge_dict(workers_dict, part_selected)
        for index in self.curt_selected:
            self.workers_nodes[index].cell.decay_lr(self.pace)

    def schedule_strategy(self):
        # js_distance = []
        # for dist in self.dataset_dist:
        #     js_distance.append(js_divergence(self.curt_dist, dist))
        #
        # sort_rank = np.argsort(np.array(js_distance))
        # self.curt_selected = sort_rank[:self.plan]
        #
        # for ind in self.curt_selected:
        #     self.curt_dist += self.dataset_dist[ind]
        super(FedLAMaster, self).schedule_strategy()

    def drive_workers(self, *_args, **kwargs):
        stu_indices = self.curt_selected[:len(self.curt_selected)]
        tea_indices = self.curt_selected[len(self.curt_selected):]

        for index in self.curt_selected:
            self.workers_nodes[index].local_train()

        for s_index, t_index in zip(stu_indices, tea_indices):
            self.workers_nodes[s_index].local_distill(self.workers_nodes[t_index].cell.access_model())
