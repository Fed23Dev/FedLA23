import random
from copy import deepcopy

import numpy as np
import torch
import torch.utils.data as tdata
from sklearn.cluster import AgglomerativeClustering

from dl.SingleCell import SingleCell
from dl.wrapper.Wrapper import ProxWrapper, LAWrapper
from env.running_env import global_container
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
                 loader: tdata.dataloader, workers_loaders: dict,
                 num_classes: int, clusters: int, drag: int,
                 threshold: float):

        master_cell = SingleCell(loader, Wrapper=LAWrapper)
        super().__init__(workers, activists, local_epoch, master_cell)

        specification = master_cell.wrapper.running_scale()
        self.merge = FedLA(master_cell.access_model(), specification, num_classes)

        workers_cells = [SingleCell(loader, Wrapper=LAWrapper) for loader in list(workers_loaders.values())]
        self.workers_nodes = [FedLAWorker(index, cell) for index, cell in enumerate(workers_cells)]

        self.prev_workers_matrix = [torch.zeros(num_classes, num_classes) for _ in range(workers)]
        self.workers_matrix = [torch.zeros(num_classes, num_classes) for _ in range(workers)]
        self.prev_matrix = torch.zeros(num_classes, num_classes)
        self.curt_matrix = torch.zeros(num_classes, num_classes)

        self.num_clusters = self.workers//2       # [2, M/2]
        self.threshold = threshold
        self.drag = drag
        self.pipeline_status = 0
        self.clusters_indices = []

    def single_select(self):
        js_dists = []
        for info_matrix in self.workers_matrix:
            js_dists.append(js_divergence(self.curt_matrix, info_matrix))
        self.curt_selected = [min(js_dists)]

    def adaptive_clusters(self):
        if self.num_clusters == 0:
            return self.num_clusters

        IM_diff = torch.abs(self.curt_matrix - self.prev_matrix)
        IM_ratio = IM_diff / self.prev_matrix
        average_ratio = torch.mean(IM_ratio)
        if average_ratio > self.threshold:
            self.num_clusters = self.num_clusters//2 if self.num_clusters//2 > 2 else 2
        else:
            self.num_clusters = 0
        return self.num_clusters

    def sync_matrix(self):
        self.prev_workers_matrix = deepcopy(self.workers_matrix)
        self.prev_matrix = deepcopy(self.curt_matrix)
        self.workers_matrix.clear()
        self.curt_matrix.zero_()

        for i in range(self.workers):
            self.workers_matrix.append(self.workers_nodes[i].cell.wrapper.get_logits_matrix())

        self.curt_matrix = torch.div(sum(self.workers_matrix), len(self.workers_matrix))
        global_container.flash('avg_matrix', deepcopy(self.curt_matrix).numpy())

    def info_aggregation(self):
        workers_dict = []
        drag_cnt = int(self.drag*len(self.curt_selected))

        for index in random.sample(self.curt_selected, len(self.curt_selected)-drag_cnt):
            workers_dict.append(self.workers_nodes[index].cell.access_model().state_dict())

        for _ in range(drag_cnt):
            workers_dict.append(self.merge.pre_dict)

        self.merge.merge_dict(workers_dict)

        for index in self.curt_selected:
            self.workers_nodes[index].cell.decay_lr(self.pace)

    def schedule_strategy(self):
        self.curt_selected.clear()
        if self.curt_round == 0:
            super(FedLAMaster, self).schedule_strategy()
            return

        clusters = self.adaptive_clusters()

        if clusters == 0:
            self.single_select()
            return

        if self.pipeline_status == 0:
            X = torch.stack(self.workers_matrix, dim=0).numpy()
            n_samples, dim1, dim2 = X.shape
            flattened_X = X.reshape(n_samples, dim1 * dim2)
            clustering = AgglomerativeClustering(n_clusters=clusters).fit(flattened_X)
            self.clusters_indices = clustering.labels_

        # doing: to modify
        self.curt_selected = np.where(self.clusters_indices == self.pipeline_status)[0].tolist()

        self.pipeline_status += 1

        if self.pipeline_status == clusters:
            self.pipeline_status = 0

        if len(self.curt_selected) == 1:
            return

        if len(self.curt_selected) > self.plan:
            self.curt_selected = random.sample(self.curt_selected, self.plan)

        # node_cnt = len(self.curt_selected) if len(self.curt_selected) < self.plan else self.plan
        # all_indices = set(list(range(len(self.workers_nodes))))
        # la_indices = set(self.curt_selected)
        # difference = list(all_indices - la_indices)
        #
        # half1 = random.sample(self.curt_selected, node_cnt // 2 + 1)
        # half2 = random.sample(difference, node_cnt // 2 + 1)
        # half1.extend(half2)
        # self.curt_selected = deepcopy(half1)

        # # debug switch: selection
        # self.sync_matrix()

        # # debug switch: selection
        # super(FedLAMaster, self).schedule_strategy()

    def drive_workers(self, *_args, **kwargs):
        global_container.flash('selected_workers', deepcopy(self.curt_selected))
        for index in self.curt_selected:
            self.workers_nodes[index].local_train(self.curt_matrix)

        self.sync_matrix()
