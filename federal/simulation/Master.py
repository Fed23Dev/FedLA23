import random
from copy import deepcopy

import numpy as np
import torch
import torch.utils.data as tdata
from scipy.spatial.distance import jensenshannon
from sklearn.cluster import AgglomerativeClustering

from dl.SingleCell import SingleCell
from dl.wrapper.Wrapper import ProxWrapper, LAWrapper, ScaffoldWrapper, MoonWrapper
from env.running_env import global_container, global_logger
from federal.aggregation.FedLA import FedLA

from federal.simulation.FLnodes import FLMaster
from federal.simulation.Worker import FedAvgWorker, FedProxWorker, FedLAWorker, ScaffoldWorker, MoonWorker
from utils.MathTools import js_divergence, remove_top_k_elements


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

        workers_cells = [SingleCell(loader, Wrapper=ProxWrapper)
                         for loader in list(workers_loaders.values())]
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

        self.num_clusters = self.workers // 2  # [2, M/2]
        self.drag = drag

        self.fix = clusters
        self.clusters = 0
        self.pipeline = [[]]
        self.pipeline_status = 0
        self.max_round = 0
        self.clusters_indices = []

        self.start_matrix = None
        self.cft = 0.3
        self.threshold = threshold

    def single_select(self):
        js_dists = []
        for info_matrix in self.workers_matrix:
            js_dists.append(js_divergence(self.curt_matrix, info_matrix).numpy())
        global_container.flash("js_dists", js_dists)
        self.curt_selected = [js_dists.index(max(js_dists))]

    def adaptive_clusters(self):
        self.num_clusters = self.num_clusters // 2 if self.num_clusters // 2 > 2 else 2
        return self.num_clusters

        # self.delta_critical_period()
        # return self.fix

    def sync_matrix(self):
        self.prev_workers_matrix = deepcopy(self.workers_matrix)
        self.prev_matrix = deepcopy(self.curt_matrix)
        self.workers_matrix.clear()
        self.curt_matrix.zero_()

        for i in range(self.workers):
            self.workers_matrix.append(self.workers_nodes[i].cell.wrapper.get_logits_matrix())

        self.curt_matrix = torch.div(sum(self.workers_matrix), len(self.workers_matrix))

        if self.start_matrix is None:
            self.start_matrix = deepcopy(self.curt_matrix)
        global_container.flash('avg_matrix', deepcopy(self.curt_matrix).numpy())

        # global_logger.info(f"======curt: {self.curt_matrix}======")
        # global_logger.info(f"======prev: {self.prev_matrix}======")
        # global_logger.info(f"======start: {self.start_matrix}======")

    def info_aggregation(self):
        workers_dict = []
        drag_cnt = int(self.drag * len(self.curt_selected))

        for index in random.sample(self.curt_selected, len(self.curt_selected) - drag_cnt):
            workers_dict.append(self.workers_nodes[index].cell.access_model().state_dict())

        for _ in range(drag_cnt):
            workers_dict.append(self.merge.pre_dict)

        self.merge.merge_dict(workers_dict)

        for index in self.curt_selected:
            self.workers_nodes[index].cell.decay_lr(self.pace)

    def schedule_strategy(self):
        self.curt_selected.clear()

        # # TODO: Ablation
        # super(FedLAMaster, self).schedule_strategy()
        # return

        if self.curt_round == 0:
            super(FedLAMaster, self).schedule_strategy()
            return

        # # CLP shrink
        # if not self.delta_critical_period():
        #     self.single_select()
        #     return

        if self.pipeline_status == 0:
            self.clusters = self.adaptive_clusters()
            global_logger.info(f"======Round{self.curt_round} >> Clusters:{self.clusters}======")
            X = torch.stack(self.workers_matrix, dim=0).numpy()
            n_samples, dim1, dim2 = X.shape
            flattened_X = X.reshape(n_samples, dim1 * dim2)
            clustering = AgglomerativeClustering(n_clusters=self.clusters).fit(flattened_X)
            self.clusters_indices = clustering.labels_

            # debug
            cnt = np.unique(self.clusters_indices, return_counts=True)[1]
            global_logger.info(f"======Round{self.curt_round} >> Cluster Ret:{len(cnt)}======")

            # # CFL - diff
            # TODO: CFL-diff
            # self.diff_cluster_most_case()
            self.diff_cluster_lea_case()

        # # CFL - sim
        # self.curt_selected = np.where(self.clusters_indices == self.pipeline_status)[0].tolist()
        # self.pipeline_status = (self.pipeline_status + 1) % self.clusters

        self.curt_selected = deepcopy(self.pipeline[self.pipeline_status])

        self.pipeline_status = (self.pipeline_status + 1) % self.max_round

        global_logger.info(f"======Round{self.curt_round} >> Select Index:{self.curt_selected}======")

        # to optim FedLA
        if len(self.curt_selected) == 1:
            return

        if len(self.curt_selected) > self.plan:
            self.curt_selected = random.sample(self.curt_selected, self.plan)

    def diff_cluster_lea_case(self):
        # 计算每个唯一元素的出现次数
        unique_elements, counts = np.unique(self.clusters_indices, return_counts=True)

        # 确定平均出现次数
        self.max_round = (np.min(counts) + np.max(counts)) // 2

        # 初始化分组
        self.pipeline = [[] for _ in range(self.max_round)]

        # 分配索引到不同的组
        for element in unique_elements:
            indices = np.where(self.clusters_indices == element)[0]
            split_indices = np.array_split(indices, self.max_round)
            for group_idx in range(self.max_round):
                self.pipeline[group_idx].extend(split_indices[group_idx] if group_idx < len(split_indices) else [])

    def diff_cluster_most_case(self):
        # 计算每个唯一元素的出现次数
        unique_elements, counts = np.unique(self.clusters_indices, return_counts=True)

        # 确定最大出现次数
        self.max_round = np.max(counts)

        # 初始化分组
        self.pipeline = [[] for _ in range(self.max_round)]

        # 分配索引到不同的组
        for element in unique_elements:
            indices = np.where(self.clusters_indices == element)[0]
            extended_indices = list(np.resize(indices, self.max_round))
            for group_idx in range(self.max_round):
                self.pipeline[group_idx].append(extended_indices[group_idx])

    def drive_workers(self, *_args, **kwargs):
        global_container.flash('selected_workers', deepcopy(self.curt_selected))
        for index in self.curt_selected:
            self.workers_nodes[index].local_train(self.curt_matrix)

        # TODO: optim cnt
        self.sync_matrix()

    def delta_critical_period(self):
        start_matrix = self.start_matrix.numpy()
        curt_matrix = self.curt_matrix.numpy()
        prev_matrix = self.prev_matrix.numpy()

        js_div_row = []
        for row in range(len(start_matrix)):
            row1 = remove_top_k_elements(start_matrix[row], 1)
            row2 = remove_top_k_elements(curt_matrix[row], 1)
            js_div_row.append(jensenshannon(row1, row2))
        mean_values_row = np.mean(np.array(js_div_row))
        js_divergences1 = -mean_values_row

        diagonal1 = np.diag(prev_matrix)
        diagonal2 = np.diag(curt_matrix)

        js_divergences2 = jensenshannon(diagonal1, diagonal2)
        adapt_dist = self.cft * js_divergences1 + js_divergences2
        global_logger.info(f"======adapt_dist: {adapt_dist}======")
        global_container.flash('adapt_dist', adapt_dist)
        return (np.isnan(adapt_dist)) or (adapt_dist >= self.threshold)


class ScaffoldMaster(FLMaster):
    def __init__(self, workers: int, activists: int, local_epoch: int,
                 loader: tdata.dataloader, workers_loaders: dict, local_batch: int):
        """

        :param workers:
        :param activists:
        :param local_epoch:
        :param loader: *only in simulation*
        :param workers_loaders: *only in simulation*
        """
        master_cell = SingleCell(loader, Wrapper=ScaffoldWrapper)
        super().__init__(workers, activists, local_epoch, master_cell)
        self.control, self.delta_control, self.delta_y = dict(), dict(), dict()
        for k, v in self.cell.access_model().named_parameters():
            self.control[k] = torch.zeros_like(v.data)
            self.delta_control[k] = torch.zeros_like(v.data)
            self.delta_y[k] = torch.zeros_like(v.data)

        workers_cells = [SingleCell(loader, Wrapper=ScaffoldWrapper)
                         for loader in list(workers_loaders.values())]
        self.workers_nodes = [ScaffoldWorker(index, cell) for index, cell in enumerate(workers_cells)]

    def drive_workers(self):
        for index in self.curt_selected:
            self.workers_nodes[index].local_train(self.control)
            self.workers_nodes[index].update_control(self.control)

    def info_aggregation(self):
        x = {}
        c = {}
        for k, v in self.workers_nodes[0].cell.access_model().named_parameters():
            x[k] = torch.zeros_like(v.data)
            c[k] = torch.zeros_like(v.data)

        for j in self.curt_selected:
            for k, v in self.workers_nodes[j].cell.access_model().named_parameters():
                x[k] += self.workers_nodes[j].delta_y[k] / len(self.curt_selected)  # averaging
                c[k] += self.workers_nodes[j].delta_control[k] / len(self.curt_selected)  # averaging

        for k, v in self.cell.access_model().named_parameters():
            v.data += x[k].data
            self.control[k].data += c[k].data * (len(self.curt_selected) / self.workers)

        self.merge.union_dict = deepcopy(self.cell.access_model().state_dict())


class MoonMaster(FLMaster):

    def __init__(self, workers: int, activists: int, local_epoch: int,
                 loader: tdata.dataloader, workers_loaders: dict,
                 mu: float, T: int):
        master_cell = SingleCell(loader, Wrapper=MoonWrapper)
        super().__init__(workers, activists, local_epoch, master_cell)

        workers_cells = [SingleCell(loader, Wrapper=MoonWrapper)
                         for loader in list(workers_loaders.values())]
        self.workers_nodes = [MoonWorker(index, cell) for index, cell in enumerate(workers_cells)]

        self.mu = mu
        self.T = T

    def drive_workers(self):
        for index in self.curt_selected:
            self.workers_nodes[index].local_train(deepcopy(self.cell.access_model()),
                                                  self.mu, self.T)
