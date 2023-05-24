import copy

import torch.utils.data as tdata
from timeit import default_timer as timer

from dl.SingleCell import SingleCell
from env.running_env import global_logger, file_repo
from federal.simulation.FLnodes import FLMaster
from federal.simulation.Worker import FedAvgWorker, FedProxWorker, HRankFLWorker, CALIMFLWorker, FedIRWorker
from utils.objectIO import pickle_mkdir_save
from typing import List


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
        master_cell = SingleCell(loader, False)
        super().__init__(workers, activists, local_epoch, master_cell)

        workers_cells = [SingleCell(loader, True) for loader in list(workers_loaders.values())]
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
        master_cell = SingleCell(loader, False)
        super().__init__(workers, activists, local_epoch, master_cell)

        workers_cells = [SingleCell(loader, True) for loader in list(workers_loaders.values())]
        self.workers_nodes = [FedProxWorker(index, cell) for index, cell in enumerate(workers_cells)]

    def drive_workers(self):
        for index in self.curt_selected:
            self.workers_nodes[index].local_train(self.cell.access_model().parameters())


class FedIRMaster(FLMaster):
    def __init__(self, workers: int, activists: int, local_epoch: int,
                 loader: tdata.dataloader, workers_loaders: dict,
                 loss_weights: List):
        """

        :param workers:
        :param activists:
        :param local_epoch:
        :param loader: *only in simulation*
        :param workers_loaders: *only in simulation*
        """
        master_cell = SingleCell(loader, False)
        super().__init__(workers, activists, local_epoch, master_cell)

        workers_cells = [SingleCell(loader, True) for loader in list(workers_loaders.values())]
        self.workers_nodes = [FedIRWorker(index, cell, ratio)
                              for index, (cell, ratio) in enumerate(zip(workers_cells, loss_weights))]

    def drive_workers(self):
        for index in self.curt_selected:
            self.workers_nodes[index].local_train()


class HRankFLMaster(FLMaster):
    remain_rounds = 10

    def __init__(self, workers: int, activists: int, local_epoch: int,
                 loader: tdata.dataloader, workers_loaders: dict):
        """

        :param workers:
        :param activists:
        :param local_epoch:
        :param loader: *only in simulation*
        :param workers_loaders: *only in simulation*
        """
        master_cell = SingleCell(loader, True)
        super().__init__(workers, activists, local_epoch, master_cell)

        workers_cells = [SingleCell(loader, True) for loader in list(workers_loaders.values())]
        self.workers_nodes = [HRankFLWorker(index, cell) for index, cell in enumerate(workers_cells)]

    def union_run(self, rounds: int):
        time_start = timer()
        for i in range(rounds):
            global_logger.info(f"======Federal Round: {i + 1}======")
            self.schedule_strategy()
            self.info_sync()
            self.drive_workers()
            self.info_aggregation()
            self.weight_redo()

            if i == rounds - 10:
                global_logger.info(f"======HRankFL======")
                self.structure_prune()

            self.curt_round = self.curt_round + 1

            time_cost = timer() - time_start
            global_logger.info(f"======Time Cost: {time_cost}s======")

        global_logger.info(f"Federal train finished======>")
        self.global_performance_detail()

    def drive_workers(self):
        for index in self.curt_selected:
            self.workers_nodes[index].local_train()

    def structure_prune(self):
        cp_model = self.cell.structure_prune_process(random=False)
        self.cell.sync_model(cp_model)
        self.merge.sync_dict(cp_model.state_dict())
        for node in self.workers_nodes:
            node.cell.sync_model(copy.deepcopy(cp_model))


class CALIMFLMaster(FLMaster):
    ERROR_MESS1 = "client_sq_grads has no data."

    def __init__(self, workers: int, activists: int, local_epoch: int,
                 loader: tdata.dataloader, workers_loaders: dict,
                 aug_loaders: dict):
        """

        :param workers:
        :param activists:
        :param local_epoch:
        :param loader: *only in simulation*
        :param workers_loaders: *only in simulation*
        """
        master_cell = SingleCell(loader, True)
        super().__init__(workers, activists, local_epoch, master_cell)

        workers_cells = [SingleCell(loader, True) for loader in list(workers_loaders.values())]

        self.workers_nodes = [CALIMFLWorker(index, cell, loader)
                              for index, (cell, loader) in enumerate(zip(workers_cells, list(aug_loaders.values())))]

        self.first_prune = True
        self.rate = None
        self.rate_provider = None
        self.check_inter = None
        self.random = None

        self.grads = []
        self.grads_container = []

    def prune_init(self, rate: list, inter: int, random_data: bool):
        self.rate = rate
        self.check_inter = inter
        self.random = random_data

    def union_run(self, rounds: int):
        time_start = timer()
        for i in range(rounds):
            global_logger.info(f"======Federal Round: {i + 1}======")

            self.schedule_strategy()
            self.info_sync()
            self.drive_workers()
            self.info_aggregation()
            self.weight_redo()

            # # 迭代自适应剪枝 剪枝率和时机都自适应
            # if i != rounds - 1:
            #     self.master_prune()
            if i == rounds - 10:
                self.structure_prune()

            self.serialize_size()
            self.curt_round = self.curt_round + 1

            time_cost = timer() - time_start
            global_logger.info(f"======Time Cost: {time_cost}s======")

        path, _ = file_repo.new_seq('model_weight_size')
        pickle_mkdir_save(self.des_size, path)

        global_logger.info(f"Federal train finished======>")
        self.global_performance_detail()

    def drive_workers(self):
        client_grads = []
        for index in self.curt_selected:
            self.workers_nodes[index].local_train(client_grads)
            self.workers_nodes[index].enhance_local_train()

        assert len(client_grads) != 0, self.ERROR_MESS1
        self.grads = [0 for _ in range(len(client_grads[0]))]
        for client_grad in client_grads:
            for ind, val in enumerate(client_grad):
                self.grads[ind] += val

    def master_prune(self):
        if self.curt_round % self.check_inter == 0:
            self.cell.prune_model(grads=self.grads, random=self.random, auto_inter=False)
        else:
            global_logger.info(f"Do not prune in round:{self.curt_round}.")
        self.grads.clear()

    def structure_prune(self):
        cp_model = self.cell.structure_prune_process(grads=self.grads, random=self.random)
        self.cell.sync_model(cp_model)
        self.merge.sync_dict(cp_model.state_dict())
        for node in self.workers_nodes:
            node.cell.sync_model(copy.deepcopy(cp_model))


class FedLAMaster(FLMaster):
    def __init__(self, workers: int, activists: int, local_epoch: int, master_cell: SingleCell,
                 loader: tdata.dataloader, workers_loaders: dict, data_dist: list):
        super().__init__(workers, activists, local_epoch, master_cell)

        master_cell = SingleCell(loader, True)
        super().__init__(workers, activists, local_epoch, master_cell)

        workers_cells = [SingleCell(loader, True) for loader in list(workers_loaders.values())]

        self.workers_nodes = [CALIMFLWorker(index, cell, loader)
                              for index, (cell, loader) in enumerate(zip(workers_cells))]

        self.dataset_dist = data_dist
        self.curt_dist = [0.] * len(data_dist[0])

    def schedule_strategy(self):
        pass



    def drive_workers(self, *_args, **kwargs):
        pass




