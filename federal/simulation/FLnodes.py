import os
import random
from abc import ABC, abstractmethod
from collections import OrderedDict
from functools import singledispatchmethod
from typing import List, Iterator
import torch.utils.data as tdata
from timeit import default_timer as timer

from dl.SingleCell import SingleCell
import dl.compress.compress_util as com_util

# FedProx Nodes
from env.running_env import args, global_logger
from federal.aggregation.FedAvg import FedAvg
from utils.objectIO import pickle_mkdir_save


class FLMaster(ABC):
    def __init__(self, workers_num: int, schedule_num: int, local_epoch: int,
                 master_cell: SingleCell):
        self.workers = workers_num
        self.plan = schedule_num
        self.pace = local_epoch * schedule_num // 2

        self.cell = master_cell

        self.merge = FedAvg(master_cell.access_model().state_dict())

        self.des_size = []
        self.curt_selected = []
        self.workers_nodes = []

        self.pre_dict = self.cell.access_model().state_dict()
        self.curt_dict = self.cell.access_model().state_dict()
        self.pre_loss = 9999
        self.curt_loss = 0
        self.curt_round = 0

    def schedule_strategy(self):
        self.curt_selected = random.sample(range(0, self.workers), self.plan)

    def global_performance_detail(self):
        self.cell.test_performance()

    def weight_redo(self):
        self.merge.weight_redo(self.cell)

    # 稀疏模型参数的反序列化优化
    def serialize_size(self, coo_path: str = "coo"):
        model_dict = self.cell.access_model().cpu().state_dict()
        coo_dict = com_util.dict_coo_express(model_dict)
        pickle_mkdir_save(coo_dict, coo_path)
        self.des_size.append(os.stat(coo_path).st_size / (1024 * 1024))
        if args.use_gpu:
            self.cell.access_model().cuda()

    def info_aggregation(self):
        workers_dict = []
        for index in self.curt_selected:
            workers_dict.append(self.workers_nodes[index].cell.access_model().state_dict())
        self.merge.merge_dict(workers_dict)
        for index in self.curt_selected:
            self.workers_nodes[index].cell.decay_lr(self.pace)

    def info_sync(self):
        workers_cells = []
        for index in self.curt_selected:
            workers_cells.append(self.workers_nodes[index].cell)
        self.merge.all_sync(workers_cells, 0)

    def union_run(self, rounds: int):
        for i in range(rounds):
            time_start = timer()
            global_logger.info(f"======Federal Round: {i + 1}======")
            self.schedule_strategy()
            self.info_sync()
            self.drive_workers()
            self.info_aggregation()
            self.weight_redo()
            self.curt_round = self.curt_round + 1

            time_cost = timer() - time_start
            global_logger.info(f"======Time Cost: {time_cost}s======")

        global_logger.info(f"Federal train finished======>")
        self.global_performance_detail()

    @abstractmethod
    def drive_workers(self, *_args, **kwargs):
        pass


class FLWorker(ABC):
    def __init__(self, worker_id: int, worker_cell: SingleCell):
        self.id = worker_id
        self.cell = worker_cell

    @abstractmethod
    def local_train(self, *_args, **kwargs):
        pass
