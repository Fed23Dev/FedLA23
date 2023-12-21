import copy
from collections import OrderedDict
from typing import List

import torch

from dl.SingleCell import SingleCell
from env.running_env import args
from federal.aggregation.fed_util import get_speech_right


class FedAvg:
    ERROR_MESS1 = "The length of clients_dicts must be equal to the length of the speech_right."
    MAX_VAL = torch.tensor(99.99)
    SUFF = torch.tensor(2)

    def __init__(self, init_dict: dict):
        self.union_dict = OrderedDict()
        self.pre_dict = copy.deepcopy(init_dict)
        self.last_loss = self.MAX_VAL

    # 输入多个worker节点的model.state_dict()得到dict的列表
    def merge_dict(self, clients_dicts: List[dict], speech_right: List[int] = None) -> dict:
        if speech_right is not None:
            assert len(clients_dicts) == len(speech_right), self.ERROR_MESS1
        else:
            speech_right = get_speech_right(len(clients_dicts))

        for right, dic in zip(speech_right, clients_dicts):
            for k, v in dic.items():
                if k in self.union_dict.keys():
                    self.union_dict[k] += v * right
                else:
                    self.union_dict[k] = v * right
        clients_dicts.clear()
        return copy.deepcopy(self.union_dict)

    # 同步传入worker列表的模型参数，并下调学习率
    def all_sync(self, workers_cells: List[SingleCell], epochs: int = 0):
        for cell in workers_cells:
            if self.union_dict:
                cell.access_model().load_state_dict(self.union_dict)
            # cell.decay_lr(epochs)
        self.union_dict.clear()

    # 撤销本次联邦学习过程，如果学习到的信息比较差
    def weight_redo(self, master_cell: SingleCell):
        master_cell.access_model().load_state_dict(self.union_dict)
        curt_loss = master_cell.run_model(batch_limit=args.test_batch_limit)
        if curt_loss < self.SUFF * self.last_loss:
            self.pre_dict = copy.deepcopy(self.union_dict)
            self.last_loss = curt_loss
        else:
            master_cell.access_model().load_state_dict(self.pre_dict)
            self.union_dict = copy.deepcopy(self.pre_dict)

    def sync_dict(self, checkpoint: dict):
        self.union_dict = checkpoint
        self.pre_dict = checkpoint

