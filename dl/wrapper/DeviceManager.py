from typing import List
from collections import OrderedDict
import torch
from torch import nn
import logging

from utils import Vlogger


# Auto config
class VDevice:
    CPU_STR_LEN = 3
    LOGGER_PATH = "logs/device_info.log"
    ERROR_MESS1 = "GPU is not available."
    ERROR_MESS2 = "Model must be not null."
    ERROR_MESS3 = "Args must be path(str) or dict."
    PREFIX = "module."

    def __init__(self, gpu: bool = True, ids=None, logger: logging.Logger = None) -> None:
        if ids is None:
            ids = [0]
        self.flag = torch.cuda.is_available()
        self.dev_list = ids
        self.model = None
        self.GPUs: bool = False
        self.expression_access = None
        if logger is None:
            self.logger = Vlogger.VLogger(self.LOGGER_PATH).logger
        if gpu:
            assert self.flag is True, self.ERROR_MESS1
            self.device = torch.device("cuda:%d" % self.dev_list[0])
            self.last_choice = True
        else:
            self.device = torch.device("cpu")
            self.last_choice = False

    def bind_model(self, model: nn.Module) -> nn.Module:
        self.model = model
        if self.last_choice:
            self.to_gpu()
        else:
            self.to_cpu()
        return self.model

    def switch_device(self):
        if self.last_choice:
            self.device = torch.device("cpu")
            self.last_choice = False
        else:
            self.device = torch.device("cuda:%d" % self.dev_list[0])
            self.last_choice = True

    def to_gpu(self):
        assert self.model is not None, self.ERROR_MESS2
        if len(self.dev_list) > 1:
            self.model = nn.DataParallel(self.model, device_ids=self.dev_list)
            self.model.to(self.device)
            self.GPUs = True
        else:
            self.model.to(self.device)
            self.GPUs = False
        if self.last_choice is False:
            self.switch_device()

    def to_cpu(self):
        assert self.model is not None, self.ERROR_MESS2
        self.model.cpu()
        self.GPUs = False
        if self.last_choice:
            self.switch_device()

    def sync_tensor(self, tensor: torch.Tensor) -> torch.Tensor:
        return tensor.to(self.device)

    def on_tensor(self, *tensors: torch.Tensor):
        for t in tensors:
            yield t.to(self.device)

    # eval() insecure!!!
    def access_model(self) -> nn.Module:
        self.expression_access = "self.model.module" if self.GPUs else "self.model"
        return eval(self.expression_access)

    def freeze_model(self) -> dict:
        assert self.model is not None, self.ERROR_MESS2
        return self.access_model().state_dict()

    def save_model(self, path: str):
        assert self.model is not None, self.ERROR_MESS2
        torch.save(self.access_model().state_dict(), path)

    def load_model(self, path_or_dic):
        assert self.model is not None, self.ERROR_MESS2
        if isinstance(path_or_dic, str):
            state_dict = torch.load(path_or_dic).state_dict()
        elif isinstance(path_or_dic, dict):
            state_dict = path_or_dic
        else:
            assert False, self.ERROR_MESS3
        adapt_dict = OrderedDict()
        load_model_gpus = self.PREFIX in list(state_dict.keys())[0]
        if self.GPUs:
            if not load_model_gpus:
                for k, v in state_dict.items():
                    adapt_dict[self.PREFIX + k] = v
            else:
                adapt_dict = state_dict
        else:
            if load_model_gpus:
                for k, v in state_dict.items():
                    adapt_dict[k.replace(self.PREFIX, '', 1)] = v
            else:
                adapt_dict = state_dict
        self.model.load_state_dict(adapt_dict)

    def direct_load_model(self):
        pass

    def direct_save_model(self):
        pass
