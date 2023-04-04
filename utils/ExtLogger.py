import torch.utils.data as tdata
import torch
import torch.nn as nn
from timeit import default_timer as timer
from copy import deepcopy
from thop import profile
import logging
import uuid


class VLogger:
    def __init__(self, file_path, sout = False) -> None:
        self.file_path = file_path
        self.sout = sout
        self.inner = logging.getLogger(self.log_id())

    def log_id(self) -> str:
        return str(uuid.uuid4())[:8]

    @property
    def logger(self) -> logging.Logger:
        log_format = '[%(levelname)s] - %(asctime)s | %(message)s'
        formatter = logging.Formatter(log_format, datefmt='%m/%d %I:%M:%S %p')
        file_handler = logging.FileHandler(self.file_path)
        file_handler.setFormatter(formatter)
        self.inner.addHandler(file_handler)

        if self.sout:
            stream_handler = logging.StreamHandler()
            stream_handler.setFormatter(formatter)
            self.inner.addHandler(stream_handler)

        self.inner.setLevel(logging.INFO)
        self.inner.propagate = False
        return self.inner


class Ext:
    def __init__(self):
        self.valid_limit = 3
        self.path = "EXT.log"
        self.logger = VLogger(self.path, True).logger
        self.device = torch.device("cpu")

    def valid_performance(self, model: nn.Module, test_dataset):
        testloader = tdata.dataloader(test_dataset, batch_size=128, shuffle=False)
        flops = 0
        params = 0

        time_start = timer()
        with torch.no_grad():
            loader_iter = iter(testloader)
            (inputs, targets) = next(loader_iter)
            tmp = deepcopy(model)
            tmp = tmp.to(self.device)
            inputs = inputs.to(self.device)
            # inputs = torch.randn(self.random_batch_size, 3, 32, 32)
            flops, params = profile(tmp, inputs=(inputs,))

        time_cost = timer() - time_start
        total_params = sum(p.numel() for p in model.parameters())
        total_trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        self.logger.info('Time cost: %.3f | FLOPs: %d | Params: %d'
                           % (time_cost, flops, params))

        self.logger.info('Total params: %d | Trainable params: %d'
                           % (total_params, total_trainable_params))
