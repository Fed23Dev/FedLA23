from copy import deepcopy
from timeit import default_timer as timer
from typing import List

import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as tdata
from thop import profile
from torch.cuda.amp import GradScaler
from torch.nn.functional import binary_cross_entropy_with_logits
from torch.optim import lr_scheduler
from torch.optim.lr_scheduler import ReduceLROnPlateau
from yacs.config import CfgNode

from dl.compress.DKD import DKD
from dl.wrapper import DeviceManager
from dl.wrapper.optimizer import SGD_PruneFL
from dl.wrapper.optimizer.WarmUpCosinLR import WarmUPCosineLR
from dl.wrapper.optimizer.WarmUpStepLR import WarmUPStepLR
from env.running_env import *
from env.static_env import *
from env.support_config import *
from utils.objectIO import pickle_mkdir_save, pickle_load


def error_mess(class_name: str, param: str) -> str:
    return f"Create an instance of the {class_name} need necessary {param} parameter."


# 参数自适应选择 kwargs*
# 为空 给出默认配置
class VWrapper:
    ERROR_MESS1 = "Model not support."
    ERROR_MESS2 = "Optimizer not support."
    ERROR_MESS3 = "Scheduler not support."
    ERROR_MESS4 = "Loss function not support."
    ERROR_MESS5 = "Checkpoint do not find model_key attribute."

    def __init__(self, model: nn.Module, train_dataloader: tdata.dataloader,
                 optimizer: VOptimizer, scheduler: VScheduler, loss: VLossFunc):
        self.optimizer_type = optimizer
        self.scheduler_type = scheduler
        self.loss_type = loss

        self.device = None
        self.loss_func = None
        self.lr_scheduler = None
        self.optimizer = None

        self.model = model
        self.loader = train_dataloader

        self.latest_acc = 0.0
        self.latest_loss = 0.0
        self.curt_batch = 0
        self.curt_epoch = 0

        self.seed = 2022
        self.scaler = GradScaler()

    # to impl
    def default_config(self):
        pass

    # 初始化当前设备状态包括CPU和GPU
    def init_device(self, use_gpu: bool, gpu_ids: List):
        self.device = DeviceManager.VDevice(use_gpu, gpu_ids)
        self.model = self.device.bind_model(self.model)

    # 初始化优化器
    def init_optim(self, learning_rate: float, momentum: float,
                   weight_decay: float, nesterov: bool):
        if self.optimizer_type == VOptimizer.SGD:
            self.optimizer = optim.SGD(self.model.parameters(), lr=learning_rate,
                                       momentum=momentum, weight_decay=weight_decay, nesterov=nesterov)
        elif self.optimizer_type == VOptimizer.SGD_PFL:
            self.optimizer = SGD_PruneFL.SGD_PFL(self.model.parameters(), lr=learning_rate)
        elif self.optimizer_type == VOptimizer.ADAM:
            self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate,
                                        weight_decay=weight_decay)
        else:
            assert False, self.ERROR_MESS2

    # 初始化学习率调度器和损失函数
    def init_scheduler_loss(self, step_size: int, gamma: float, T_max: int, warm_up_steps: int, min_lr: float):
        if self.scheduler_type == VScheduler.StepLR:
            self.lr_scheduler = lr_scheduler.StepLR(self.optimizer, step_size=step_size, gamma=gamma)
        elif self.scheduler_type == VScheduler.CosineAnnealingLR:
            self.lr_scheduler = lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=T_max)
        elif self.scheduler_type == VScheduler.WarmUPCosineLR:
            self.lr_scheduler = WarmUPCosineLR(self.optimizer, warm_up_steps, T_max, lr_min=min_lr)
        elif self.scheduler_type == VScheduler.ReduceLROnPlateau:
            self.lr_scheduler = ReduceLROnPlateau(self.optimizer, 'min')
        elif self.scheduler_type == VScheduler.WarmUPStepLR:
            self.lr_scheduler = WarmUPStepLR(self.optimizer, step_size=step_size, gamma=gamma)
        else:
            assert False, self.ERROR_MESS3

        if self.loss_type == VLossFunc.Cross_Entropy:
            self.loss_func = binary_cross_entropy_with_logits
        else:
            assert False, self.ERROR_MESS4

    # 获取数据和标签的形状
    def running_scale(self):
        inputs, label = next(iter(self.loader))
        data_size = inputs.size()
        label_size = label.size()
        return data_size, label_size

    def step_run(self, batch_limit: int, train: bool = False,
                 loader: tdata.dataloader = None, **kwargs) -> (int, int, float):
        """
        单个Epoch的训练或测试过程
        :param batch_limit: Batch数量上限
        :param train: 是否训练模型
        :param loader: 可以选择提供测试集或验证集的Loader
        :return: 正确数量，总数量，loss值
        """
        if train:
            self.model.train()
        else:
            self.model.eval()
        process = "Train" if train else "Test"

        train_loss = 0
        correct = 0
        total = 0

        curt_loader = self.loader if loader is None else loader

        for batch_idx, (inputs, targets) in enumerate(curt_loader):
            if batch_idx > batch_limit:
                break

            inputs, labels = self.device.on_tensor(inputs, targets)
            pred = self.model(inputs)

            loss = self.loss_compute(pred, labels, **kwargs)

            if train:
                self.optim_step(loss)

            _, predicted = pred.max(1)
            _, targets = labels.max(1)

            correct += predicted.eq(targets).sum().item()
            train_loss += loss.item()
            total += targets.size(0)

            self.latest_acc = 100. * correct / total
            self.latest_loss = train_loss / (batch_idx + 1)

            if batch_idx % print_interval == 0 and batch_idx != 0:
                global_logger.info('%s:batch_idx:%d | Loss: %.6f | Acc: %.3f%% (%d/%d)'
                                   % (process, batch_idx, self.latest_loss, self.latest_acc, correct, total))
            self.curt_batch += 1

        # total epoch scheduler_step
        if train:
            self.curt_epoch += 1
            global_container.flash(f"{args.exp_name}_acc", round(self.latest_acc, 4))
            self.scheduler_step()
        return correct, total, self.latest_loss

    # 优化器步进过程
    def optim_step(self, loss: torch.Tensor, speedup: bool = False):
        if speedup:
            self.optimizer.zero_grad()
            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

    # 学习率调度器步进过程
    def scheduler_step(self):
        if self.scheduler_type == VScheduler.ReduceLROnPlateau:
            self.lr_scheduler.step(metrics=self.latest_loss)
        else:
            self.lr_scheduler.step()

    # 获得模型的实例对象
    def access_model(self) -> nn.Module:
        return self.device.access_model()

    # 手动调整学习率
    def adjust_lr(self, factor: float):
        for param_group in self.optimizer.param_groups:
            param_group["lr"] *= factor

    # 获取展示最新的的学习率
    def show_lr(self) -> float:
        lr = self.optimizer.state_dict()['param_groups'][0]['lr']
        global_logger.info(f"The current learning rate: {lr}======>")
        return lr

    # 将tensor与模型处于的设备调整一致
    def sync_tensor(self, tensor: torch.Tensor) -> torch.Tensor:
        return next(self.device.on_tensor(tensor))

    # 冻结保存当前的模型参数 to modify
    def save_checkpoint(self, file_path: str):
        exp_checkpoint = {"exp_name": args.exp_name, "state_dict": self.device.freeze_model(),
                          "batch_size": args.batch_size, "last_epoch": self.curt_epoch,
                          "init_lr": args.learning_rate}
        pickle_mkdir_save(exp_checkpoint, file_path)

    # 加载已有的模型参数 to modify
    def load_checkpoint(self, path: str, model_key: str = 'state_dict'):
        if path.find('.pt') == -1:
            checkpoint = pickle_load(path)
        else:
            checkpoint = torch.load(path, map_location=torch.device('cpu'))
        assert model_key in checkpoint.keys(), self.ERROR_MESS5
        self.device.load_model(checkpoint[model_key])

    # 测试当前模型的性能，传入测试集的dataloader
    def valid_performance(self, test_dataloader: tdata.dataloader):
        # inputs = torch.rand(*(self.running_scale()[0]))
        # cpu_model = deepcopy(self.device.access_model()).cpu()
        # flops, params = profile(cpu_model, inputs=(inputs,))
        inputs = torch.rand(*(self.running_scale()[0])).cuda()
        gpu_model = deepcopy(self.device.access_model()).cuda()
        flops, params = profile(gpu_model, inputs=(inputs,))

        time_start = timer()
        correct, total, test_loss = self.step_run(valid_limit, loader=test_dataloader)
        time_cost = timer() - time_start
        total_params = sum(p.numel() for p in self.model.parameters())
        total_trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)

        global_logger.info('Loss: %.3f | Acc: %.3f%% (%d/%d)'
                           % (test_loss, 100. * correct / total, correct, total))

        global_logger.info('Time cost: %.6f | FLOPs: %d | Params: %d'
                           % (time_cost, flops, params))

        global_logger.info('Total params: %d | Trainable params: %d'
                           % (total_params, total_trainable_params))

    # 自定义损失函数计算
    # 覆写该方法
    def loss_compute(self, pred: torch.Tensor, targets: torch.Tensor, **kwargs) -> torch.Tensor:
        return self.loss_func(pred, targets)


class ProxWrapper(VWrapper):
    ERROR_MESS6 = "FedProx must provide pre_params parameter."

    def __init__(self, model: nn.Module, train_dataloader: tdata.dataloader, optimizer: VOptimizer,
                 scheduler: VScheduler, loss: VLossFunc):
        super().__init__(model, train_dataloader, optimizer, scheduler, loss)

    def loss_compute(self, pred: torch.Tensor, targets: torch.Tensor, **kwargs) -> torch.Tensor:
        assert "pre_params" in kwargs.keys(), self.ERROR_MESS6
        loss = self.loss_func(pred, targets)
        proximal_term = 0.0
        for w, w_t in zip(self.model.parameters(), kwargs["pre_params"]):
            proximal_term += (w - w_t).norm(2)
        loss += (args.mu / 2) * proximal_term
        return loss


class LAWrapper(VWrapper):
    ERROR_MESS7 = "FedLA must provide teacher_model parameter."

    def __init__(self, model: nn.Module, train_dataloader: tdata.dataloader, optimizer: VOptimizer,
                 scheduler: VScheduler, loss: VLossFunc):
        super().__init__(model, train_dataloader, optimizer, scheduler, loss)
        cfg = CfgNode()
        cfg.CE_WEIGHT = args.CE_WEIGHT
        cfg.ALPHA = args.ALPHA
        cfg.BETA = args.BETA
        cfg.T = args.T
        cfg.WARMUP = args.WARMUP

        self.kd_batch = args.KD_BATCH
        self.kd_epoch = args.KD_EPOCH
        self.kd_curt_epoch = 0
        self.distillers = DKD(cfg)

    def dkd_loss_optim(self, teacher_model: nn.Module):
        self.model.train()
        teacher_model.eval()
        for e in range(self.kd_epoch):
            for batch_idx, (inputs, targets) in enumerate(self.loader):
                if batch_idx > self.kd_batch:
                    break

                inputs, labels = self.device.on_tensor(inputs, targets)

                stu_pred = self.model(inputs)
                with torch.no_grad():
                    tea_pred = teacher_model(inputs)

                losses_dict = self.distillers.forward_train(stu_pred, tea_pred, labels, self.kd_curt_epoch)[1]

                loss = sum([ls.mean() for ls in losses_dict.values()])

                self.optim_step(loss)
            self.scheduler_step()

        self.kd_curt_epoch += self.kd_epoch

    # Tensor Size: classes * classes
    def get_logits_dist(self, batch_limit: int = args.logits_batch_limit) -> torch.Tensor:
        label_dtype = torch.int64

        sum_logits = torch.zeros(args.num_classes, args.num_classes)
        sum_labels = torch.zeros(args.num_classes, dtype=label_dtype)

        with torch.no_grad():
            for batch_idx, (inputs, targets) in enumerate(self.loader):
                if batch_idx > batch_limit:
                    break
                labels = torch.argmax(targets, -1)
                _labels, _cnt = torch.unique(labels, return_counts=True)
                labels_cnt = torch.zeros(args.num_classes, dtype=label_dtype) \
                    .scatter_(dim=0, index=_labels, src=_cnt)

                logits = self.model(inputs).cpu()

                if batch_idx == 0:
                    sum_logits = sum_logits.type(logits.dtype)

                # 扩展的标签索引 [0, 1] >> [[0, 0], [1, 1]]
                logits_index = labels.unsqueeze(1).expand(logits.size())
                # 自然数索引
                labels_index = torch.tensor(list(range(args.num_classes)))

                sum_logits.scatter_add_(dim=0, index=logits_index, src=logits)
                sum_labels.scatter_add_(dim=0, index=labels_index, src=labels_cnt)

        # 消掉无穷和未定义，因为non-iid
        zero = torch.zeros_like(sum_logits)
        one = torch.ones_like(sum_logits)
        div_labels = sum_labels.unsqueeze(1).expand(sum_logits.size())
        sum_logits = torch.where(sum_logits == 0, one, sum_logits)
        avg_logits = sum_logits / div_labels
        avg_logits = torch.where(avg_logits == torch.inf, zero, avg_logits)
        return avg_logits

# 传入真实数据的dataloader对模型进行测试或训练
# def step_run(self, batch_limit: int, train: bool = False,
#              pre_params: Iterator = None, loader: tdata.dataloader = None,
#              loss_weight: torch.Tensor = None) -> (int, float, int):
#     if train:
#         self.model.train()
#     else:
#         self.model.eval()
#
#     train_loss = 0
#     correct = 0
#     total = 0
#     process = "Train" if train else "Test"
#
#     curt_loader = loader if loader is not None else self.loader
#
#     for batch_idx, (inputs, targets) in enumerate(curt_loader):
#         if batch_idx > batch_limit:
#             break
#
#         inputs, labels = self.device.on_tensor(inputs, targets)
#
#         with autocast():
#             pred = self.model(inputs)
#
#             if loss_weight is None:
#                 loss = self.loss_func(pred, labels)
#             else:
#                 loss_weight = self.sync_tensor(loss_weight)
#                 loss = self.loss_func(pred, labels, loss_weight)
#
#         if train:
#             if pre_params is not None:
#                 # fedprox
#                 proximal_term = 0.0
#                 for w, w_t in zip(self.model.parameters(), pre_params):
#                     proximal_term += (w - w_t).norm(2)
#                 loss += (args.mu / 2) * proximal_term
#                 # fedprox
#
#             self.optim_step(loss, True)
#
#         _, predicted = pred.max(1)
#         _, targets = labels.max(1)
#         correct += predicted.eq(targets).sum().item()
#         train_loss += loss.item()
#         total += targets.size(0)
#
#         self.latest_acc = 100. * correct / total
#         self.latest_loss = train_loss / (batch_idx + 1)
#
#         if batch_idx % print_interval == 0 and batch_idx != 0:
#             global_logger.info('%s:batch_idx:%d | Loss: %.6f | Acc: %.3f%% (%d/%d)'
#                                % (process, batch_idx, self.latest_loss, self.latest_acc, correct, total))
#         self.curt_batch += 1
#
#     if train:
#         # gc.collect()
#         # torch.cuda.empty_cache()
#         self.curt_epoch += 1
#         global_container.flash(f"{args.exp_name}_acc", round(self.latest_acc, 3))
#         if loader is None:
#             self.scheduler_step()
#
#     return correct, total, self.latest_loss
