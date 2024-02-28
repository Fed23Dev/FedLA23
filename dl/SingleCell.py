import math
from typing import Any, List

import torch
import torch.nn as nn
import torch.utils.data as tdata

from dl.model.model_util import create_model
from dl.wrapper.Wrapper import VWrapper
from dl.wrapper.ExitDriver import ExitManager
from env.running_env import args, global_file_repo, global_container
from dl.data.dataProvider import get_data_loader
from env.running_env import global_logger


class SingleCell:
    ERROR_MESS1 = "Test must specify batch_limit."

    def __init__(self, train_loader: tdata.DataLoader = None, test_loader: tdata.DataLoader = None,
                 Wrapper: Any = VWrapper):
        """

        :param train_loader: 联邦仿真参数，便于批量创建分配dataloader *only in simulation*
        """
        # 训练数据个数
        self.latest_feed_amount = 0
        self.latest_grad = []

        # 当前训练的批次
        self.train_epoch = 1

        # Wrapper init
        model = create_model(args.model, num_classes=args.num_classes, in_channels=args.in_channels)
        if train_loader is None:
            dataloader = get_data_loader(args.dataset, data_type="train",
                                         batch_size=args.batch_size, shuffle=True)
        else:
            dataloader = train_loader

        # Wrapper init
        self.wrapper = Wrapper(model, dataloader, args.optim, args.scheduler, args.loss_func)
        self.wrapper.init_device(args.use_gpu, args.gpu_ids)
        self.wrapper.init_optim(args.learning_rate, args.momentum, args.weight_decay, args.nesterov)

        total_epoch = args.local_epoch * args.federal_round * args.active_workers \
            if args.federal else args.local_epoch
        self.wrapper.init_scheduler_loss(args.step_size, args.gamma, total_epoch, args.warm_steps, args.min_lr)

        if args.pre_train:
            self.wrapper.load_checkpoint(global_file_repo.model_path)

        self.test_dataloader = dataloader if test_loader is None else test_loader
        self.exit_manager = ExitManager(self.wrapper)

    # 更新模型
    def sync_model(self, model: nn.Module):
        # if model.state_dict() != self.wrapper.model.state_dict():
        #     self.wrapper.model.load_state_dict(model.state_dict())
        self.wrapper.device.bind_model(model)

    # 获取模型对象
    def access_model(self) -> nn.Module:
        return self.wrapper.access_model()

    def run_model(self, train: bool = False, batch_limit: int = 0, **kwargs) -> float:
        """
        根据特定的epoch数目和特定的batch上限对模型进行反复训练，或是基于特定batch上限测试
        :param train: 是否训练模型参数
        :param batch_limit: 每一个epoch下的batch数目限制
        :param kwargs: 特定算法的扩展参数
        :return: 返回本次计算损失
        """
        sum_loss = 0.0
        self.latest_feed_amount = 0
        self.latest_grad.clear()

        if train:
            for i in range(args.local_epoch):
                global_logger.info(f"******The current train epoch: {self.train_epoch + i}******")
                if batch_limit == 0:
                    cort, total, loss = self.wrapper.step_run(args.batch_limit, train, **kwargs)
                else:
                    cort, total, loss = self.wrapper.step_run(batch_limit, train, **kwargs)
                sum_loss += loss
                self.latest_feed_amount += total
                if len(self.latest_grad) == 0:
                    self.latest_grad = self.wrapper.get_last_grad()
                else:
                    self.latest_grad += self.wrapper.get_last_grad()
                self.show_lr()
            self.train_epoch += args.local_epoch
            return sum_loss / args.local_epoch
        else:
            assert batch_limit != 0, self.ERROR_MESS1
            cort, total, loss = self.wrapper.step_run(batch_limit, train=False)
            global_logger.info("======Current Test Acc: %.3f%% (%d/%d)======" % (cort / total * 100, cort, total))
            global_container.flash(f'{args.exp_name}-test_acc', cort / total * 100)
            return loss

    def max_model_performance(self, models: List[torch.nn.Module]) -> torch.nn.Module:
        optim_model = models[0]
        min_loss = torch.tensor(1000000000)
        for model in models:
            self.sync_model(model)
            _, _, loss = self.wrapper.step_run(batch_limit=10, train=False)
            if loss < min_loss:
                min_loss = loss
                optim_model = model
        self.sync_model(optim_model)
        return optim_model


    # 测试模型性能
    def test_performance(self):
        self.wrapper.valid_performance(self.test_dataloader)

    # 调整模型学习率
    def decay_lr(self, epoch: int):
        self.wrapper.adjust_lr(math.pow(args.gamma, epoch))

    # 查看学习率
    def show_lr(self):
        global_logger.info(f"The current learning rate: {self.wrapper.get_lr()}======>")

    # 程序退出时保存关键度量指标、配置信息和模型参数
    def exit_proc(self, check: bool = False, one_key: str = None):
        if check:
            self.exit_manager.checkpoint_freeze()
        self.exit_manager.running_freeze(one_key)
        config_str = self.exit_manager.config_freeze()
        config_str = f"{{{config_str}}}".replace("\n", " || ")
        global_logger.info(config_str)
