import math
from copy import deepcopy
from typing import Iterator

import torch
import torch.nn as nn
import torch.utils.data as tdata

from dl.compress.HyperProvider import IntervalProvider, RateProvider
from dl.compress.VHRank import HRank
from dl.model.model_util import create_model
from dl.wrapper.ExitDriver import ExitManager
from dl.wrapper.Wrapper import VWrapper
from env.running_env import args, file_repo, global_container
from dl.data.dataProvider import get_data_loader
from env.running_env import global_logger
from env.static_env import wu_epoch, wu_batch, vgg16_candidate_rate
from env.support_config import VState


class SingleCell:
    ERROR_MESS1 = "Test must specify batch_limit."

    def __init__(self, train_loader: tdata.DataLoader = None, prune: bool = False,
                 test_loader: tdata.DataLoader = None):
        """

        :param train_loader: 联邦仿真参数，便于批量创建分配dataloader *only in simulation*
        :param prune:
        """
        # 训练数据个数
        self.latest_feed_amount = 0
        # 当前训练的批次
        self.train_epoch = 1

        # Wrapper init
        model = create_model(args.model, num_classes=args.num_classes)
        if train_loader is None:
            dataloader = get_data_loader(args.dataset, data_type="train",
                                         batch_size=args.batch_size, shuffle=True)
        else:
            dataloader = train_loader

        self.wrapper = VWrapper(model, dataloader, args.optim, args.scheduler, args.loss_func)
        self.wrapper.init_device(args.use_gpu, args.gpu_ids)
        self.wrapper.init_optim(args.learning_rate, args.momentum, args.weight_decay, args.nesterov)

        total_epoch = args.local_epoch * args.federal_round * args.active_workers \
            if args.federal else args.local_epoch
        self.wrapper.init_scheduler_loss(args.step_size, args.gamma, total_epoch, args.warm_steps, args.min_lr)
        # Wrapper init

        # Pruning init
        if prune:
            self.prune_ext = HRank(self.wrapper)
            self.hyper_inter = IntervalProvider()
            self.hyper_rate = RateProvider(args.prune_rate, args.federal_round, args.check_inter)
        # Pruning init

        if args.pre_train:
            self.wrapper.load_checkpoint(file_repo.model_path)

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

    def run_model(self, train: bool = False,
                  pre_params: Iterator = None,
                  batch_limit: int = 0,
                  loss_weight: torch.Tensor = None) -> int:
        sum_loss = 0.0
        self.latest_feed_amount = 0

        if train:
            for i in range(args.local_epoch):
                global_logger.info(f"******The current train epoch: {self.train_epoch + i}******")
                if batch_limit == 0:
                    cort, total, loss = self.wrapper.step_run(args.batch_limit, train, pre_params,
                                                              loss_weight=loss_weight)
                else:
                    cort, total, loss = self.wrapper.step_run(batch_limit, train, pre_params,
                                                              loss_weight=loss_weight)
                sum_loss += loss
                self.latest_feed_amount += total
                self.wrapper.show_lr()
            self.train_epoch += args.local_epoch
            return sum_loss / args.local_epoch
        else:
            assert batch_limit != 0, self.ERROR_MESS1
            cort, total, loss = self.wrapper.step_run(batch_limit, train=False)
            global_container.flash(f'{args.exp_name}-test_acc', cort / total * 100)
            return loss

    # 获取模型参数的最新梯度
    def get_latest_grads(self) -> list:
        grads = []
        layers = self.prune_ext.flow_layers_params
        for layer in layers:
            grads.append(layer.grad)
        return grads

    # 测试模型性能
    def test_performance(self):
        self.wrapper.valid_performance(self.test_dataloader)

    # 调整模型学习率
    def decay_lr(self, epoch: int):
        self.wrapper.adjust_lr(math.pow(args.gamma, epoch))

    # 查看学习率
    def show_lr(self):
        self.wrapper.show_lr()

    def prune_process(self, random: bool, grads: list = None):
        # 前向剪枝信息度量计算
        path_id = self.prune_ext.get_rank(random=random)
        # args.rank_norm_path = file_repo.fetch_path(path_id)

        # 后向剪枝信息度量计算，并附加到前向度量上
        if args.curt_mode == VState.CALIMFL:
            path_id = self.prune_ext.rank_plus(info_norm=args.info_norm, backward=args.backward,
                                               grads=grads)
            # args.rank_plus_path = file_repo.fetch_path(path_id)

        self.prune_ext.mask_prune(args.prune_rate)

        if args.curt_mode == VState.Single:
            self.prune_ext.warm_up(wu_epoch, wu_batch)

    # 依赖于模型参数的自适应时机剪枝 to modify
    def prune_model(self, grads: list = None, random: bool = False, auto_inter: bool = False):
        if auto_inter:
            self.prune_ext.get_rank_simp(random=random)
            self.hyper_inter.push_container(deepcopy(self.prune_ext.rank_list))
            if self.hyper_inter.is_timing():
                global_logger.info(f"Will prune in this round.")
                self.prune_process(grads=grads, random=random)
            else:
                global_logger.info(f"Do not prune in this round.")
        else:
            global_logger.info(f"Will prune in this round.")
            self.prune_process(grads=grads, random=random)

    def structure_prune_process(self, grads: list = None, random: bool = False) -> nn.Module:
        self.prune_ext.get_rank(random=random)
        if args.curt_mode == VState.CALIMFL:
            self.prune_ext.rank_plus(info_norm=args.info_norm, backward=args.backward,
                                     grads=grads)
        return self.prune_ext.structure_prune(args.prune_rate)

    # 程序退出时保存关键度量指标、配置信息和模型参数
    def exit_proc(self, check: bool = False, one_key: str = None):
        if check:
            self.exit_manager.checkpoint_freeze()
        self.exit_manager.config_freeze()
        self.exit_manager.running_freeze(one_key)
