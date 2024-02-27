from ruamel.yaml import YAML
from copy import deepcopy

from env.args_request import DEFAULT_ARGS
from env.support_config import *


def model_str2enum(value: str) -> VModel:
    if value == 'vgg16':
        ret = VModel.VGG16
    elif value == 'resnet56':
        ret = VModel.ResNet56
    elif value == 'resnet110':
        ret = VModel.ResNet110
    elif value == 'mobilenetV2':
        ret = VModel.MobileNetV2
    elif value == 'conv2':
        ret = VModel.Conv2
    elif value == 'shufflenetV2':
        ret = VModel.ShuffleNetV2
    else:
        ret = VModel.Init
    return ret


def dataset_str2enum(value: str) -> VDataSet:
    if value == 'cifar10':
        ret = VDataSet.CIFAR10
    elif value == 'cifar100':
        ret = VDataSet.CIFAR100
    elif value == 'tinyimagenet':
        ret = VDataSet.TinyImageNet
    elif value == 'fmnist':
        ret = VDataSet.FMNIST
    else:
        ret = VDataSet.Init
    return ret


def optim_str2enum(value: str) -> VOptimizer:
    if value == 'sgd':
        ret = VOptimizer.SGD
    elif value == 'sgd_pfl':
        ret = VOptimizer.SGD_PFL
    elif value == 'adam':
        ret = VOptimizer.ADAM
    else:
        ret = VOptimizer.Init
    return ret


def scheduler_str2enum(value: str) -> VScheduler:
    if value == 'step_lr':
        ret = VScheduler.StepLR
    elif value == 'cosine_lr':
        ret = VScheduler.CosineAnnealingLR
    elif value == 'warmup_cos_lr':
        ret = VScheduler.WarmUPCosineLR
    elif value == 'reduce_lr':
        ret = VScheduler.ReduceLROnPlateau
    elif value == 'warmup_step_lr':
        ret = VScheduler.WarmUPStepLR
    else:
        ret = VScheduler.Init
    return ret


def loss_str2enum(value: str) -> VLossFunc:
    if value == 'cross_entropy':
        ret = VLossFunc.Cross_Entropy
    else:
        ret = VLossFunc.UPPER
    return ret


def alg_str2enum(value: str) -> VState:
    if value == 'fedavg':
        ret = VState.FedAvg
    elif value == 'fedprox':
        ret = VState.FedProx
    elif value == 'fedla':
        ret = VState.FedLA
    elif value == 'scaffold':
        ret = VState.SCAFFOLD
    elif value == 'moon':
        ret = VState.MOON
    elif value == 'criticalfl':
        ret = VState.CriticalFL
    elif value == 'ifca':
        ret = VState.IFCA
    elif value == 'single':
        ret = VState.Single
    else:
        ret = VState.Init
    return ret


class ArgRepo:
    ERROR_MESS1 = "The yaml file lacks necessary parameters."

    def __init__(self, yml_path: str):
        self.r_yaml = YAML(typ="safe")
        self.yml_path = yml_path
        self.init_attr_placeholder()
        self.runtime_attr_placeholder()

    def init_attr_placeholder(self):
        self.exp_name = None
        self.model = None
        self.pre_train = None
        self.use_gpu = None
        self.gpu_ids = None
        self.dataset = None
        self.batch_size = None
        self.optim = None
        self.nesterov = None
        self.learning_rate = None
        self.min_lr = None
        self.momentum = None
        self.weight_decay = None
        self.loss_func = None
        self.scheduler = None
        self.step_size = None
        self.gamma = None
        self.warm_steps = None
        self.federal = None
        self.non_iid = None
        self.workers = None
        self.active_workers = None
        self.federal_round = None
        self.local_epoch = None
        self.batch_limit = None
        self.logits_batch_limit = None
        self.loss_back = None
        self.test_batch_limit = None

        self.alg = None

        # Temporarily deprecated
        self.CE_WEIGHT = None
        self.ALPHA = None
        self.BETA = None
        self.T = None
        self.WARMUP = None
        self.KD_BATCH = None
        self.KD_EPOCH = None

        self.clusters = None
        self.drag = None
        self.threshold = None

        self.mu = None
        self.T = None


    def runtime_attr_placeholder(self):
        self.num_classes = None
        self.in_channels = None
        self.curt_mode = None

    @property
    def exp_name(self) -> str:
        return f"{self._exp_name}-{self.alg}"

    def activate(self, strict: bool = False):
        options = self.parse_args()
        if strict:
            assert self.is_legal(options), self.ERROR_MESS1
        self.mount_args(options)

    def parse_args(self) -> dict:
        with open(self.yml_path, 'r') as f:
            args = deepcopy(DEFAULT_ARGS)
            args.update(dict(self.r_yaml.load(f)))
        return args

    def is_legal(self, options: dict) -> bool:
        pass

    def mount_args(self, options: dict):
        for k, v in options.items():
            if k == 'model':
                setattr(self, k, model_str2enum(v))
            elif k == 'dataset':
                setattr(self, k, dataset_str2enum(v))
            elif k == 'optim':
                setattr(self, k, optim_str2enum(v))
            elif k == 'scheduler':
                setattr(self, k, scheduler_str2enum(v))
            elif k == 'loss_func':
                setattr(self, k, loss_str2enum(v))
            elif k == 'alg':
                setattr(self, 'curt_mode', alg_str2enum(v))
                setattr(self, 'alg', v)
            else:
                setattr(self, k, v)

        self.supplement_args()

    def supplement_args(self):
        if self.dataset == VDataSet.CIFAR10:
            self.num_classes = 10
            self.in_channels = 3
        elif self.dataset == VDataSet.CIFAR100:
            self.num_classes = 100
            self.in_channels = 3
        elif self.dataset == VDataSet.TinyImageNet:
            self.num_classes = 200
            self.in_channels = 3
        elif self.dataset == VDataSet.FMNIST:
            self.num_classes = 10
            self.in_channels = 1
        else:
            print("The dataset is not supported.")
            exit(1)

    # call after mount_args()
    def get_snapshot(self) -> str:
        optim = str(self.optim).split('.')[1]
        scheduler = str(self.optim).split('.')[1]

        return f"optim:{optim}\n" \
               f"learning rate:{self.learning_rate}\n" \
               f"scheduler:{scheduler}\n" \
               f"warm steps:{self.warm_steps}\n" \
               f"non_iid:{self.non_iid}\n" \
               f"workers:{self.workers}\n" \
               f"active_workers:{self.active_workers}\n"\
               f"federal_round:{self.federal_round}\n" \
               f"local epoch:{self.local_epoch}\n" \
               f"drag:{self.drag}"

    @exp_name.setter
    def exp_name(self, value):
        self._exp_name = value
