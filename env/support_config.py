from enum import Enum, unique


# dataset type
@unique
class VDataSet(Enum):
    CIFAR10 = "cifar10"
    CIFAR100 = "cifar100"
    FMNIST = "fmnist"
    TinyImageNet = "tinyimagenet"
    EMNIST = "emnist"


# Model Type
@unique
class VModel(Enum):
    Init = 0
    VGG11 = 1
    VGG16 = 2
    ResNet8 = 3
    ResNet56 = 4
    ResNet110 = 5
    MobileNetV2 = 6
    Conv2 = 7
    ShuffleNetV2 = 8


# Optimizer Type
@unique
class VOptimizer(Enum):
    Init = 0
    SGD = 1
    SGD_PFL = 2
    ADAM = 3
    RMSprop = 4


# Optimizer Type
@unique
class VScheduler(Enum):
    Init = 0
    StepLR = 1
    CosineAnnealingLR = 2
    WarmUPCosineLR = 3
    ReduceLROnPlateau = 4
    WarmUPStepLR = 5


# loss func
@unique
class VLossFunc(Enum):
    Init = 0
    Cross_Entropy = 1


# running statue
@unique
class VState(Enum):
    Init = 0
    FedAvg = 1
    FedProx = 2
    SCAFFOLD = 3
    FedDAS = 4
    MOON = 5
    CriticalFL = 6
    IFCA = 7
    Single = 8

@unique
class VNonIID(Enum):
    IID = "iid"
    Hetero = "hetero"
    Shards = "shards"
