from enum import Enum, unique


# dataset type
@unique
class VDataSet(Enum):
    Init = 0
    CIFAR10 = 1
    CIFAR100 = 2
    FMNIST = 3
    TinyImageNet = 4


# Model Type
@unique
class VModel(Enum):
    Init = 0
    VGG11 = 1
    VGG16 = 2
    ResNet56 = 3
    ResNet110 = 4
    MobileNetV2 = 5
    Conv2 = 6


# Optimizer Type
@unique
class VOptimizer(Enum):
    Init = 0
    SGD = 1
    SGD_PFL = 2
    ADAM = 3


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
    FedLA = 4
    MOON = 5
    Single = 6
