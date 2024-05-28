from enum import Enum, unique

from utils.VContainer import VContainer
from utils.Vlogger import VLogger

# Uniform const
CPU = -6
GPU = -66
CPU_STR_LEN = 3

ORIGIN_CP_RATE = [0.] * 100

# simulation
MAX_ROUND = 10001
MAX_DEC_DIFF = 0.3
ADJ_INTERVAL = 50
ADJ_HALF_LIFE = 10000


# CIFAR10 const config
CIFAR10_NAME = "CIFAR10"
CIFAR10_CLASSES = 10
CIFAR10_NUM_TRAIN_DATA = 50000
CIFAR10_NUM_TEST_DATA = 10000
CIFAR10_MEAN = [0.4914, 0.4822, 0.4465]
CIFAR10_STD = [0.2023, 0.1994, 0.2010]



# CIFAR100 const config
CIFAR100_CLASSES = 100
CIFAR100_MEAN = [0.5070751592371323, 0.48654887331495095, 0.4409178433670343]
CIFAR100_STD = [0.2673342858792401, 0.2564384629170883, 0.27615047132568404]

# UCM const config
UCM_CLASSES = 21
UCM_MEAN = [0.485, 0.456, 0.406]
UCM_STD = [0.229, 0.224, 0.225]

# Tiny const config
TinyImageNet_CLASSES = 200
Train_Each_CLASS = 500
Test_Each_CLASS = 50
Val_Each_CLASS = 50
ImageNet_MEAN = [0.485, 0.456, 0.406]
ImageNet_STD = [0.229, 0.224, 0.225]

# FMNIST const config
FMNIST_CLASSES = 10

# EMNIST const config
EMNIST_CLASSES = 62

# VGG const config

# Others
MAX_HOOK_LAYER = 50
valid_limit = 99
rank_limit = 10

# Default_config
YAML_PATH = r'share/default_config.yml'

# Warm-up config
# wu_epoch = 50
# wu_batch = 32
wu_epoch = 1
wu_batch = 32

# acc_info
print_interval = 10



