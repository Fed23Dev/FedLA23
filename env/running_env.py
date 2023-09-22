from env import yaml2args
from env.support_config import VModel
from utils.PathManager import HRankPathManager
from utils.VContainer import VContainer
from utils.Vlogger import VLogger
from custom_path import *


# Args
args = yaml2args.ArgRepo(test_config)
args.activate()

### 
# Pre Default Env
###

# base static path
milestone_base = r'res/milestone'
image_base = r'res/images'
exp_base = r'res/exp'
log_base = r'logs'

if args.model == VModel.VGG16:
    model_path = vgg16_model
elif args.model == VModel.ResNet56:
    model_path = resnet56_model
elif args.model == VModel.ResNet110:
    model_path = resnet110_model
elif args.model == VModel.MobileNetV2:
    model_path = mobilenetv2_model
elif args.model == VModel.Conv2:
    model_path = conv2_model
elif args.model == VModel.ShuffleNetV2:
    model_path = shufflenetv2_model
else:
    model_path = vgg16_model
    print('Not supported model type.')
    exit(1)

### 
# Dynamic Env
###

# Path
file_repo = HRankPathManager(model_path, datasets_base)
file_repo.derive_path(exp_base, image_base, milestone_base, log_base)


# Logger
# global_logger_PATH = "logs/hrankFL.log"
logger_path, _ = file_repo.new_log()
global_logger = VLogger(logger_path, True).logger

# Container
global_container = VContainer()
