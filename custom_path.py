###
# Default Env (refer)
###
import argparse
import os

# deprecated
exp1_config = r'share/cifar10-vgg16-final.yml'
exp2_config = r'share/cifar10-resnet56.yml'
exp3_config = r'share/cifar100-resnet110-final.yml'
exp4_config = r'share/cifar100-mobilenetV2.yml'

###
# Default Env (refer)
###

###
# Custom Env (to fill)
###

# entire path
datasets_base = r'~/la/datasets'

test_config = r'share/configs/default_config.yml'

vgg16_model = r'res/checkpoint/vgg/vgg_16_bn.pt'
resnet56_model = r'res/checkpoint/resnet/resnet_56.pt'
resnet110_model = r'res/checkpoint/resnet/ResNet110.snap'
mobilenetv2_model = r'res/checkpoint/mobilenet/MobileNetV2.snap'
conv2_model = r'res/checkpoint/conv2/conv2.snap'
shufflenetv2_model = r'res/checkpoint/shufflenet/ShufflenetV2.snap'

debug_config = r'share/configs/hyper-exps.yml'
# debug_config = r'share/configs/cifar10-vgg16-final.yml'


def auto_config(option: str):
    global test_config
    if option == 'e':
        test_config = debug_config
    elif option == 'e1':
        test_config = exp1_config
    elif option == 'e2':
        test_config = exp2_config
    elif option == 'e3':
        test_config = exp3_config
    elif option == 'e4':
        test_config = exp4_config
    else:
        if os.access(option, os.R_OK):
            test_config = option
        else:
            print('Can not access to config file.')
            exit(1)

###
# Custom Env (to fill)
###
