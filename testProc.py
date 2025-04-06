import torch
from torchsummary import summary

from dl.model.model_util import create_model
from env.support_config import VModel
from utils.OverheadCounter import OverheadCounter

ohc = OverheadCounter()

conv2 = create_model(VModel.Conv2, num_classes=10, in_channels=1).cuda()
summary(conv2, (1, 28, 28))

tensor = torch.rand(10, 10, dtype=torch.float32)
load = ohc.calculate_memory_size(tensor, 'MB')
print(f"fashsionmnist: {load}")

###############
resnet = create_model(VModel.ResNet110, num_classes=62, in_channels=1).cuda()
summary(resnet, (1, 28, 28))

tensor = torch.rand(62, 62, dtype=torch.float32)
load = ohc.calculate_memory_size(tensor, 'MB')
print(f"emnist: {load}")

############
mobilenet = create_model(VModel.MobileNetV2, num_classes=100, in_channels=3).cuda()
summary(mobilenet, (3, 32, 32))

tensor = torch.rand(100, 100, dtype=torch.float32)
load = ohc.calculate_memory_size(tensor, 'MB')
print(f"cifar100: {load}")