import torch.nn as nn
from collections import OrderedDict
from abc import ABC, abstractmethod

vgg11_cfg = [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M']
vgg16_cfg = [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512]
relucfg = [2, 6, 9, 13, 16, 19, 23, 26, 29, 33, 36, 39, 42]


# 卷积层的输入对应卷积核的通道，输出对应卷积核的个数，卷积层的weight参数为 f,c,w,h f为卷积核个数，c为卷积核的通道
# 钩子总是注册在当前卷积层后下一卷积层的前一层，从当前特征图决定args.exp_name值
# 最终删减的目标是卷积层的输出通道，连带网络其他参数，但只是加载权重，其他的重新训练

# 单次剪枝 + 剪枝时机 + rank强化

class VGG(nn.Module, ABC):
    def __init__(self, compress_rate, cfg=None, in_channels=3):
        super(VGG, self).__init__()
        if cfg is None:
            cfg = vgg16_cfg
        self.relucfg = relucfg
        self.compress_rate = compress_rate[:]
        self.compress_rate.append(0.0)
        self.features = self._make_layers(cfg, in_channels)

        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.constant_(m.bias, 0)
                # 也可以判断是否为conv2d，使用相应的初始化方式
            elif isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                # 是否为批归一化层
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layers(self, cfg, in_channels):
        layers = nn.Sequential()
        cnt = 0
        for i, x in enumerate(cfg):
            if x == 'M':
                layers.add_module('pool%d' % i, nn.MaxPool2d(kernel_size=2, stride=2))
            else:
                x = int(x * (1-self.compress_rate[cnt]))
                cnt += 1
                conv2d = nn.Conv2d(in_channels, x, kernel_size=3, padding=1)
                layers.add_module('conv%d' % i, conv2d)
                layers.add_module('norm%d' % i, nn.BatchNorm2d(x))
                layers.add_module('relu%d' % i, nn.ReLU(inplace=True))
                in_channels = x
        return layers

    @abstractmethod
    def forward(self, x):
        pass


class VGG11(VGG):
    def __init__(self, compress_rate, in_channels=3, num_classes=10):
        super().__init__(compress_rate, vgg11_cfg, in_channels)
        self.classifier = nn.Sequential(OrderedDict([
            ('linear1', nn.Linear(vgg11_cfg[-2], 512)),
            ('relu1', nn.ReLU(inplace=True)),
            ('linear2', nn.Linear(512, 512)),
            ('relu2', nn.ReLU(inplace=True)),
            ('linear3', nn.Linear(512, num_classes)),
        ]))

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x


class VGG16(VGG):
    def __init__(self, compress_rate, in_channels=3, num_classes=10):
        super().__init__(compress_rate, vgg16_cfg, in_channels)
        self.classifier = nn.Sequential(OrderedDict([
            ('linear1', nn.Linear(vgg16_cfg[-2], vgg16_cfg[-1])),
            ('dropout', nn.Dropout(p=0.5)),
            ('norm1', nn.BatchNorm1d(vgg16_cfg[-1])),
            ('relu1', nn.ReLU(inplace=True)),
            ('linear2', nn.Linear(vgg16_cfg[-1], num_classes)),
        ]))

    def forward(self, x):
        x = self.features(x)
        x = nn.AvgPool2d(2)(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x
