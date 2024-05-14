# 构建CNN模型
import torch
import torch.nn as nn


# only for 32*32
class Conv2(nn.Module):
    def __init__(self, compress_rate, in_channels=3, num_classes=10):
        super(Conv2, self).__init__()
        self.inner = int(128 * (1-compress_rate[0]))
        self.out = 64 * self.inner
        self.features = nn.Sequential(
            nn.Conv2d(in_channels, out_channels=self.inner,
                      kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(self.inner),
            nn.ReLU6(inplace=True)
        )

        self.fc = torch.nn.Linear(self.out, num_classes)

    def forward(self, x):
        batch_size = x.size(0)
        x = self.features(x)
        x = nn.AdaptiveAvgPool2d((8, 8))(x)
        x = x.view(batch_size, -1)
        x = self.fc(x)
        return x
