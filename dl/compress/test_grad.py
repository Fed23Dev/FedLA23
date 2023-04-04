import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict

from torchsummary import summary


class ToyModel(nn.Module):
    def __init__(self, in_channels, out_channels, num_classes=2):
        super().__init__()
        # tmp only for testing, not valid
        self.tmp = nn.Conv2d(in_channels, in_channels * 2, (3, 3))
        self.dim = out_channels
        self.conv1 = nn.Conv2d(in_channels=in_channels,
                               out_channels=in_channels * 2,
                               kernel_size=(3, 3),
                               stride=(1, 1),
                               padding=0)
        self.conv2 = nn.Conv2d(in_channels=in_channels * 2,
                               out_channels=out_channels,
                               kernel_size=(3, 3),
                               stride=(1, 1),
                               padding=0)
        self.pool = nn.AdaptiveAvgPool2d(output_size=(1))
        self.fc = nn.Linear(out_channels, num_classes, bias=False)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        x = self.fc(x.view(-1, self.dim))
        return x


def get_model_norm_gradient(model):
    """
    Description:
        - get norm gradients from model, and store in a OrderDict

    Args:
        - model: (torch.nn.Module), torch model

    Returns:
        - grads in OrderDict
    """
    grads = OrderedDict()
    for name, params in model.named_parameters():
        grad = params.grad
        if grad is not None:
            grads[name] = grad
    return grads


def cat_dict_shape(dic):
    for k, v in dic.items():
        print(f"Name:{k}, shape: {v.size()}")
        print(v)
        print(v ** 2)
        print(v ** 2)


def test_process():
    torch.manual_seed(0)
    num_data = 40
    data = torch.randn(num_data, 3, 224, 224)
    label = torch.randint(0, 2, (num_data,))

    toy_model = ToyModel(3, 64, 2)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(toy_model.parameters(), lr=1e-3)
    toy_model.train()

    summary(toy_model, input_size=(3, 224, 224), batch_size=-1, device="cpu")

    for i, data in enumerate(data):
        data = data.unsqueeze(0)
        out = toy_model(data)
        target = label[i].unsqueeze(0)
        loss = criterion(out, target)
        loss.backward()
        if (i + 1) % 10 == 0:
            print('=' * 80)
            cat_dict_shape(get_model_norm_gradient(toy_model))
            optimizer.step()
            optimizer.zero_grad()
            break

