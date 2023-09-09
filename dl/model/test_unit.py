import torch

from dl.SingleCell import SingleCell
from dl.model.ModelExt import Extender
from dl.model.model_util import create_model
from env.static_env import ORIGIN_CP_RATE
from env.support_config import VModel
from torchsummary import summary


def test_model():
    model = create_model(VModel.ResNet56)
    relucfg = [2, 6, 9, 13, 16, 19, 23, 26, 29, 33, 36, 39]
    convcfg = [(3 * i + 2) for i in range(9 * 3 * 2 + 1)]

    convcfg110 = [(3 * i + 2) for i in range(18 * 3 * 2 + 1)]
    # (cov_id - 1) * 4
    params = model.named_parameters()
    cnt = 0
    for name, item in params:
        print(f"{name}:{item.size()}")
        cnt += 1
    cnt = 0
    params = model.named_parameters()
    for name, item in params:
        if cnt in convcfg:
            print(f"---{name}:{item.size()}")
        cnt += 1

    mods = model.named_modules()
    for name, item in mods:
        print(f"+++{name}:")
    print(cnt)

    for id in convcfg:
        print(model.features[id])


def test_sub_model():
    model = create_model(VModel.ResNet56)
    ext = Extender(model)
    layers = ext.prune_layers()
    cov_idx = [2, 6, 9, 13, 16, 19, 23, 26, 29, 33, 36, 39, 42]
    for idx, layer in zip(cov_idx, layers):
        cov_layer = model.features[idx]
        print(cov_layer is layer)


def test_pre_model():
    cell = SingleCell(prune=True)
    cell.test_performance()


def test_shunet():
    model = create_model(VModel.ShuffleNetV2, num_classes=10, in_channels=3)

    data = torch.randn(32, 3, 32, 32)
    out = model(data)
    print(out.size())

    summary(model, (3, 32, 32))


def model_forward():
    data = torch.randn(32, 3, 32, 32)

    model = create_model(VModel.Conv2, ORIGIN_CP_RATE, 10)
    model = create_model(VModel.VGG16, ORIGIN_CP_RATE, 10)

    out = model(data)
    print(out)
