import torch
import torch.nn as nn

from env.running_env import args
from env.support_config import VModel


def is_pruned(module: nn.Module) -> bool:
    if isinstance(module, nn.BatchNorm2d) or isinstance(module, nn.BatchNorm1d):
        return False
    else:
        return True


def traverse_module(module, criterion, layers: list, names: list, prefix="", leaf_only=True):
    if leaf_only:
        for key, submodule in module._modules.items():
            new_prefix = prefix
            if prefix != "":
                new_prefix += '.'
            new_prefix += key
            # is leaf and satisfies criterion
            if len(submodule._modules.keys()) == 0 and criterion(submodule):
                layers.append(submodule)
                names.append(new_prefix)
            traverse_module(submodule, criterion, layers, names, prefix=new_prefix, leaf_only=leaf_only)
    else:
        raise NotImplementedError("Supports only leaf modules")


class Extender:
    DICT_KEY1 = "layers"
    DICT_KEY2 = "layers_prefixes"
    DICT_KEY3 = "relu_layers"
    DICT_KEY4 = "relu_layers_prefixes"
    DICT_KEY5 = "prune_layers"
    DICT_KEY6 = "prune_layers_prefixes"

    def __init__(self, model: nn.Module):
        self.model = model
        self.masks = torch.tensor(0.)
        self.groups = []

    def collect_layers_params(self) -> dict:
        layers = []
        layers_prefixes = []
        relu_layers = [m for (k, m) in self.model.named_modules() if isinstance(m, nn.ReLU)]
        relu_layers_prefixes = [k for (k, m) in self.model.named_modules() if isinstance(m, nn.ReLU)]

        traverse_module(self.model, lambda x: len(list(x.parameters())) != 0, layers, layers_prefixes)

        prune_indices = [ly_id for ly_id, layer in enumerate(layers) if is_pruned(layer)]
        prune_layers = [layers[ly_id] for ly_id in prune_indices]
        prune_layers_prefixes = [layers_prefixes[ly_id] for ly_id in prune_indices]

        ret = {
            self.DICT_KEY1: layers,
            self.DICT_KEY2: layers_prefixes,
            self.DICT_KEY3: relu_layers,
            self.DICT_KEY4: relu_layers_prefixes,
            self.DICT_KEY5: prune_layers,
            self.DICT_KEY6: prune_layers_prefixes
        }
        return ret

    # feature_map_layers() return a list, the length of it is n.
    # flow_layers_parameters() return a list, the length of it must be n. Neglect the first conv layer params.
    # prune_layers() return a list, the length of it must be n+1 or 2n+2.
    # prune_layer_parameters() is same as prune_layers().
    # why + 1? A: we do not prune the last conv layer
    # why 2n? A: Every convolution layer has a BN layer behind it, BN layers also need to be pruned
    # n+1 show that model does not contain batch-norm layer
    # 2n+2 show that model contains batch-norm layer

    # conv pre one layer
    def feature_map_layers(self) -> list:
        if args.model == VModel.MobileNetV2:
            return self.mv2_layer()

        layers = []
        pre_module = None
        for module in self.model.modules():
            if isinstance(module, nn.Conv2d) and pre_module is not None:
                layers.append(pre_module)
            if len(list(module.modules())) == 1 and not isinstance(module, nn.Sequential):
                pre_module = module

        # model only has one single conv-2d layer
        if not layers:
            for module in self.model.modules():
                if isinstance(module, nn.ReLU6) or isinstance(module, nn.ReLU):
                    layers.append(module)
        return layers

    # info_flow layer for rank_plus
    # 不只是获取权重
    # 还要获取该权重对应的梯度
    def flow_layers_parameters(self) -> list:
        if args.model == VModel.MobileNetV2:
            return self.mv2_flow_layer_params()

        layer_parameters = []
        first = True
        for module in self.model.modules():
            if isinstance(module, nn.Conv2d):
                if first:
                    first = False
                    continue
                else:
                    for name, params in module.named_parameters():
                        if name == 'weight':
                            layer_parameters.append(params)

        # model only has one single conv-2d layer
        if not layer_parameters:
            for module in self.model.modules():
                if isinstance(module, nn.Conv2d):
                    for name, params in module.named_parameters():
                        if name == 'weight':
                            layer_parameters.append(params)

        self.groups = [1 for _ in range(len(layer_parameters))]
        return layer_parameters

    # conv & BN
    def prune_layers(self) -> list:
        layers = []
        for module in self.model.modules():
            if isinstance(module, nn.Conv2d):
                layers.append(module)
            if isinstance(module, nn.BatchNorm2d):
                layers.append(module)
        return layers

    # conv & BN parameter: torch.Tensor
    def prune_layer_parameters(self) -> list:
        layers = self.prune_layers()
        layer_parameters = []
        for layer in layers:
            for name, params in layer.named_parameters():
                if name == 'weight':
                    layer_parameters.append(params)
        return layer_parameters

    def mask_compute(self):
        pass

    # 卷积层输出经过BN和ReLU的特征图层，后一个卷积层前
    # [3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29,
    #                  30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51]
    def mv2_layer(self) -> list:
        layers = [self.model.features[0]]
        for i in range(1, 19):
            if i == 1:
                block = self.model.features[i].conv
                relu_list = [2, 4]
            elif i == 18:
                block = self.model.features[i]
                relu_list = [2]
            else:
                block = self.model.features[i].conv
                relu_list = [2, 5, 7]
            for j in relu_list:
                cov_layer = block[j]
                layers.append(cov_layer)
        return layers[2:]

    # 卷积层后一个卷积层
    def mv2_flow_layer_params(self) -> list:
        self.groups.append(self.model.features[1].conv[0].groups)
        params = [self.model.features[1].conv[0].weight]

        for i in range(1, 18):
            if i == 1:
                layer1 = self.model.features[i].conv[3]
                layer2 = self.model.features[i+1].conv[0]
                layers = [layer1, layer2]
            else:
                layer1 = self.model.features[i].conv[3]
                layer2 = self.model.features[i].conv[6]
                if i == 17:
                    # 最后一个卷积层
                    layer3 = self.model.features[i+1][0]
                else:
                    layer3 = self.model.features[i+1].conv[0]
                layers = [layer1, layer2, layer3]
            for cov_layer in layers:
                self.groups.append(cov_layer.groups)
                params.append(cov_layer.weight)

        self.groups.append(1)
        params.append(self.model.classifier[1].weight)

        del self.groups[0]
        del self.groups[1]
        return params[2:]

    def flow_layer_groups(self) -> list:
        return self.groups
