import random
from copy import deepcopy

import torch
from thop import profile

from dl.SingleCell import SingleCell
from dl.data.dataProvider import get_data_loader
from dl.model.ModelExt import Extender
from dl.model.model_util import create_model, dict_diff
from env.static_env import *
from env.running_env import *

from dl.compress.Sparse import TopKSparse
from dl.compress.Quantization import QuantizationSGD
import dl.compress.compress_util as util
from env.support_config import VDataSet
from utils.objectIO import pickle_mkdir_save, str_save


def mask_gen():
    model = create_model(VModel.VGG16)
    ext = Extender(model)
    prune = ext.prune_layers()
    params = ext.prune_layer_parameters()
    fm = ext.feature_map_layers()
    for pa in params:
        print(pa.size())


def sparse_and_quan():
    tensor = torch.randn(2, 2, 2, 2)
    sparser = TopKSparse(0.5)
    quaner = QuantizationSGD(16)

    print("origin tensor:", util.get_size(tensor))
    ct = quaner.compress(tensor)["com_tensor"]

    print(util.get_size(ct))


def sparse_optim():
    dic = dict()
    ones = torch.ones(64, 3, 3, 3)
    zero_indices = [0, 3, 4, 32, 56, 16, 46, 33, 9, 10, 12]
    for index in zero_indices:
        ones[index:index + 1] *= 0
    dic['conv.weight'] = ones
    coo_dict = util.dict_coo_express(dic)
    pickle_mkdir_save(dic, 'ori')
    pickle_mkdir_save(coo_dict, 'coo')


def test_self_flops():
    inputs = torch.randn(32, 3, 56, 56)
    net = create_model(VModel.VGG16)
    flops, params = profile(net, inputs=(inputs,))
    print(f"ORI-FLOPs:{flops}, params:{params}")

    net1 = deepcopy(net)
    net_params = net1.named_parameters()

    for k, v in net_params:
        if k.find('weight') != -1 and k.find('conv') != -1:
            f, c, w, h = v.size()
            zeros = torch.zeros(f, 1, 1, 1)

            all_ind = list(range(f))
            ind = random.sample(all_ind, len(all_ind)//2)

            for i in range(len(ind)):
                zeros[ind[i], 0, 0, 0] = 1.

            v.data = v.data * zeros

    flops, params = profile(net, inputs=(inputs,))

    dic1 = net.state_dict()
    dic2 = net1.state_dict()
    print(f"ORI-FLOPs:{flops}, params:{params}")


def dkd(alpha: float, beta: float, temperature: int,
        batch_size: int, num_classes: int):
    from dl.compress.DKD import _dkd_loss

    logits_tea = torch.randn(batch_size, num_classes)
    logits_stu = torch.randn(batch_size, num_classes)
    target = torch.randint(0, num_classes, (batch_size, 1))

    loss = _dkd_loss(logits_stu, logits_tea, target, alpha, beta, temperature)
    print(loss)


def main():
    dkd(0.1, 0.1, 1, 64, 10)
