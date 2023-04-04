import random
from copy import deepcopy

import torch
from thop import profile

from dl.SingleCell import SingleCell
from dl.compress.HyperProvider import RateProvider
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


def test_vrank():
    test_loader = get_data_loader(VDataSet.CIFAR10, data_type="test", batch_size=args.batch_size,
                                  shuffle=True, num_workers=0, pin_memory=False)
    master_cell = SingleCell(test_loader, True)
    master_cell.prune_ext.get_rank(random=True)
    master_cell.prune_ext.rank_plus(info_norm=args.info_norm, backward=args.backward)


def hyper_cosine():
    test_loader = get_data_loader(VDataSet.CIFAR10, data_type="test", batch_size=args.batch_size,
                                  shuffle=True, num_workers=0, pin_memory=False)
    master_cell = SingleCell(test_loader, True)
    for i in range(100):
        master_cell.run_model(train=True)
        master_cell.prune_model(plus=False, random=False, auto_inter=True)
    str_save(str(global_container['cos']), 'cos_test.ori')
    pickle_mkdir_save(global_container['cos'], 'cos_test.pickle')


def coo_recover():
    test_loader = get_data_loader(args.dataset, data_type="test", batch_size=args.batch_size,
                                  shuffle=True, num_workers=0, pin_memory=False)
    args.federal_round = 1
    args.check_inter = 1
    master_cell = SingleCell(test_loader, True)
    master_cell.prune_ext.get_rank()
    master_cell.prune_ext.mask_prune(resnet110_candidate_rate)

    model_dict = master_cell.access_model().cpu().state_dict()

    coo_dict = util.dict_coo_express(model_dict)
    recover_dict = util.dict_coo_recover(coo_dict)

    print(f"COO:{dict_diff(model_dict, recover_dict)}")

    pickle_mkdir_save(model_dict, 'ori')
    pickle_mkdir_save(coo_dict, 'coo')


def test_progress_coo():
    args.federal_round = 150
    args.check_inter = 10
    files_size = []
    test_loader = get_data_loader(args.dataset, data_type="test", batch_size=args.batch_size,
                                  shuffle=True, num_workers=0, pin_memory=False)
    master_cell = SingleCell(test_loader, True)
    hyper_rate = RateProvider(args.prune_rate, args.federal_round, args.check_inter)
    coo_path = r'coo'

    for i in range(150//10):
        master_cell.prune_ext.get_rank()
        rate = hyper_rate.get_curt_rate()
        master_cell.prune_ext.mask_prune(rate)
        model_dict = master_cell.access_model().cpu().state_dict()
        coo_dict = util.dict_coo_express(model_dict)
        pickle_mkdir_save(coo_dict, coo_path)
        files_size.append(os.stat(coo_path).st_size / (1024 * 1024))
        master_cell.access_model().cuda()

    path, _ = file_repo.new_seq('vgg16_file_size')
    pickle_mkdir_save(files_size, path)


def test_flops():
    test_loader = get_data_loader(VDataSet.CIFAR10, data_type="test", batch_size=args.batch_size,
                                  shuffle=True, num_workers=0, pin_memory=False)
    args.federal_round = 1
    args.check_inter = 1
    master_cell = SingleCell(test_loader, True)
    master_cell.test_performance()

    master_cell.prune_ext.get_rank()
    master_cell.prune_ext.mask_prune(vgg16_candidate_rate)
    master_cell.test_performance()


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


def main():
    test_progress_coo()
