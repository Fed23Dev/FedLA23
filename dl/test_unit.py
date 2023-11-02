import torch

from dl.SingleCell import SingleCell
from dl.compress.compress_util import dict_coo_express
from dl.data.dataProvider import get_data_loader
from dl.model.model_util import create_model
from env.running_env import args
from env.support_config import VModel
from utils.objectIO import pickle_mkdir_save


# def test_logits():
#     test_loader = get_data_loader(args.dataset, data_type="test", batch_size=args.batch_size,
#                                   shuffle=True, num_workers=0, pin_memory=False)
#     model = create_model(VModel.VGG16, num_classes=args.num_classes)
#     label_dtype = torch.int64
#
#     sum_logits = torch.zeros(args.num_classes, args.num_classes)
#     sum_labels = torch.zeros(args.num_classes, dtype=label_dtype)
#
#     for batch_idx, (inputs, targets) in enumerate(test_loader):
#         if batch_idx > 1:
#             break
#         labels = torch.argmax(targets, -1)
#         _labels, _cnt = torch.unique(labels, return_counts=True)
#         labels_cnt = torch.zeros(args.num_classes, dtype=label_dtype)\
#             .scatter_(dim=0, index=_labels, src=_cnt)
#
#         logits = model(inputs)
#
#         if batch_idx == 0:
#             sum_logits = sum_logits.type(logits.dtype)
#
#         # 扩展的标签索引 [0, 1] >> [[0, 0], [1, 1]]
#         logits_index = labels.unsqueeze(1).expand(logits.size())
#         # 自然数索引
#         labels_index = torch.tensor(list(range(args.num_classes)))
#
#         sum_logits.scatter_add_(dim=0, index=logits_index, src=logits)
#         sum_labels.scatter_add_(dim=0, index=labels_index, src=labels_cnt)
#
#     avg_logits = sum_logits / sum_labels
#     print(avg_logits)


def test_center_train():
    cell = SingleCell()
    cell.run_model(train=True)
    cell.test_performance()
    cell.exit_proc(check=False)


def test_valid():
    cell = SingleCell(prune=True)
    cell.test_performance()


# vgg16 resnet56 - cifar10
def test_prune_model():
    cell = SingleCell(prune=True)
    cell.prune_model()
    cell.test_performance()
    cell.exit_proc()


def test_prune_model_plus():
    cell = SingleCell(prune=True)
    # cell.run_model(train=True)
    cell.prune_model(grads=cell.get_latest_grads())
    cell.test_performance()
    cell.exit_proc()

    # board = HRankBoard()
    # # board.simp_rank_img(args.rank_norm_path)
    # # board.simp_rank_img(args.rank_plus_path)


def test_prune_model_random():
    cell = SingleCell(prune=True)
    cell.prune_model(random=True)
    cell.test_performance()
    cell.exit_proc()


def test_prune_model_interval():
    pass


def init_interval_compare():
    pass


# vgg16 resnet56 resnet100 mobilenetV2 - cifar10 cifar100
def test_auto_prune():
    pass


def total_auto_line():
    pass


def hrank():
    pass


def coo_size():
    print(args.model)
    args.pre_train = False
    cell = SingleCell(prune=True)
    cell.prune_ext.get_rank()
    cell.prune_ext.mask_prune(vgg16_candidate_rate)

    m = cell.access_model()
    ori = m.state_dict()
    com = dict_coo_express(ori)
    pickle_mkdir_save(com, "com.m")
    pickle_mkdir_save(ori, "orim.m")


def main():
    test_logits()
