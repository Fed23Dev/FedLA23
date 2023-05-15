from dl.SingleCell import SingleCell
from dl.compress.compress_util import dict_coo_express
from env.running_env import args
from env.static_env import vgg16_candidate_rate
from utils.objectIO import pickle_mkdir_save


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
    coo_size()
