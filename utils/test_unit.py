from random import random

from utils.Cleaner import FileCleaner
from utils.Visualizer import HRankBoard
from env.running_env import args, file_repo, global_container


def random_list(length=100):
    random_int_list = []
    for i in range(length):
        random_int_list.append(random.randint(0, 10))
    return random_int_list


def get_lists():
    lists = [[range(100), random_list(), random_list()]]
    return lists


def test_hrank_visual():
    sim = HRankBoard()
    sim.simp_acc_compare_img()


def test_rank_img():
    board = HRankBoard()
    board.simp_rank_img(args.rank_norm_path)
    board.simp_rank_img(args.rank_plus_path)


def cleaner_test():
    log_test = r'2022.08.02_11-04-23.log'
    file_test = r'Norm_Rank---07.30.npy'
    false_test = r'norm'

    cleaner = FileCleaner(7)
    date = cleaner.fetch_date(file_test)
    days = cleaner.day_consumed(date)
    print(f"days:{days}")
    date = cleaner.fetch_date(log_test)
    days = cleaner.day_consumed(date)
    print(f"days:{days}")


def res_and_log_clean():
    cleaner = FileCleaner(7)
    cleaner.clear_files()


def test_container():
    global_container.flash('test', 1)
    global_container.flash('test', 2)
    global_container.flash('test', 3)

    print(f"Test{global_container['test']}")
    print(f"=====")


def main():
    test_container()

