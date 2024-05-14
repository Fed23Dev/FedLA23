from random import random

import torch

from utils.Cleaner import FileCleaner
from env.running_env import args, global_file_repo, global_container
from utils.MathTools import kl_divergence, js_divergence, _kl_divergence
from utils.AccExtractor import AccExtractor

def random_list(length=100):
    random_int_list = []
    for i in range(length):
        random_int_list.append(random.randint(0, 10))
    return random_int_list


def get_lists():
    lists = [[range(100), random_list(), random_list()]]
    return lists


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


def kl_and_js():
    a = torch.tensor([1, 2, 3, 4, 5]).float()
    b = torch.tensor([5, 4, 3, 2, 1]).float()
    print(f"KL:{kl_divergence(a, b)}")
    print(f"JS:{js_divergence(a, b)}")
    print(f"JS:{js_divergence(b, a)}")
    print(f"JS:{js_divergence(a, a)}")

    a = torch.tensor([0.5, 0.5]).float()
    b = torch.tensor([1., 0.]).float()
    c = torch.tensor([0., 1.]).float()
    print(f"KL:{kl_divergence(b, a)}")
    print(f"KL:{kl_divergence(b, c)}")


def extract():
    path = "logs/super"
    extractor = AccExtractor(path)
    extractor.extract_acc_data()
    extractor.show_detail_rets()
    extractor.show_avg_rets()

def main():
    extract()

