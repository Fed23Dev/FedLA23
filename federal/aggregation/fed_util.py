from typing import List
from dl.compress.VHRank import HRank
from env.running_env import global_logger
from dl.data.dataProvider import DataLoader


def get_speech_right(device_cnt: int) -> List[float]:
    rights = [1 for _ in range(device_cnt)]
    sum_right = sum(rights)
    return [right / sum_right for right in rights]


def round_train():
    pass


def fed_test_performance(text: str, hrank: HRank, loader: DataLoader, is_save: bool = False) -> float:
    global_logger.info(text)
    hrank.wrapper.valid_performance(loader)
    if is_save:
        hrank.interrupt_disk('union.snap')
    return 0.0
