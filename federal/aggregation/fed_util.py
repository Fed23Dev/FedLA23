from typing import List


def get_speech_right(device_cnt: int) -> List[float]:
    rights = [1 for _ in range(device_cnt)]
    sum_right = sum(rights)
    return [right / sum_right for right in rights]


def round_train():
    pass
