import math

from scipy import spatial
from sklearn.metrics.pairwise import pairwise_distances
from env.running_env import global_logger, global_container
from env.static_env import pruning_inter


class IntervalProvider:
    SIMP_LEN = 2

    def __init__(self, tiny_const: float = 1e-8):
        self.tiny_limit = tiny_const
        self.cont_list = []

    def is_timing(self, ranks_list: list = None):
        if ranks_list is None:
            ranks_list = self.cont_list
        if len(ranks_list) == self.SIMP_LEN:
            cos_distance = spatial.distance.cosine(ranks_list[0][0], ranks_list[1][0])
            # cos_distance = pairwise_distances(ranks_list[0], ranks_list[1])
            global_container.flash('cos', cos_distance)
            if cos_distance < self.tiny_limit:
                return True
            else:
                return False
        else:
            global_logger.info(f"Length of cont_list must be 2.")
            return False

    def push_container(self, ranks: list):
        if len(self.cont_list) < 2:
            self.cont_list.append(ranks)
        else:
            self.cont_list[0] = self.cont_list[1]
            self.cont_list[1] = ranks


class RateProvider:
    # 三种get_*_rate()方法在整个训练周期中只能始终调用一个
    def __init__(self, pruning_rate: list, total_round: int, interval: int = 1):
        self.or_rates = pruning_rate
        self.rates = [0 for _ in range(len(pruning_rate))]
        self.rate_step = []
        if interval == 0:
            self.remain_steps = pruning_inter
            self.ratio = 1./pruning_inter
        else:
            self.remain_steps = total_round // interval
            self.ratio = interval / total_round

        for rate in pruning_rate:
            self.rate_step.append(rate * self.ratio)

    # pruning max
    def get_curt_rate(self) -> list:
        if self.remain_steps > 0:
            self.rates = [rate + step for rate, step in zip(self.rates, self.rate_step)]
            self.remain_steps -= 1
            return self.rates
        return self.or_rates

    # pruning mid
    def get_progress_rate(self):
        if self.remain_steps > 1:
            self.remain_steps -= 1
            return [math.pow(rate, self.remain_steps) for rate in self.or_rates]
        elif self.remain_steps == 1:
            return self.or_rates
        else:
            exit(1)

    # pruning min
    def get_same_rate(self):
        if self.remain_steps > 0:
            self.remain_steps -= 1
            return [math.pow(rate, self.ratio) for rate in self.or_rates]
        else:
            return self.rates
