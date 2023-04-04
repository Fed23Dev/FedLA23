from torch.utils.data import Dataset
import math
import numpy as np


def KL_divergence(p1, p2):
    d = 0
    for i in range(len(p1)):
        if p2[i] == 0 or p1[i] == 0:
            continue
        d += p1[i] * math.log(p1[i] / p2[i], 2)
    return d


def JS_divergence(p1, p2):
    p3 = []
    for i in range(len(p1)):
        p3.append((p1[i] + p2[i]) / 2)
    return KL_divergence(p1, p3) / 2 + KL_divergence(p2, p3) / 2


def get_noniid_degree(d1, d2):
    return JS_divergence(d1, d2)


def _get_ratio(t, density_eigen, density_weight):
    """
    根据特征值获取剪枝率
    :param t:
    :param density_eigen:
    :param density_weight:
    :return:
    """
    density, grids = _density_generate(density_eigen, density_weight)
    sum = 0
    for i in range(len(grids - 1) - 1):
        if grids[i + 1] <= t:
            sum += density[i] * (grids[i + 1] - grids[i])
            i += 1
    ratio = 1 - sum
    return ratio


def _gaussian(x, x0, sigma_squared):
    return np.exp(-(x0 - x) ** 2 /
                  (2.0 * sigma_squared)) / np.sqrt(2 * np.pi * sigma_squared)


def _density_generate(eigenvalues, weights, num_bins=10000, sigma_squared=1e-5, overhead=0.01):
    """
    生成特征密度网格
    :param eigenvalues:
    :param weights:
    :param num_bins:
    :param sigma_squared:
    :param overhead:
    :return:
    """

    eigenvalues = np.array(eigenvalues)
    weights = np.array(weights)

    lambda_max = np.mean(np.max(eigenvalues, axis=1), axis=0) + overhead
    lambda_min = np.mean(np.min(eigenvalues, axis=1), axis=0) - overhead

    grids = np.linspace(lambda_min, lambda_max, num=num_bins)
    sigma = sigma_squared * max(1, (lambda_max - lambda_min))

    num_runs = eigenvalues.shape[0]
    density_output = np.zeros((num_runs, num_bins))

    for i in range(num_runs):
        for j in range(num_bins):
            x = grids[j]
            tmp_result = _gaussian(eigenvalues[i, :], x, sigma)
            density_output[i, j] = np.sum(tmp_result * weights[i, :])
    density = np.mean(density_output, axis=0)
    normalization = np.sum(density) * (grids[1] - grids[0])
    density = density / normalization
    return density, grids


class Analyzer:
    def __init__(self, train_set: Dataset, test_set: Dataset, user_groups: dict):
        self.train_dataset = train_set
        self.test_dataset = test_set
        self.user_groups = user_groups

    def get_distribution(self, idxs):
        cnt_category = [0] * len(self.train_dataset.classes)
        for idx in idxs:
            _, label = self.train_dataset[int(idx)]
            cnt_category[int(label)] += 1
        total = sum(cnt_category)
        if total == 0:
            return cnt_category
        distribution = [round(cnt / total, 3) for cnt in cnt_category]
        return distribution

    def get_target_users_distribution(self, user_ids):
        """
        获取指定客户端总体数据的分布
        :param user_ids: 
        :return:
        """
        total_idxs = np.array([])
        for user_id in user_ids:
            total_idxs = np.append(total_idxs, self.user_groups[user_id])

        users_distribution = self.get_distribution(total_idxs)
        return users_distribution

    def get_global_distribution(self):
        """
        获取总体客户端数据的分布
        :return:
        """
        total_idxs = np.array([])
        for user, idxs in self.user_groups.items():
            if user == len(self.user_groups) - 1:
                continue
            total_idxs = np.append(total_idxs, idxs)

        global_distribution = self.get_distribution(total_idxs)
        return global_distribution

    def total_rates(self, num_slices: int = 100, tiny: int = 0.0001) -> list:
        # 计算论文中提到的Non-IID度
        degrees = []
        for i in range(num_slices):
            distribution = self.get_target_users_distribution([i])
            degree = get_noniid_degree(distribution, self.get_global_distribution())
            degrees.append(degree)

        degrees = [1 / (i + tiny) for i in degrees]
        for i in range(num_slices):
            degrees[i] = degrees[i] * len(self.user_groups[i])
        rates = [i / sum(degrees) for i in degrees]
        return rates
