import numpy as np
import torch
from sklearn.cluster import DBSCAN, OPTICS, AgglomerativeClustering, BisectingKMeans


def tes():
    all = torch.tensor([[2, 2, 2], [3, 3, 3], [4, 4, 4]])
    zero = torch.tensor([[0, 0, 0], [1, 0.2, 1], [0, 0, 0]])

    mask = ~zero.bool()
    print(mask * all + zero)


def dbscan():
    a = torch.randn(10, 10).reshape(-1)
    b = torch.randn(10, 10).reshape(-1)
    c = torch.randn(10, 10).reshape(-1)

    lis = [a, b, c]

    X = torch.stack(lis, dim=0).numpy()
    clustering = DBSCAN(eps=200, min_samples=2).fit(X)
    print(clustering.labels_)

    clustering = OPTICS(min_samples=3).fit(X)
    print(clustering.labels_)

    # Specific
    clustering = AgglomerativeClustering(n_clusters=3).fit(X)
    print(clustering.labels_)

    clustering = BisectingKMeans(n_clusters=2).fit(X)
    print(clustering.labels_)


def timestamp():
    import time

    st3 = 11
    st4 = 14
    st5 = 17

    def only_time_stamp():
        pattern = '%Y.%m.%d_%H-%M-%S'
        time_str = time.strftime(pattern, time.localtime(time.time()))
        return time_str[st3: st3 + 2] + time_str[st4: st4 + 2] + time_str[st5: st5 + 2]

    print(only_time_stamp())


def test_lis():
    sample = [2, 1, 0, 1]


if __name__ == "__main__":
    from utils.AccExtractor import AccExtractor
    path = "logs/super"
    extractor = AccExtractor(path)
    extractor.extract_acc_data()
    extractor.show_detail_rets()
    extractor.show_avg_rets()
    print("----------------------")
