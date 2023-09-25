import numpy as np
import torch
from sklearn.cluster import DBSCAN, OPTICS, AgglomerativeClustering, BisectingKMeans


def tes():
    all = torch.tensor([[2, 2, 2], [3, 3, 3], [4, 4, 4]])
    zero = torch.tensor([[0, 0, 0], [1, 0.2, 1], [0, 0, 0]])

    mask = ~zero.bool()
    print(mask * all + zero)


if __name__ == "__main__":
    a = torch.randn(10, 10).reshape(-1)
    b = torch.randn(10, 10).reshape(-1)
    c = torch.randn(10, 10).reshape(-1)

    X = torch.stack((a, b, c), dim=0).numpy()
    clustering = DBSCAN(eps=200, min_samples=2).fit(X)
    print(clustering.labels_)

    clustering = OPTICS(min_samples=3).fit(X)
    print(clustering.labels_)

    clustering = OPTICS(min_samples=3).fit(X)
    print(clustering.labels_)

    clustering = AgglomerativeClustering(n_clusters=2).fit(X)
    print(clustering.labels_)

    clustering = BisectingKMeans(n_clusters=2).fit(X)
    print(clustering.labels_)
    print("----------------------")
