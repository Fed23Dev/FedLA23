import numpy as np
import torch

if __name__ == "__main__":
    # from dl.test_unit import main
    # main()
    labels = torch.tensor([1, 1, 1, 1, 2, 1, 2])
    _labels, _cnt = torch.unique(labels, return_counts=True)
    labels_cnt = torch.zeros(3, dtype=_cnt.dtype) \
        .scatter_(dim=0, index=_labels, src=_cnt)
    print(labels_cnt)

    info_matrix = torch.zeros(3, 3)
    info_matrix[0][0] = 1
    info_matrix[1][1] = 1
    info_matrix[2][2] = 1

    target = torch.tensor([0, 1, 2, 1])
    target = target.unsqueeze(1).expand(target.size()[0], info_matrix.size()[0])

    # info_target = torch.zeros(target.size()[0], info_matrix.size()[0]) \
    #     .scatter_(dim=0, index=target, src=info_matrix)
    # print(info_target)

    print(torch.gather(input=info_matrix, dim=0, index=target))
    print("----------------------")
