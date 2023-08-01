import numpy as np
import torch

if __name__ == "__main__":
    # from dl.test_unit import main
    # main()
    a = torch.ones(3, 5)
    a[2][2] = 0
    print(torch.nn.functional.log_softmax(a, dim=1))
    print(torch.nn.functional.softmax(a, dim=1))
    print("----------------------")
