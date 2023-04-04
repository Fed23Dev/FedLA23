import torch
from math import floor
from collections import OrderedDict
from dl.compress.Compressor import Compressor

TC_KEYS = ["values", "indices", "ori_shape"]


def is_legal(cr: float) -> bool:
    if 1 > cr > 0:
        return True
    else:
        return False


class TopKSparse(Compressor):
    ERROR_MESS1 = "compress_rate must be in (0,1)."
    ERROR_MESS2 = "tensor_compressed must be created from TopKSparse.compress()."

    def __init__(self, compress_rate: float):
        assert is_legal(compress_rate), self.ERROR_MESS1
        super().__init__(TC_KEYS)
        self.cr = round(compress_rate, 2)

    def compress(self, tensor: torch.Tensor) -> OrderedDict:
        origin_shape = tensor.shape
        total = origin_shape[-1]
        k = floor(total * self.cr)
        values, indices = torch.topk(tensor, k)
        self.compress_context[TC_KEYS[0]] = values
        self.compress_context[TC_KEYS[1]] = indices
        self.compress_context[TC_KEYS[2]] = tensor.shape
        return self.compress_context

    def decompress(self, tensor_compressed: OrderedDict = None) -> torch.Tensor:
        if tensor_compressed is None:
            tensor_compressed = self.compress_context
        else:
            assert self.dict_legal(tensor_compressed), self.ERROR_MESS2
        ori_tensor = torch.zeros(tensor_compressed[TC_KEYS[2]])
        return ori_tensor.scatter_(dim=-1, index=tensor_compressed[TC_KEYS[1]],
                                   src=tensor_compressed[TC_KEYS[0]])
