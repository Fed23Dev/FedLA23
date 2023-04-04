from collections import OrderedDict

import torch
from torch import Tensor

from dl.compress.Compressor import Compressor

QUAN_BIT = {8, 16, 32}
TC_KEYS = ["com_tensor", "ori_shape", "ori_type", "l2_norm"]


def is_legal(qb: int) -> bool:
    if qb in QUAN_BIT:
        return True
    else:
        return False


def switch_type(qb: int) -> torch.dtype:
    if qb == 8:
        return torch.int8
    elif qb == 16:
        return torch.int16
    else:
        return torch.int32


class QuantizationSGD(Compressor):
    """
        (2016). QSGD: Communication-Efficient SGD via Gradient Quantization and Encoding.
        Retrieved from https://arxiv.org/abs/1610.02132
    """
    ERROR_MESS1 = "quan_bit must be 8, 16, 32."
    ERROR_MESS2 = "tensor_compressed must be created from QuantizationSGD.compress()."

    def __init__(self, quan_bit: int):
        assert is_legal(quan_bit), self.ERROR_MESS1
        super().__init__(TC_KEYS)
        self.qb = quan_bit
        self.type = switch_type(quan_bit)

    def compress(self, tensor: Tensor) -> OrderedDict:
        vector = torch.flatten(tensor)
        l2_norm = torch.norm(vector, p=2)
        abs_vector = torch.abs(vector)
        tensor_qb = torch.tensor(self.qb, dtype=vector.dtype)

        tmp_compress = tensor_qb / l2_norm * abs_vector
        tmp_int_compress = torch.floor(tmp_compress)
        random_vector = torch.randn(vector.shape)
        tmp_add_compress = torch.less(random_vector, (tmp_compress - tmp_int_compress))

        abs_compress = tmp_int_compress + tmp_add_compress
        sign_vector = torch.sign(vector)
        vector_compress = sign_vector * abs_compress

        vector_compressed = vector_compress.type(self.type)
        self.compress_context[TC_KEYS[0]] = vector_compressed
        self.compress_context[TC_KEYS[1]] = tensor.shape
        self.compress_context[TC_KEYS[2]] = tensor.dtype
        self.compress_context[TC_KEYS[3]] = l2_norm

        # to realize: vector_compressed -> COO vector

        return self.compress_context

    def decompress(self, tensor_compressed: OrderedDict = None) -> Tensor:
        if tensor_compressed is None:
            tensor_compressed = self.compress_context
        else:
            assert self.dict_legal(tensor_compressed), self.ERROR_MESS2

        com_tensor = tensor_compressed[TC_KEYS[0]]
        tmp_tensor = com_tensor.type(tensor_compressed[TC_KEYS[2]])
        tensor_qb = torch.tensor(self.qb, dtype=tensor_compressed[TC_KEYS[2]])
        tensor_decompressed = tensor_compressed[TC_KEYS[3]] / tensor_qb * tmp_tensor
        tensor_decompressed.resize_(*tensor_compressed[TC_KEYS[1]])

        return tensor_decompressed
