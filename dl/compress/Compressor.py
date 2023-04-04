import abc
from typing import List

from torch import Tensor
from collections import OrderedDict


class Compressor(abc.ABC):
    def __init__(self, keys: List[str]):
        super().__init__()
        self.compress_context = OrderedDict()
        self.keys = keys

    def dict_legal(self, tc: OrderedDict) -> bool:
        if all(k in tc.keys() for k in self.keys):
            return True
        else:
            return False

    @abc.abstractmethod
    def compress(self, tensor: Tensor) -> OrderedDict:
        pass

    @abc.abstractmethod
    def decompress(self, tensor_compressed: OrderedDict = None) -> Tensor:
        pass
