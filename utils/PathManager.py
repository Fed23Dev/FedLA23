from abc import ABC
import os
import time
from enum import Enum, unique
from typing import Any

from utils.objectIO import pickle_mkdir_save, pickle_load, touch_file, create_path


# 待实现优化
# 将milestone内容迁移到checkpoint对应模型下
# checkout 当目录不存在时创建
# 对象的序列化与反序列化实现
simp_time_stamp_index1 = 5
simp_time_stamp_index2 = 10


@unique
class FileType(Enum):
    # MODEL_TYPE = '.pt'
    # CONFIG_TYPE = '.yml'
    IMG_TYPE = '.png'
    LOG_TYPE = '.log'
    EXP_TYPE = '.txt'
    SEQ_TYPE = '.seq'
    RANK_TYPE = '.npy'
    CHECKPOINT_TYPE = '.snap'


def curt_time_stamp(simp: bool = False):
    pattern = '%Y.%m.%d_%H-%M-%S'
    time_str = time.strftime(pattern, time.localtime(time.time()))
    if simp:
        return time_str[simp_time_stamp_index1: simp_time_stamp_index2]
    else:
        return time_str


def file_name(file_type: FileType, name: str = None, ext_time: bool = True) -> str:
    if name is None:
        return f"{curt_time_stamp()}{file_type.value}"
    else:
        if ext_time:
            return f"{name}---{curt_time_stamp(ext_time)}{file_type.value}"
        else:
            return f"{name}{file_type.value}"


class PathManager(ABC):
    ERROR_MESS1 = "Given directory doesn't exists."
    ERROR_MESS2 = "Given key doesn't exists."

    def __init__(self, model_path: str, dataset_path: str):
        self.model_path: str = model_path
        self.dataset_path: str = dataset_path

        self.image_path = None
        self.mile_path = None
        self.log_path = None
        self.exp_path = None
        self.checkpoint_path = None

        self.curt_id = 0
        self.reg_path = []

    @staticmethod
    def load(path: str) -> Any:
        return pickle_load(path)

    @staticmethod
    def store(obj: Any, path: str):
        pickle_mkdir_save(obj, path)

    def derive_path(self, exp_base: str, image_base: str, milestone_base: str, log_base: str):
        path_base, file = os.path.split(self.model_path)
        _file_name, file_postfix = os.path.splitext(file)
        self.image_path = os.path.join(image_base, _file_name)
        self.mile_path = os.path.join(milestone_base, _file_name)
        self.log_path = os.path.join(log_base, _file_name)
        self.exp_path = os.path.join(exp_base, _file_name)
        self.checkpoint_path = path_base

    def fetch_path(self, path_id: int) -> str:
        return self.reg_path[path_id]

    def is_new(self, new: str) -> bool:
        for path in self.reg_path:
            if path == new:
                return False
        return True

    def sync_path(self, path: str) -> int:
        create_path(path)
        if self.is_new(path):
            self.reg_path.append(path)
            self.curt_id += 1
        ret = self.reg_path.index(path)
        return ret

    def latest_path(self) -> str:
        return self.fetch_path(self.curt_id-1)

    def new_log(self, name: str = None) -> (str, int):
        new_file = os.path.join(self.log_path, file_name(FileType.LOG_TYPE, name))
        touch_file(new_file)
        file_id = self.sync_path(new_file)
        return new_file, file_id

    def new_img(self, name: str = None) -> (str, int):
        new_file = os.path.join(self.image_path, file_name(FileType.IMG_TYPE, name))
        file_id = self.sync_path(new_file)
        return new_file, file_id

    def new_checkpoint(self, name: str = None, fixed: bool = False) -> (str, int):
        new_file = os.path.join(self.checkpoint_path,
                                file_name(FileType.CHECKPOINT_TYPE, name, not fixed))
        file_id = self.sync_path(new_file)
        return new_file, file_id

    def new_exp(self, name: str = None) -> (str, int):
        new_file = os.path.join(self.exp_path, file_name(FileType.EXP_TYPE, name))
        file_id = self.sync_path(new_file)
        return new_file, file_id


class HRankPathManager(PathManager):

    def __init__(self, model_path: str, dataset_path: str) -> None:
        super().__init__(model_path, dataset_path)

    # file_id 方便将来取用之前生成的str类型的file地址，通过fetch_path取用
    def new_rank(self, name: str = None) -> (str, int):
        new_file = os.path.join(self.mile_path, file_name(FileType.RANK_TYPE, name))
        file_id = self.sync_path(new_file)
        return new_file, file_id

    def new_seq(self, name: str = None) -> (str, int):
        new_file = os.path.join(self.mile_path, file_name(FileType.SEQ_TYPE, name))
        file_id = self.sync_path(new_file)
        return new_file, file_id
