import torch

from dl.data.datasets import get_data
from env.support_config import VDataSet


class DataLoader(torch.utils.data.DataLoader):
    def __init__(self, dataset, batch_size=1, shuffle=False, sampler=None, batch_sampler=None, num_workers=8,
                 collate_fn=None, pin_memory=False, drop_last=False, timeout=0, worker_init_fn=None,
                 multiprocessing_context=None):
        super(DataLoader, self).__init__(dataset, batch_size, shuffle, sampler, batch_sampler, num_workers, collate_fn,
                                         pin_memory, drop_last, timeout, worker_init_fn, multiprocessing_context)
        self.current_iter = self.__iter__()

    def get_next_batch(self):
        try:
            return self.current_iter.__next__()
        except StopIteration:
            self.current_iter = self.__iter__()
            return self.current_iter.__next__()

    def skip_epoch(self):
        self.current_iter = self.__iter__()

    @property
    def len_data(self):
        return len(self.dataset)


def get_data_loader(name: VDataSet, data_type: str, batch_size=None, shuffle: bool = False,
                    sampler=None, transform=None, target_transform=None, subset_indices=None,
                    num_workers=8, pin_memory=False):
    assert data_type in ["train", "val", "test"]
    if data_type == "train":
        assert batch_size is not None, "Batch size for training data is required"
    if shuffle is True:
        assert sampler is None, "Cannot shuffle when using sampler"

    data = get_data(name, data_type=data_type, transform=transform, target_transform=target_transform)
    if subset_indices is not None:
        data = torch.utils.data.Subset(data, subset_indices)
    if data_type != "train" and batch_size is None:
        batch_size = len(data)

    return DataLoader(data, batch_size=batch_size, shuffle=shuffle, sampler=sampler, num_workers=num_workers,
                      pin_memory=pin_memory, drop_last=True)


def get_data_loaders(name: VDataSet, data_type: str, batch_size: int, users_indices: dict,
                     shuffle: bool = True, transform=None, target_transform=None,
                     num_workers=8, pin_memory=False) -> dict:
    assert data_type in ["train", "val", "test"]
    dataset = get_data(name, data_type=data_type, transform=transform, target_transform=target_transform)
    loaders = dict()
    for k, v in users_indices.items():
        sub_set = torch.utils.data.Subset(dataset, v)
        loaders[k] = DataLoader(sub_set, batch_size=batch_size, shuffle=shuffle,
                                num_workers=num_workers, pin_memory=pin_memory,
                                drop_last=True)
    return loaders
