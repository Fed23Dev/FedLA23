import torch
import torchvision
from torch.nn.functional import one_hot
import torchvision.transforms as transforms


class Flatten:
    def __call__(self, img: torch.FloatTensor):
        return img.reshape((-1))


class OneHot:
    def __init__(self, n_classes, to_float: bool = False):
        self.n_classes = n_classes
        self.to_float = to_float

    def __call__(self, label: torch.Tensor):
        return one_hot(label, self.n_classes).float() if self.to_float else one_hot(label, self.n_classes)


class DataToTensor:
    def __init__(self, dtype=None):
        if dtype is None:
            dtype = torch.float
        self.dtype = dtype

    def __call__(self, data):
        return torch.tensor(data, dtype=self.dtype)


# compatible for torchgeo
def _dic2img(dic, num_classes: int):
    img = transforms.ToPILImage()(dic["image"])
    label = dic["label"]

    img = transforms.Resize((256, 256))(img)
    img = transforms.RandomHorizontalFlip()(img)
    img = transforms.RandomCrop(224)(img)
    img = torchvision.transforms.ToTensor()(img)
    img = transforms.Normalize(mean=(0.485, 0.456, 0.406),
                               std=(0.229, 0.224, 0.225))(img)

    label = OneHot(num_classes, to_float=True)(label)
    return img, label


def init_transform(data_type: str, mean: list, std: list):
    if data_type == "train":
        return transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            # transforms.RandomRotation(15),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)])
    else:
        return transforms.Compose([transforms.ToTensor(),
                                   transforms.Normalize(mean, std)])


def init_target_transform(num_classes: int):
    return transforms.Compose([DataToTensor(dtype=torch.long),
                               OneHot(num_classes, to_float=True)])


def init_imgfolder_transform(mean: list, std: list):
    return transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ])


def init_imgfolder_ucm_transform():
    return transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(224),
        torchvision.transforms.ToTensor(),
        transforms.Normalize(mean=(0.485, 0.456, 0.406),
                             std=(0.229, 0.224, 0.225))
    ])


def init_tiny_imagenet_transform():
    return transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.48024505, 0.4480726, 0.39754787),
                             std=(0.2717199, 0.26526922, 0.27396977))])
