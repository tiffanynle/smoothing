import os
from typing import *

import torch
from torch.utils.data import Dataset
from torchvision import datasets, transforms

# set this environment variable to the location of your imagenet directory if you want to read ImageNet data.
# make sure your val directory is preprocessed to look like the train directory, e.g. by running this script
# https://raw.githubusercontent.com/soumith/imagenetloader.torch/master/valprep.sh
IMAGENET_LOC_ENV = "IMAGENET_DIR"

# list of all datasets
DATASETS = ["imagenet", "cifar10"]

# list of all embeddings
EMBEDDINGS = ["dinov2"]


def get_dataset(dataset: str, split: str, embedding: str = None) -> Dataset:
    """Return the dataset as a PyTorch Dataset object"""
    if dataset == "imagenet":
        return _imagenet(split, embedding)
    elif dataset == "cifar10":
        return _cifar10(split, embedding)


def get_num_classes(dataset: str):
    """Return the number of classes in the dataset."""
    if dataset == "imagenet":
        return 1000
    elif dataset == "cifar10":
        return 10


def get_normalize_layer(dataset: str, embedding: str = None) -> torch.nn.Module:
    """Return the dataset's normalization layer"""
    if embedding is not None:
        return NormalizeLayer(_EMBEDDING_MEAN[embedding], _EMBEDDING_STDDEV[embedding])
    elif dataset == "imagenet":
        return NormalizeLayer(_DATASET_MEAN[dataset], _DATASET_STTDEV[dataset])
    elif dataset == "cifar10":
        return NormalizeLayer(_DATASET_MEAN[dataset], _DATASET_STTDEV[dataset])


_DATASET_MEAN = {
    "cifar10": [0.4914, 0.4822, 0.4465],
    "imagenet": [0.485, 0.456, 0.406],
}

_DATASET_STTDEV = {
    "cifar10": [0.2023, 0.1994, 0.2010],
    "imagenet": [0.229, 0.224, 0.225],
}

_EMBEDDING_MEAN = {
    "dinov2": [0.485, 0.456, 0.406],
}

_EMBEDDING_STDDEV = {
    "dinov2": [0.229, 0.224, 0.225],
}

_DATASET_TRANSFORM = {
    ("cifar10", "train"): [
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
    ],
    ("cifar10", "test"): [transforms.ToTensor()],
    ("imagenet", "train"): [
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
    ],
    ("imagenet", "test"): [
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
    ],
    # transforms specifically so inputs can be fed into the pre-trained models to generate embeddings
    ("dinov2", "train"): [
        transforms.Resize(256),
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
    ],
    ("dinov2", "test"): [
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
    ],
}


def _cifar10(split: str, embedding: str = None) -> Dataset:
    transforms_list = []
    if embedding is not None:
        transforms_list += _DATASET_TRANSFORM[embedding, split]
    else:
        transforms_list += _DATASET_TRANSFORM["cifar10", split]
    if split == "train":
        return datasets.CIFAR10(
            "./dataset_cache",
            train=True,
            download=True,
            transform=transforms.Compose(transforms_list),
        )
    elif split == "test":
        return datasets.CIFAR10(
            "./dataset_cache",
            train=False,
            download=True,
            transform=transforms.Compose(transforms_list),
        )


def _imagenet(split: str, embedding: str = None) -> Dataset:
    if not IMAGENET_LOC_ENV in os.environ[IMAGENET_LOC_ENV]:
        raise RuntimeError("environment variable for ImageNet directory not set")

    dir = os.environ[IMAGENET_LOC_ENV]
    transforms_list = []
    if embedding is not None:
        transforms_list += _DATASET_TRANSFORM[embedding, split]
    else:
        transforms_list += _DATASET_TRANSFORM["imagenet", split]
    if split == "train":
        subdir = os.path.join(dir, "train")
    elif split == "test":
        subdir = os.path.join(dir, "val")

    transform = transforms.Compose(transforms_list)
    return datasets.ImageFolder(subdir, transform)


class NormalizeLayer(torch.nn.Module):
    """Standardize the channels of a batch of images by subtracting the dataset mean
    and dividing by the dataset standard deviation.

    In order to certify radii in original coordinates rather than standardized coordinates, we
    add the Gaussian noise _before_ standardizing, which is why we have standardization be the first
    layer of the classifier rather than as a part of preprocessing as is typical.
    """

    def __init__(self, means: List[float], sds: List[float]):
        """
        :param means: the channel means
        :param sds: the channel standard deviations
        """
        super(NormalizeLayer, self).__init__()
        self.means = torch.tensor(means)
        self.sds = torch.tensor(sds)
        self.means = self.means.cuda()
        self.sds = self.sds.cuda()

    def forward(self, input: torch.tensor):
        if input.ndim == 4:
            (batch_size, num_channels, height, width) = input.shape
            means = self.means.repeat((batch_size, height, width, 1)).permute(0, 3, 1, 2)
            sds = self.sds.repeat((batch_size, height, width, 1)).permute(0, 3, 1, 2)
            return (input - means) / sds
        elif input.ndim == 2:
            # should work fine to just do broadcasting here
            return (input - self.means) / self.sds
