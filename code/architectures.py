import torch
import torch.backends.cudnn as cudnn
from archs.cifar_resnet import resnet as resnet_cifar
from archs.classifier_heads import *
from datasets import get_normalize_layer, get_num_classes
from torchvision.models.resnet import resnet50

# resnet50 - the classic ResNet-50, sized for ImageNet
# cifar_resnet20 - a 20-layer residual network sized for CIFAR
# cifar_resnet110 - a 110-layer residual network sized for CIFAR
ARCHITECTURES = [
    "resnet50",
    "cifar_resnet20",
    "cifar_resnet110",
]


def get_architecture(arch: str, dataset: str) -> torch.nn.Module:
    """Return a neural network (with random weights)

    :param arch: the architecture - should be in the ARCHITECTURES list above
    :param dataset: the dataset - should be in the datasets.DATASETS list
    :return: a Pytorch module
    """
    if arch == "resnet50" and dataset == "imagenet":
        model = torch.nn.DataParallel(resnet50(pretrained=False)).cuda()
        cudnn.benchmark = True
    elif arch == "cifar_resnet20":
        model = resnet_cifar(depth=20, num_classes=10).cuda()
    elif arch == "cifar_resnet110":
        model = resnet_cifar(depth=110, num_classes=10).cuda()
    normalize_layer = get_normalize_layer(dataset)
    return torch.nn.Sequential(normalize_layer, model)


BACKBONES = [
    "dinov2_vits14",
    "dinov2_vits14_reg",
    "dinov2_vitb14",
    "dinov2_vitb14_reg",
]
_EMBEDDING_DIMS = {
    "dinov2_vits14": 384,
    "dinov2_vits14_reg": 384,
    "dinov2_vitb14": 768,
    "dinov2_vitb14_reg": 768,
}
HEADS = ["linear"]


def _load_backbone(backbone: str) -> torch.nn.Module:
    if backbone == "dinov2_vits14":
        return torch.hub.load("facebookresearch/dinov2", "dinov2_vits14")
    elif backbone == "dinov2_vits14_reg":
        return torch.hub.load("facebookresearch/dinov2", "dinov2_vits14_reg")
    elif backbone == "dinov2_vitb14":
        return torch.hub.load("facebookresearch/dinov2", "dinov2_vitb14")
    elif backbone == "dinov2_vitb14_reg":
        return torch.hub.load("facebookresearch/dinov2", "dinov2_vitb14_reg")


def get_backbone(embedding: str, backbone: str, dataset: str) -> torch.nn.Module:
    """Returns pre-trained vision transformer

    :param embedding: the embedding to generate - should be in datasets.EMBEDDINGS list
    :type embedding: str
    :param backbone: pre-trained model to use for embeddings - should be in BACKBONES list
    :type backbone: str
    :param dataset: the dataset - should be in datasets.DATASETS list
    :type dataset: str
    :return: a PyTorch module
    :rtype: torch.nn.Module
    """
    model = _load_backbone(backbone).cuda()
    normalize_layer = get_normalize_layer(dataset, embedding)
    return torch.nn.Sequential(normalize_layer, model)


def get_head(head: str, backbone: str, dataset: str) -> torch.nn.Module:
    """Returns classifier head

    :param head: classifier head to use - should be in HEADS list
    :type head: str
    :param backbone: backbone used to generate embeddings - should be in BACKBONES list
    :type backbone: str
    :param dataset: the dataset -should be in datasets.DATASETS list
    :type dataset: str
    :return: a PyTorch module
    :rtype: torch.nn.Module
    """
    if head == "linear":
        return LinearHead(get_num_classes(dataset), _EMBEDDING_DIMS[backbone])
