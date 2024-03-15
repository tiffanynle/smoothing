import torch
import torch.backends.cudnn as cudnn
from archs.cifar_resnet import resnet as resnet_cifar
from archs.classifier_heads import LinearHead
from datasets import get_normalize_layer
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


BACKBONES = ["dinov2_vits14"]
_EMBED_DIMS = {"dinov2_vits14": 384}
HEADS = ["linear"]


def _load_backbone(arch: str):
    if arch == "dinov2_vits14":
        return torch.hub.load("facebookresearch/dinov2", "dinov2_vits14")


def get_backbone(arch: str, num_classes: int = None, head: str = None):
    model = _load_backbone(arch)
    if head == "linear":
        model = LinearHead(model, num_classes=num_classes, embed_dim=_EMBED_DIMS[arch])
    return model
