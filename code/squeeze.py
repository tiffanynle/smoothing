from abc import ABC, abstractmethod

import torch
import torchvision.transforms.functional as TF

SQUEEZERS = ["BitSqueeze", "GaussianBlurSqueeze"]


class FeatureSqueeze(ABC):
    @abstractmethod
    def __call__(self, input: torch.Tensor) -> torch.Tensor:
        pass


class BitSqueeze(FeatureSqueeze):
    def __init__(self, bits: int) -> None:
        if bits not in range(1, 8 + 1):
            raise ValueError("bits should be in range [1, 8]")
        self.bits = bits

    def __call__(self, input: torch.Tensor) -> torch.Tensor:
        return bit_squeeze(input, self.bits)


def bit_squeeze(input: torch.Tensor, bits: int) -> torch.Tensor:
    if bits not in range(1, 8 + 1):
        raise ValueError("bits should be in range [1, 8]")

    precision = (2**bits) - 1
    squeezed = torch.round(input * precision)
    squeezed /= precision
    return squeezed


class GaussianBlurSqueeze(FeatureSqueeze):
    def __init__(
        self, kernel_size: list[int], sigma: list[float] | None = None
    ) -> None:
        self.kernel_size = kernel_size
        self.sigma = sigma

    def __call__(self, input: torch.Tensor) -> torch.Tensor:
        return TF.gaussian_blur(input, self.kernel_size, self.sigma)
