from abc import ABC, abstractmethod

import torch

SQUEEZERS = ["BitSqueeze"]


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
