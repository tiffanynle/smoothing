import torch


class LinearHead(torch.nn.Module):
    def __init__(self, num_classes: int, embed_dim: int):
        super().__init__()
        self.head = torch.nn.Linear(embed_dim, num_classes)

    def forward(self, x):
        return self.head(x)
