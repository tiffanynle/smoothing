import torch


class LinearHead(torch.nn.Module):
    def __init__(self, model: torch.nn.Module, num_classes: int, embed_dim: int):
        super().__init__()
        self.model = model
        self.linear = torch.nn.Linear(embed_dim, num_classes)

    def forward(self, x):
        x = self.model(x).detach()
        return self.linear(x)
