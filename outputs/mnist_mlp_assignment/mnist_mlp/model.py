import torch
from torch import nn


class MLP(nn.Module):
    """28x28 görüntüyü düzleştirip tam bağlantılı katmanlarla sınıflandırır."""

    def __init__(self, hidden_sizes):
        super().__init__()
        layers = [nn.Flatten()]
        input_size = 28 * 28
        for hidden_size in hidden_sizes:
            layers.extend([nn.Linear(input_size, hidden_size), nn.ReLU()])
            input_size = hidden_size
        layers.append(nn.Linear(input_size, 10))
        self.network = nn.Sequential(*layers)

    def forward(self, images):
        return self.network(images)
