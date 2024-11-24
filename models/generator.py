import torch
import torch.nn as nn
from utils.attention import SelfAttention

class Generator(nn.Module):
    def __init__(self, noise_size=100, output_size=512, hidden_sizes=[512, 256], dropout_rate=0.1):
        super(Generator, self).__init__()
        layers = [SelfAttention(noise_size)]
        hidden_sizes = [noise_size] + hidden_sizes
        for i in range(len(hidden_sizes) - 1):
            layers.extend([
                nn.utils.spectral_norm(nn.Linear(hidden_sizes[i], hidden_sizes[i + 1])),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Dropout(dropout_rate)
            ])
        layers.append(nn.utils.spectral_norm(nn.Linear(hidden_sizes[-1], output_size)))
        self.layers = nn.Sequential(*layers)

    def forward(self, noise):
        return self.layers(noise)
