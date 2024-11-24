import torch
import torch.nn as nn

class Discriminator(nn.Module):
    def __init__(self, input_size=512, hidden_sizes=[512], num_labels=3, dropout_rate=0.1):
        super(Discriminator, self).__init__()
        self.input_dropout = nn.Dropout(p=dropout_rate)
        layers = []
        hidden_sizes = [input_size] + hidden_sizes
        for i in range(len(hidden_sizes) - 1):
            layers.extend([
                nn.Linear(hidden_sizes[i], hidden_sizes[i + 1]),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Dropout(dropout_rate)
            ])
        self.layers = nn.Sequential(*layers)
        self.logit = nn.Linear(hidden_sizes[-1], num_labels + 1)  # +1 for real/fake classification
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, input_rep):
        input_rep = self.input_dropout(input_rep)
        last_rep = self.layers(input_rep)
        logits = self.logit(last_rep)
        return last_rep, logits, self.softmax(logits)
