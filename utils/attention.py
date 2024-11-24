import torch
import torch.nn as nn

class SelfAttention(nn.Module):
    def __init__(self, in_dim):
        super(SelfAttention, self).__init__()
        self.query = nn.Conv1d(in_channels=in_dim, out_channels=in_dim // 8, kernel_size=1)
        self.key = nn.Conv1d(in_channels=in_dim, out_channels=in_dim // 8, kernel_size=1)
        self.value = nn.Conv1d(in_channels=in_dim, out_channels=in_dim, kernel_size=1)
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        batch_size, num_channels, length = x.size()
        proj_query = self.query(x).view(batch_size, -1, length)
        proj_key = self.key(x).view(batch_size, -1, length)
        energy = torch.bmm(proj_query.permute(0, 2, 1), proj_key)
        attention = torch.softmax(energy, dim=2)
        proj_value = self.value(x).view(batch_size, -1, length)
        out = torch.bmm(proj_value, attention.permute(0, 2, 1))
        out = out.view(batch_size, num_channels, length)
        return self.gamma * out + x
