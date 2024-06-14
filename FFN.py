import torch
import math
from torch import nn
import torch.nn.functional as F

class PostionwiseFeedForward(nn.Module):
    def __init__(self, d_model, hidden, dropout=0.1):
        super(PostionwiseFeedForward, self).__init__()

        self.fc1 = nn.Linear(d_model, hidden)
        self.fc2 = nn.Linear(hidden, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        x = self.dropout(x)

        return x

