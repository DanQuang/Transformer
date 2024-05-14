import torch
import torch.nn as nn


class PositionWiseFeedForward(nn.Module):
    def __init__(self, embedding_dim, hidden, dropout):
        super(PositionWiseFeedForward, self).__init__()

        self.linear1 = nn.Linear(embedding_dim, hidden)
        self.linear2 = nn.Linear(hidden, embedding_dim)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.linear2(self.dropout(self.relu(self.linear1(x))))