import math
import torch
import torch.nn as nn


class ScaleDotProductAttention(nn.Module):
    def __init__(self):
        super(ScaleDotProductAttention, self).__init__()
        self.softmax = nn.Softmax(dim= -1)

    def forward(self, q: torch.Tensor, k: torch.Tensor ,v: torch.Tensor, mask= None):
        batch_size, head, seq_len, d_tensor = k.size()

        # 1. dot product Query with Key^T to compute similarity
        k_t = k.transpose(2, 3)
        score = (q @ k_t) / math.sqrt(d_tensor)

        # 2. apply masking
        if mask is not None:
            score = score.masked_fill_(mask == 0, 1e-9)

        # 3. apply softmax to make [0, 1] range
        score = self.softmax(score)

        # 4. multiply with value
        v = score @ v

        return v, score