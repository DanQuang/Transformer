import torch
import torch.nn as nn


class LayerNorm(nn.Module):
    def __init__(self, embedding_dim, eps = 1e-12):
        super(LayerNorm, self).__init__()
        self.gamma = nn.Parameter(torch.ones(embedding_dim))
        self.beta = nn.Parameter(torch.zeros(embedding_dim))
        self.eps = eps

    def forward(self, x: torch.Tensor):
        mean = x.mean(-1, keepdim= True)
        var = x.var(-1, unbiased= False, keepdim= True)

        out = (x - mean) / torch.sqrt(var + self.eps)
        out = self.gamma*out +self.beta
        return out