import torch
import torch.nn as nn

from modules.encoder import Encoder
from modules.decoder import Decoder


class Transformer(nn.Module):
    def __init__(self, src_pad_idx, trg_pad_idx, enc_vocab_size, dec_vocab_size, embedding_dim,
                 n_head, seq_len, hidden, n_layers, dropout):
        
        super(Transformer, self).__init__()
        self.src_pad_idx = src_pad_idx
        self.trg_pad_idx = trg_pad_idx
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        self.encoder = Encoder(enc_vocab_size= enc_vocab_size,
                               enc_pad_idx= src_pad_idx,
                               seq_len= seq_len,
                               embedding_dim= embedding_dim,
                               hidden= hidden,
                               n_head= n_head,
                               n_layers= n_layers,
                               dropout= dropout)
        
        self.decoder = Decoder(dec_vocab_size= dec_vocab_size,
                               dec_pad_idx= trg_pad_idx,
                               seq_len= seq_len,
                               embedding_dim= embedding_dim,
                               hidden= hidden,
                               n_head= n_head,
                               n_layers= n_layers,
                               dropout= dropout)
        
        self.linear = nn.Linear(embedding_dim, dec_vocab_size)

    def forward(self, src, trg):
        src_mask = self.make_src_mask(src)
        trg_mask = self.make_trg_mask(trg)
        enc_src = self.encoder(src, src_mask)
        dec_trg = self.decoder(trg, enc_src, trg_mask, src_mask)
        out = self.linear(dec_trg)
        return torch.log_softmax(out, dim= -1)

    def make_src_mask(self, src: torch.Tensor):
        src_mask = (src != self.src_pad_idx).unsqueeze(1).unsqueeze(2) # [batch_size, 1, 1, seq_len]
        return src_mask
    
    def make_trg_mask(self, trg: torch.Tensor):
        trg_pad_mask = (trg != self.trg_pad_idx).unsqueeze(1).unsqueeze() # [batch_size, 1, 1, seq_len]
        seq_len = trg.shape[-1]
        trg_sub_mask = torch.tril(torch.ones(seq_len, seq_len)).type(torch.ByteTensor).to(self.device)
        trg_mask = trg_pad_mask & trg_sub_mask
        return trg_mask

