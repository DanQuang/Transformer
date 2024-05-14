import torch.nn as nn

from blocks.encoder_block import EncoderBlock
from embedding.transformer_embedding import TransformerEmbedding

class Encoder(nn.Module):
    def __init__(self, enc_vocab_size, enc_pad_idx, seq_len, embedding_dim, hidden, n_head, n_layers, dropout):
        super(Encoder, self).__init__()

        self.embedding = TransformerEmbedding(vocab_size= enc_vocab_size,
                                              embedding_dim= embedding_dim,
                                              seq_len= seq_len,
                                              dropout= dropout,
                                              padding_idx= enc_pad_idx)
        
        self.layers = nn.ModuleList([EncoderBlock(embedding_dim= embedding_dim,
                                                  hidden= hidden,
                                                  n_head= n_head,
                                                  dropout= dropout)]
                                    for _ in range(n_layers))
        
    
    def forward(self, x, src_mask):
        x = self.embedding(x)

        for layer in self.layers:
            x = layer(x, src_mask)