import math

import torch
import torch.nn as nn

from src.positional_encoding import PositionalEncoding
from src.encoder import Encoder
from src.decoder import Decoder


class Transformer(nn.Module):
    def __init__(self, src_vocab_size, tgt_vocab_size,
                 d_model=512, num_heads=8, d_ff=2048, num_layers=6,
                 max_len=1000, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        self.src_embedding = nn.Embedding(src_vocab_size, d_model)
        self.tgt_embedding = nn.Embedding(tgt_vocab_size, d_model)
        self.pos_encoding  = PositionalEncoding(d_model, max_len, dropout)
        self.encoder = Encoder(d_model, num_heads, d_ff, num_layers, dropout)
        self.decoder = Decoder(d_model, num_heads, d_ff, num_layers, dropout)
        self.generator = nn.Linear(d_model, tgt_vocab_size)
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def make_src_mask(self, src, pad_idx=0):
        return (src != pad_idx).unsqueeze(1).unsqueeze(2)

    def make_tgt_mask(self, tgt, pad_idx=0):
        pad_mask = (tgt != pad_idx).unsqueeze(1).unsqueeze(2)
        causal_mask = torch.tril(torch.ones(tgt.size(1), tgt.size(1), device=tgt.device)).bool()
        return pad_mask & causal_mask

    def encode(self, src):
        src_mask = self.make_src_mask(src)
        x = self.pos_encoding(self.src_embedding(src) * math.sqrt(self.d_model))
        return self.encoder(x, src_mask), src_mask

    def decode(self, tgt, memory, src_mask):
        tgt_mask = self.make_tgt_mask(tgt)
        x = self.pos_encoding(self.tgt_embedding(tgt) * math.sqrt(self.d_model))
        return self.decoder(x, memory, src_mask, tgt_mask)

    def forward(self, src, tgt):
        memory, src_mask = self.encode(src)
        return self.generator(self.decode(tgt, memory, src_mask))

    @torch.no_grad()
    def greedy_decode(self, src, bos_idx=1, max_len=50):
        self.eval()
        memory, src_mask = self.encode(src)
        ys = torch.full((src.size(0), 1), bos_idx, dtype=torch.long, device=src.device)
        for _ in range(max_len - 1):
            next_tok = self.generator(self.decode(ys, memory, src_mask)[:, -1]).argmax(-1, keepdim=True)
            ys = torch.cat([ys, next_tok], dim=1)
        return ys
