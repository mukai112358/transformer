import math

import torch
import torch.nn as nn

from src.positional_encoding import PositionalEncoding
from src.encoder import Encoder
from src.decoder import Decoder


class Transformer(nn.Module):
    """論文 Attention Is All You Need の Encoder-Decoder Transformer."""

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

        # Xavier 初期化
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def make_src_mask(self, src, pad_idx=0):
        # pad 以外を 1 にしたマスク. shape: (batch, 1, 1, src_len)
        return (src != pad_idx).unsqueeze(1).unsqueeze(2)

    def make_tgt_mask(self, tgt, pad_idx=0):
        # pad mask と下三角 (causal) mask の AND. shape: (batch, 1, tgt_len, tgt_len)
        tgt_len = tgt.size(1)
        pad_mask    = (tgt != pad_idx).unsqueeze(1).unsqueeze(2)
        causal_mask = torch.tril(torch.ones(tgt_len, tgt_len, device=tgt.device)).bool()
        return pad_mask & causal_mask

    def encode(self, src):
        # src -> memory (encoder 出力)
        src_mask = self.make_src_mask(src)
        x = self.src_embedding(src) * math.sqrt(self.d_model)             # Embedding * √d_model
        x = self.pos_encoding(x)                                          # + Positional Encoding
        return self.encoder(x, src_mask), src_mask

    def decode(self, tgt, memory, src_mask):
        # tgt + memory -> decoder 出力
        tgt_mask = self.make_tgt_mask(tgt)
        x = self.tgt_embedding(tgt) * math.sqrt(self.d_model)
        x = self.pos_encoding(x)
        return self.decoder(x, memory, src_mask, tgt_mask)

    def forward(self, src, tgt):
        # 訓練: (batch, src_len), (batch, tgt_len) -> (batch, tgt_len, tgt_vocab_size)
        memory, src_mask = self.encode(src)
        out = self.decode(tgt, memory, src_mask)
        return self.generator(out)

    @torch.no_grad()
    def greedy_decode(self, src, bos_idx=1, eos_idx=2, max_len=50):
        """src を 1 トークンずつ argmax で生成. encoder は1回だけ実行."""
        self.eval()
        memory, src_mask = self.encode(src)
        batch = src.size(0)
        ys = torch.full((batch, 1), bos_idx, dtype=torch.long, device=src.device)
        finished = torch.zeros(batch, dtype=torch.bool, device=src.device)
        for _ in range(max_len - 1):
            out = self.decode(ys, memory, src_mask)
            next_tok = self.generator(out[:, -1]).argmax(-1, keepdim=True)
            ys = torch.cat([ys, next_tok], dim=1)
            finished = finished | (next_tok.squeeze(1) == eos_idx)
            if finished.all():
                break
        return ys
