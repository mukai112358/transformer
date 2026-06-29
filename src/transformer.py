import math

import torch
import torch.nn as nn

from src.positional_encoding import PositionalEncoding
from src.encoder import Encoder
from src.decoder import Decoder, make_causal_mask


class Transformer(nn.Module):
    def __init__(self,
                 src_vocab_size,
                 tgt_vocab_size,
                 d_model=512,
                 num_heads=8,
                 d_ff=2048,
                 num_layers=6,
                 max_len=1000,
                 dropout=0.1):
        super().__init__()
        self.d_model = d_model

        #埋め込み層
        self.src_embedding = nn.Embedding(src_vocab_size, d_model)
        self.tgt_embedding = nn.Embedding(tgt_vocab_size, d_model)

        #位置エンコーディング
        self.pos_encoding = PositionalEncoding(d_model, max_len, dropout)

        #エンコーダー、デコーダー
        self.encoder = Encoder(d_model, num_heads, d_ff, num_layers, dropout)
        self.decoder = Decoder(d_model, num_heads, d_ff, num_layers, dropout)

        # 最終出力層
        self.generator = nn.Linear(d_model, tgt_vocab_size)

        # パラメータの初期化
        self._init_parameters()

    def _init_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def make_src_mask(self, src, pad_idx=0):
        src_mask = (src != pad_idx).unsqueeze(1).unsqueeze(2)
        return src_mask

    def make_tgt_mask(self, tgt, pad_idx=0):
        pad_mask = (tgt != pad_idx).unsqueeze(1).unsqueeze(2)
        causal_mask = make_causal_mask(tgt.size(1)).to(tgt.device).bool()
        tgt_mask = pad_mask & causal_mask
        return tgt_mask

    def forward(self, src, tgt, pad_idx=0):
        src_mask = self.make_src_mask(src, pad_idx)
        tgt_mask = self.make_tgt_mask(tgt, pad_idx)

        src_emb = self.src_embedding(src) * math.sqrt(self.d_model)
        src_emb = self.pos_encoding(src_emb)
        enc_output = self.encoder(src_emb, src_mask)

        tgt_emb = self.tgt_embedding(tgt) * math.sqrt(self.d_model)
        tgt_emb = self.pos_encoding(tgt_emb)
        dec_output = self.decoder(tgt_emb, enc_output, src_mask, tgt_mask)

        output = self.generator(dec_output)
        return output

    # ↓ 推論用 (notebook 07 にはない、評価のために追加)
    @torch.no_grad()
    def greedy_decode(self, src, bos_idx=1, max_len=50):
        self.eval()
        src_mask = self.make_src_mask(src)
        src_emb = self.pos_encoding(self.src_embedding(src) * math.sqrt(self.d_model))
        memory = self.encoder(src_emb, src_mask)

        ys = torch.full((src.size(0), 1), bos_idx, dtype=torch.long, device=src.device)
        for _ in range(max_len - 1):
            tgt_mask = self.make_tgt_mask(ys)
            tgt_emb = self.pos_encoding(self.tgt_embedding(ys) * math.sqrt(self.d_model))
            dec_output = self.decoder(tgt_emb, memory, src_mask, tgt_mask)
            next_tok = self.generator(dec_output[:, -1]).argmax(-1, keepdim=True)
            ys = torch.cat([ys, next_tok], dim=1)
        return ys
