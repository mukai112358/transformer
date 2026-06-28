import torch
import torch.nn as nn


class LSTMEncoder(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, num_layers, dropout=0.1, pad_idx=0):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=pad_idx)
        self.lstm = nn.LSTM(
            embed_dim, hidden_dim, num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0.0,
            batch_first=True,
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, src):
        # src: (batch, src_len)
        emb = self.dropout(self.embedding(src))
        outputs, (h, c) = self.lstm(emb)
        return outputs, h, c  # h, c: (num_layers, batch, hidden_dim)


class LSTMDecoder(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, num_layers, dropout=0.1, pad_idx=0):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=pad_idx)
        self.lstm = nn.LSTM(
            embed_dim, hidden_dim, num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0.0,
            batch_first=True,
        )
        self.fc = nn.Linear(hidden_dim, vocab_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, tgt, h, c):
        # tgt: (batch, tgt_len)
        emb = self.dropout(self.embedding(tgt))
        outputs, (h, c) = self.lstm(emb, (h, c))
        logits = self.fc(outputs)  # (batch, tgt_len, vocab_size)
        return logits, h, c


class LSTMSeq2Seq(nn.Module):
    def __init__(
        self,
        src_vocab_size,
        tgt_vocab_size,
        embed_dim=256,
        hidden_dim=512,
        num_layers=2,
        dropout=0.1,
        pad_idx=0,
    ):
        super().__init__()
        self.encoder = LSTMEncoder(src_vocab_size, embed_dim, hidden_dim, num_layers, dropout, pad_idx)
        self.decoder = LSTMDecoder(tgt_vocab_size, embed_dim, hidden_dim, num_layers, dropout, pad_idx)
        self.pad_idx = pad_idx

    def forward(self, src, tgt):
        # 訓練時の Teacher Forcing: tgt をそのまま decoder に入れる
        _, h, c = self.encoder(src)
        logits, _, _ = self.decoder(tgt, h, c)
        return logits  # (batch, tgt_len, tgt_vocab_size)

    @torch.no_grad()
    def generate(self, src, bos_idx, eos_idx, max_len=50):
        self.eval()
        _, h, c = self.encoder(src)
        batch = src.size(0)
        ys = torch.full((batch, 1), bos_idx, dtype=torch.long, device=src.device)
        finished = torch.zeros(batch, dtype=torch.bool, device=src.device)
        for _ in range(max_len - 1):
            logits, h, c = self.decoder(ys[:, -1:], h, c)
            next_tok = logits[:, -1].argmax(-1, keepdim=True)
            ys = torch.cat([ys, next_tok], dim=1)
            finished = finished | (next_tok.squeeze(1) == eos_idx)
            if finished.all():
                break
        return ys
