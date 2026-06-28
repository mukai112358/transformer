import torch
import torch.nn as nn


class LSTMEncoder(nn.Module):
    """src 系列を読んで最終隠れ状態を作る LSTM."""

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
        emb = self.dropout(self.embedding(src))                # (batch, src_len, embed_dim)
        outputs, (h, c) = self.lstm(emb)                       # h, c: (num_layers, batch, hidden_dim)
        return outputs, h, c


class LSTMDecoder(nn.Module):
    """encoder の隠れ状態を初期値に、tgt の次トークンを予測する LSTM."""

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
        emb = self.dropout(self.embedding(tgt))                # (batch, tgt_len, embed_dim)
        outputs, (h, c) = self.lstm(emb, (h, c))               # (batch, tgt_len, hidden_dim)
        return self.fc(outputs), h, c                          # logits: (batch, tgt_len, vocab_size)


class LSTMSeq2Seq(nn.Module):
    """素の Encoder-Decoder LSTM (Attention 無し). Transformer 比較用ベースライン."""

    def __init__(self, src_vocab_size, tgt_vocab_size,
                 embed_dim=256, hidden_dim=512, num_layers=2, dropout=0.1, pad_idx=0):
        super().__init__()
        self.encoder = LSTMEncoder(src_vocab_size, embed_dim, hidden_dim, num_layers, dropout, pad_idx)
        self.decoder = LSTMDecoder(tgt_vocab_size, embed_dim, hidden_dim, num_layers, dropout, pad_idx)
        self.pad_idx = pad_idx

    def forward(self, src, tgt):
        # 訓練 (Teacher Forcing): tgt をそのまま decoder に渡す
        _, h, c = self.encoder(src)
        logits, _, _ = self.decoder(tgt, h, c)
        return logits                                          # (batch, tgt_len, tgt_vocab_size)

    @torch.no_grad()
    def greedy_decode(self, src, bos_idx=1, eos_idx=2, max_len=50):
        """src を 1 トークンずつ argmax で生成."""
        self.eval()
        _, h, c = self.encoder(src)
        batch = src.size(0)
        ys = torch.full((batch, 1), bos_idx, dtype=torch.long, device=src.device)
        finished = torch.zeros(batch, dtype=torch.bool, device=src.device)
        for _ in range(max_len - 1):
            logits, h, c = self.decoder(ys[:, -1:], h, c)      # 直近のトークンだけ入力
            next_tok = logits[:, -1].argmax(-1, keepdim=True)
            ys = torch.cat([ys, next_tok], dim=1)
            finished = finished | (next_tok.squeeze(1) == eos_idx)
            if finished.all():
                break
        return ys
