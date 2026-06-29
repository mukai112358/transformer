import torch
import torch.nn as nn


class LSTMEncoder(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, num_layers, dropout=0.1, pad_idx=0):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=pad_idx)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, num_layers=num_layers,
                            dropout=dropout if num_layers > 1 else 0.0, batch_first=True)

    def forward(self, src):
        return self.lstm(self.embedding(src))


class LSTMDecoder(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, num_layers, dropout=0.1, pad_idx=0):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=pad_idx)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, num_layers=num_layers,
                            dropout=dropout if num_layers > 1 else 0.0, batch_first=True)
        self.fc = nn.Linear(hidden_dim, vocab_size)

    def forward(self, tgt, hc):
        out, hc = self.lstm(self.embedding(tgt), hc)
        return self.fc(out), hc


class LSTMSeq2Seq(nn.Module):
    def __init__(self, src_vocab_size, tgt_vocab_size,
                 embed_dim=256, hidden_dim=512, num_layers=2, dropout=0.1, pad_idx=0):
        super().__init__()
        self.encoder = LSTMEncoder(src_vocab_size, embed_dim, hidden_dim, num_layers, dropout, pad_idx)
        self.decoder = LSTMDecoder(tgt_vocab_size, embed_dim, hidden_dim, num_layers, dropout, pad_idx)

    def forward(self, src, tgt):
        _, hc = self.encoder(src)
        logits, _ = self.decoder(tgt, hc)
        return logits

    @torch.no_grad()
    def greedy_decode(self, src, bos_idx=1, max_len=50):
        self.eval()
        _, hc = self.encoder(src)
        ys = torch.full((src.size(0), 1), bos_idx, dtype=torch.long, device=src.device)
        for _ in range(max_len - 1):
            logits, hc = self.decoder(ys[:, -1:], hc)
            ys = torch.cat([ys, logits[:, -1].argmax(-1, keepdim=True)], dim=1)
        return ys
