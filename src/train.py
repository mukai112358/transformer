import time

import torch
import torch.nn as nn


def train_epoch(model, loader, criterion, optimizer, device):
    """1 epoch 訓練して、平均 train loss を返す."""
    model.train()
    total_loss, total_tokens = 0.0, 0
    for src, tgt in loader:
        src, tgt = src.to(device), tgt.to(device)
        # Teacher Forcing: 入力 = [BOS, ..., x_{n-1}], ラベル = [x_1, ..., EOS]
        tgt_in  = tgt[:, :-1]
        tgt_out = tgt[:, 1:]

        optimizer.zero_grad()
        logits = model(src, tgt_in)                                       # (batch, tgt_len-1, vocab_size)
        loss = criterion(logits.reshape(-1, logits.size(-1)), tgt_out.reshape(-1))
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        n_tok = (tgt_out != 0).sum().item()                               # pad を除いたトークン数
        total_loss += loss.item() * n_tok
        total_tokens += n_tok
    return total_loss / max(total_tokens, 1)


@torch.no_grad()
def validate(model, loader, criterion, device):
    """val set で平均 loss を返す."""
    model.eval()
    total_loss, total_tokens = 0.0, 0
    for src, tgt in loader:
        src, tgt = src.to(device), tgt.to(device)
        tgt_in, tgt_out = tgt[:, :-1], tgt[:, 1:]
        logits = model(src, tgt_in)
        loss = criterion(logits.reshape(-1, logits.size(-1)), tgt_out.reshape(-1))
        n_tok = (tgt_out != 0).sum().item()
        total_loss += loss.item() * n_tok
        total_tokens += n_tok
    return total_loss / max(total_tokens, 1)


def run_training(model, train_loader, val_loader, *, epochs=20, lr=5e-4, device='cpu', pad_idx=0):
    """epoch ごとに train/val loss を出力し、history を返す.

    両モデル (Transformer / LSTM) で共通に使える: どちらも model(src, tgt_in) を持つ.
    """
    criterion = nn.CrossEntropyLoss(ignore_index=pad_idx)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    history = {'train_loss': [], 'val_loss': [], 'epoch_time': []}
    model.to(device)
    for ep in range(1, epochs + 1):
        t0 = time.time()
        tr = train_epoch(model, train_loader, criterion, optimizer, device)
        vl = validate(model, val_loader, criterion, device)
        elapsed = time.time() - t0
        history['train_loss'].append(tr)
        history['val_loss'].append(vl)
        history['epoch_time'].append(elapsed)
        print(f'epoch {ep:02d} | train {tr:.4f} | val {vl:.4f} | {elapsed:.1f}s')
    return history
