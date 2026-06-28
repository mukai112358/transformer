"""両モデル共通の訓練ループ. model_type で forward の呼び方を切替."""

import time

import torch
import torch.nn as nn


def shift_tgt(tgt):
    """tgt: [BOS, ..., EOS] を 入力(右端を捨てる)/ラベル(左端を捨てる) に分ける."""
    return tgt[:, :-1], tgt[:, 1:]


def _forward(model, src, tgt_in, model_type):
    if model_type in ("transformer", "lstm"):
        return model(src, tgt_in)
    raise ValueError(f"unknown model_type: {model_type}")


def train_epoch(model, loader, criterion, optimizer, device, model_type, clip=1.0):
    model.train()
    total_loss = 0.0
    total_tokens = 0
    for src, tgt in loader:
        src, tgt = src.to(device), tgt.to(device)
        tgt_in, tgt_out = shift_tgt(tgt)
        optimizer.zero_grad()
        logits = _forward(model, src, tgt_in, model_type)
        loss = criterion(logits.reshape(-1, logits.size(-1)), tgt_out.reshape(-1))
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), clip)
        optimizer.step()
        n_tok = (tgt_out != 0).sum().item()
        total_loss += loss.item() * n_tok
        total_tokens += n_tok
    return total_loss / max(total_tokens, 1)


@torch.no_grad()
def validate(model, loader, criterion, device, model_type):
    model.eval()
    total_loss = 0.0
    total_tokens = 0
    for src, tgt in loader:
        src, tgt = src.to(device), tgt.to(device)
        tgt_in, tgt_out = shift_tgt(tgt)
        logits = _forward(model, src, tgt_in, model_type)
        loss = criterion(logits.reshape(-1, logits.size(-1)), tgt_out.reshape(-1))
        n_tok = (tgt_out != 0).sum().item()
        total_loss += loss.item() * n_tok
        total_tokens += n_tok
    return total_loss / max(total_tokens, 1)


def run_training(
    model, train_loader, val_loader, *,
    epochs=20, lr=1e-3, device="cpu", model_type="transformer",
    pad_idx=0, clip=1.0, log_fn=print,
):
    criterion = nn.CrossEntropyLoss(ignore_index=pad_idx)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    history = {"train_loss": [], "val_loss": [], "epoch_time": []}
    model.to(device)
    for ep in range(1, epochs + 1):
        t0 = time.time()
        tr = train_epoch(model, train_loader, criterion, optimizer, device, model_type, clip)
        vl = validate(model, val_loader, criterion, device, model_type)
        elapsed = time.time() - t0
        history["train_loss"].append(tr)
        history["val_loss"].append(vl)
        history["epoch_time"].append(elapsed)
        log_fn(f"[{model_type}] epoch {ep:02d} | train {tr:.4f} | val {vl:.4f} | {elapsed:.1f}s")
    return history
