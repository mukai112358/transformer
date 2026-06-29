import torch
import torch.nn as nn


def train_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss, total_tokens = 0.0, 0
    for src, tgt in loader:
        src, tgt = src.to(device), tgt.to(device)
        tgt_in, tgt_out = tgt[:, :-1], tgt[:, 1:]
        optimizer.zero_grad()
        logits = model(src, tgt_in)
        loss = criterion(logits.reshape(-1, logits.size(-1)), tgt_out.reshape(-1))
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        n_tok = (tgt_out != 0).sum().item()
        total_loss += loss.item() * n_tok
        total_tokens += n_tok
    return total_loss / max(total_tokens, 1)


@torch.no_grad()
def validate(model, loader, criterion, device):
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
    criterion = nn.CrossEntropyLoss(ignore_index=pad_idx)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    history = {'train_loss': [], 'val_loss': []}
    model.to(device)
    for ep in range(1, epochs + 1):
        tr = train_epoch(model, train_loader, criterion, optimizer, device)
        vl = validate(model, val_loader, criterion, device)
        history['train_loss'].append(tr)
        history['val_loss'].append(vl)
        print(f'epoch {ep:02d} | train {tr:.4f} | val {vl:.4f}')
    return history
