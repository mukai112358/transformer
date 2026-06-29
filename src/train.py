import torch
import torch.nn as nn


def train_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total = 0
    for src, tgt in loader:
        src, tgt = src.to(device), tgt.to(device)
        tgt_in, tgt_out = tgt[:, :-1], tgt[:, 1:]
        optimizer.zero_grad()
        logits = model(src, tgt_in)
        loss = criterion(logits.reshape(-1, logits.size(-1)), tgt_out.reshape(-1))
        loss.backward()
        optimizer.step()
        total += loss.item()
    return total / len(loader)


@torch.no_grad()
def validate(model, loader, criterion, device):
    model.eval()
    total = 0
    for src, tgt in loader:
        src, tgt = src.to(device), tgt.to(device)
        tgt_in, tgt_out = tgt[:, :-1], tgt[:, 1:]
        logits = model(src, tgt_in)
        loss = criterion(logits.reshape(-1, logits.size(-1)), tgt_out.reshape(-1))
        total += loss.item()
    return total / len(loader)


def run_training(model, train_loader, val_loader, epochs=20, lr=5e-4, device='cpu'):
    criterion = nn.CrossEntropyLoss(ignore_index=0)
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
