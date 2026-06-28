"""BLEU / 文長別精度 / 推論 (Greedy Decode)."""

import math

import torch

from src.data import PAD_IDX, BOS_IDX, EOS_IDX


def greedy_decode_transformer(model, src, bos_idx=BOS_IDX, eos_idx=EOS_IDX, max_len=50):
    """Encoder を1回だけ通して memory にしてから、Decoder を1トークンずつ進める."""
    model.eval()
    device = src.device

    src_mask = model.make_src_mask(src)
    src_emb = model.src_embedding(src) * math.sqrt(model.d_model)
    src_emb = model.pos_encoding(src_emb)
    memory = model.encoder(src_emb, src_mask)

    batch = src.size(0)
    ys = torch.full((batch, 1), bos_idx, dtype=torch.long, device=device)
    finished = torch.zeros(batch, dtype=torch.bool, device=device)
    for _ in range(max_len - 1):
        tgt_mask = model.make_tgt_mask(ys)
        tgt_emb = model.tgt_embedding(ys) * math.sqrt(model.d_model)
        tgt_emb = model.pos_encoding(tgt_emb)
        out = model.decoder(tgt_emb, memory, src_mask, tgt_mask)
        next_tok = model.generator(out[:, -1]).argmax(-1, keepdim=True)
        ys = torch.cat([ys, next_tok], dim=1)
        finished = finished | (next_tok.squeeze(1) == eos_idx)
        if finished.all():
            break
    return ys


def decode_ids_to_text(ids, vocab):
    out = []
    for i in ids:
        i = int(i)
        if i == EOS_IDX:
            break
        if i in (PAD_IDX, BOS_IDX):
            continue
        out.append(vocab.itos[i] if 0 <= i < len(vocab.itos) else "<unk>")
    return " ".join(out)


@torch.no_grad()
def translate_dataset(model, loader, tgt_vocab, model_type, device, max_len=50):
    model.eval()
    hyps, refs, src_lens = [], [], []
    for src, tgt in loader:
        src = src.to(device)
        if model_type == "transformer":
            gen = greedy_decode_transformer(model, src, max_len=max_len)
        elif model_type == "lstm":
            gen = model.generate(src, BOS_IDX, EOS_IDX, max_len=max_len)
        else:
            raise ValueError(model_type)
        for i in range(src.size(0)):
            hyps.append(decode_ids_to_text(gen[i].cpu().tolist(), tgt_vocab))
            refs.append(decode_ids_to_text(tgt[i].cpu().tolist(), tgt_vocab))
            # src 長は <bos>/<eos> を除いて数える
            src_lens.append(int((src[i] != PAD_IDX).sum().item()) - 2)
    return hyps, refs, src_lens


def compute_bleu(hyps, refs):
    import sacrebleu
    return sacrebleu.corpus_bleu(hyps, [refs]).score


def bleu_by_length(hyps, refs, src_lens, buckets=((1, 5), (6, 10), (11, 15), (16, 99))):
    results = []
    for lo, hi in buckets:
        idx = [i for i, L in enumerate(src_lens) if lo <= L <= hi]
        if not idx:
            results.append({"range": f"{lo}-{hi}", "n": 0, "bleu": None})
            continue
        b_hyps = [hyps[i] for i in idx]
        b_refs = [refs[i] for i in idx]
        results.append({
            "range": f"{lo}-{hi}",
            "n": len(idx),
            "bleu": compute_bleu(b_hyps, b_refs),
        })
    return results
