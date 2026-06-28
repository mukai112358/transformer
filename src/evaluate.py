import torch

from src.data import PAD_IDX, BOS_IDX, EOS_IDX


def decode_ids_to_text(ids, vocab):
    """token id 列を単語の文字列に戻す. PAD/BOS は無視、EOS で打ち切り."""
    words = []
    for i in ids:
        i = int(i)
        if i == EOS_IDX:
            break
        if i in (PAD_IDX, BOS_IDX):
            continue
        words.append(vocab.itos[i] if 0 <= i < len(vocab.itos) else '<unk>')
    return ' '.join(words)


@torch.no_grad()
def translate_dataset(model, loader, tgt_vocab, device, max_len=50):
    """loader 全件を Greedy Decode で翻訳して (生成文, 参照文, src 長) を返す.

    model は greedy_decode(src, max_len=...) を持つ前提 (Transformer/LSTM とも実装済み).
    """
    model.eval()
    hyps, refs, src_lens = [], [], []
    for src, tgt in loader:
        src = src.to(device)
        gen = model.greedy_decode(src, max_len=max_len)
        for i in range(src.size(0)):
            hyps.append(decode_ids_to_text(gen[i].cpu().tolist(), tgt_vocab))
            refs.append(decode_ids_to_text(tgt[i].cpu().tolist(), tgt_vocab))
            # src の長さ (BOS/EOS を除く)
            src_lens.append(int((src[i] != PAD_IDX).sum().item()) - 2)
    return hyps, refs, src_lens


def compute_bleu(hyps, refs):
    """sacrebleu の corpus BLEU を返す."""
    import sacrebleu
    return sacrebleu.corpus_bleu(hyps, [refs]).score


def bleu_by_length(hyps, refs, src_lens, buckets=((1, 5), (6, 10), (11, 15), (16, 99))):
    """src の文長別に BLEU を計算する."""
    results = []
    for lo, hi in buckets:
        idx = [i for i, L in enumerate(src_lens) if lo <= L <= hi]
        if not idx:
            results.append({'range': f'{lo}-{hi}', 'n': 0, 'bleu': None})
            continue
        b_hyps = [hyps[i] for i in idx]
        b_refs = [refs[i] for i in idx]
        results.append({
            'range': f'{lo}-{hi}',
            'n': len(idx),
            'bleu': compute_bleu(b_hyps, b_refs),
        })
    return results
