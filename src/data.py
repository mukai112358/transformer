from collections import Counter
import re

import torch
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence
from datasets import load_dataset

# 特殊トークン
PAD_IDX = 0
BOS_IDX = 1
EOS_IDX = 2
UNK_IDX = 3
SPECIALS = ['<pad>', '<bos>', '<eos>', '<unk>']


# 単語と記号に分けるだけのシンプル tokenizer
_TOK_RE = re.compile(r'\w+|[^\w\s]')


def simple_tokenize(text):
    return _TOK_RE.findall(text.lower())


class Vocabulary:
    """train データから単語 → id の対応表を作る."""

    def __init__(self, token_iter, min_freq=2):
        counter = Counter()
        for tokens in token_iter:
            counter.update(tokens)
        self.itos = SPECIALS + [tok for tok, c in counter.most_common() if c >= min_freq]
        self.stoi = {tok: i for i, tok in enumerate(self.itos)}

    def __len__(self):
        return len(self.itos)

    def encode(self, tokens):
        """単語列 → id 列. 先頭 BOS、末尾 EOS、未知語は UNK."""
        return [BOS_IDX] + [self.stoi.get(t, UNK_IDX) for t in tokens] + [EOS_IDX]


def load_pairs(split):
    """HuggingFace datasets から Multi30k を読み込み (独tokens, 英tokens) のペア列にする."""
    ds = load_dataset('bentrevett/multi30k', split=split)
    return [(simple_tokenize(ex['de']), simple_tokenize(ex['en'])) for ex in ds]


def collate_fn(batch):
    """バッチ内で一番長い文に合わせて pad で揃える."""
    src_batch, tgt_batch = zip(*batch)
    src_pad = pad_sequence(src_batch, batch_first=True, padding_value=PAD_IDX)
    tgt_pad = pad_sequence(tgt_batch, batch_first=True, padding_value=PAD_IDX)
    return src_pad, tgt_pad


def get_dataloaders(batch_size=64, min_freq=2, num_workers=0):
    """train/val/test の DataLoader と src/tgt の語彙をまとめて返す."""
    train_pairs = load_pairs('train')
    val_pairs   = load_pairs('validation')
    test_pairs  = load_pairs('test')

    # 語彙は train だけから作る (val/test の未知語は UNK 扱い)
    src_vocab = Vocabulary((s for s, _ in train_pairs), min_freq=min_freq)
    tgt_vocab = Vocabulary((t for _, t in train_pairs), min_freq=min_freq)

    def to_tensors(pairs):
        return [(torch.tensor(src_vocab.encode(s)), torch.tensor(tgt_vocab.encode(t)))
                for s, t in pairs]

    common = dict(batch_size=batch_size, collate_fn=collate_fn, num_workers=num_workers)
    return {
        'train_loader': DataLoader(to_tensors(train_pairs), shuffle=True,  **common),
        'val_loader':   DataLoader(to_tensors(val_pairs),   shuffle=False, **common),
        'test_loader':  DataLoader(to_tensors(test_pairs),  shuffle=False, **common),
        'src_vocab': src_vocab,
        'tgt_vocab': tgt_vocab,
    }
