import gzip
import re
import urllib.request
from collections import Counter
from pathlib import Path

import torch
from torch.utils.data import Dataset, DataLoader

# 特殊トークン
PAD_IDX = 0
BOS_IDX = 1
EOS_IDX = 2
UNK_IDX = 3
SPECIALS = ['<pad>', '<bos>', '<eos>', '<unk>']

# Multi30k は GitHub に置かれているので raw URL で取得
BASE_URL = 'https://raw.githubusercontent.com/multi30k/dataset/master/data/task1/raw/'
FILES = {
    'train': ('train.de.gz', 'train.en.gz'),
    'val':   ('val.de.gz',   'val.en.gz'),
    'test':  ('test_2016_flickr.de.gz', 'test_2016_flickr.en.gz'),
}


def download_multi30k(data_dir='data/multi30k'):
    """Multi30k の独/英 txt を取得して解凍 (既にあればスキップ)."""
    data_dir = Path(data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)
    paths = {}
    for split, (de_gz, en_gz) in FILES.items():
        split_paths = {}
        for gz, lang in ((de_gz, 'de'), (en_gz, 'en')):
            txt_path = data_dir / gz[:-3]                      # .gz を外した txt パス
            if not txt_path.exists():
                gz_path = data_dir / gz
                print(f'Downloading {gz} ...')
                urllib.request.urlretrieve(BASE_URL + gz, gz_path)
                with gzip.open(gz_path, 'rb') as f_in, open(txt_path, 'wb') as f_out:
                    f_out.write(f_in.read())
            split_paths[lang] = str(txt_path)
        paths[split] = split_paths
    return paths


# 単語と記号に分けるだけのシンプル tokenizer
_TOK_RE = re.compile(r'\w+|[^\w\s]', re.UNICODE)


def simple_tokenize(text):
    return _TOK_RE.findall(text.lower())


class Vocabulary:
    """train データから単語 → id の対応表を作る."""

    def __init__(self, token_iter, min_freq=2):
        # 全文の単語頻度を数える
        counter = Counter()
        for tokens in token_iter:
            counter.update(tokens)
        # 頻度順 (同頻度はアルファベット順) で並べて min_freq 以上を残す
        self.itos = list(SPECIALS) + [
            tok for tok, c in sorted(counter.items(), key=lambda x: (-x[1], x[0]))
            if c >= min_freq and tok not in SPECIALS
        ]
        self.stoi = {tok: i for i, tok in enumerate(self.itos)}

    def __len__(self):
        return len(self.itos)

    def encode(self, tokens, add_bos_eos=False):
        """単語列 → id 列. 未知語は UNK."""
        ids = [self.stoi.get(t, UNK_IDX) for t in tokens]
        if add_bos_eos:
            ids = [BOS_IDX] + ids + [EOS_IDX]
        return ids


def load_pairs(de_path, en_path):
    """1行1文の独/英ファイルを読んで (独tokens, 英tokens) のペア列にする."""
    with open(de_path, encoding='utf-8') as f:
        de_lines = [simple_tokenize(l.strip()) for l in f if l.strip()]
    with open(en_path, encoding='utf-8') as f:
        en_lines = [simple_tokenize(l.strip()) for l in f if l.strip()]
    return list(zip(de_lines, en_lines))


class Multi30kDataset(Dataset):
    def __init__(self, pairs, src_vocab, tgt_vocab):
        self.pairs = pairs
        self.src_vocab = src_vocab
        self.tgt_vocab = tgt_vocab

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        src, tgt = self.pairs[idx]
        src_ids = self.src_vocab.encode(src, add_bos_eos=True)
        tgt_ids = self.tgt_vocab.encode(tgt, add_bos_eos=True)
        return torch.tensor(src_ids), torch.tensor(tgt_ids)


def collate_fn(batch):
    """バッチ内で一番長い文に合わせて pad で揃える."""
    src_batch, tgt_batch = zip(*batch)
    src_max = max(s.size(0) for s in src_batch)
    tgt_max = max(t.size(0) for t in tgt_batch)
    src_pad = torch.full((len(batch), src_max), PAD_IDX, dtype=torch.long)
    tgt_pad = torch.full((len(batch), tgt_max), PAD_IDX, dtype=torch.long)
    for i, (s, t) in enumerate(zip(src_batch, tgt_batch)):
        src_pad[i, :s.size(0)] = s
        tgt_pad[i, :t.size(0)] = t
    return src_pad, tgt_pad


def get_dataloaders(batch_size=64, data_dir='data/multi30k', min_freq=2, num_workers=0):
    """train/val/test の DataLoader と src/tgt の語彙をまとめて返す."""
    paths = download_multi30k(data_dir)
    train_pairs = load_pairs(paths['train']['de'], paths['train']['en'])
    val_pairs   = load_pairs(paths['val']['de'],   paths['val']['en'])
    test_pairs  = load_pairs(paths['test']['de'],  paths['test']['en'])

    # 語彙は train だけから作る (val/test の未知語は UNK 扱い)
    src_vocab = Vocabulary((s for s, _ in train_pairs), min_freq=min_freq)
    tgt_vocab = Vocabulary((t for _, t in train_pairs), min_freq=min_freq)

    return {
        'train_loader': DataLoader(Multi30kDataset(train_pairs, src_vocab, tgt_vocab),
                                   batch_size=batch_size, shuffle=True,
                                   collate_fn=collate_fn, num_workers=num_workers),
        'val_loader':   DataLoader(Multi30kDataset(val_pairs, src_vocab, tgt_vocab),
                                   batch_size=batch_size, shuffle=False,
                                   collate_fn=collate_fn, num_workers=num_workers),
        'test_loader':  DataLoader(Multi30kDataset(test_pairs, src_vocab, tgt_vocab),
                                   batch_size=batch_size, shuffle=False,
                                   collate_fn=collate_fn, num_workers=num_workers),
        'src_vocab': src_vocab,
        'tgt_vocab': tgt_vocab,
    }
