import gzip
import re
import urllib.request
from collections import Counter
from pathlib import Path

import torch
from torch.utils.data import Dataset, DataLoader

PAD_IDX = 0
BOS_IDX = 1
EOS_IDX = 2
UNK_IDX = 3
SPECIALS = ["<pad>", "<bos>", "<eos>", "<unk>"]

BASE_URL = "https://raw.githubusercontent.com/multi30k/dataset/master/data/task1/raw/"
FILES = {
    "train": ("train.de.gz", "train.en.gz"),
    "val":   ("val.de.gz",   "val.en.gz"),
    "test":  ("test_2016_flickr.de.gz", "test_2016_flickr.en.gz"),
}


def download_multi30k(data_dir="data/multi30k"):
    data_dir = Path(data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)
    paths = {}
    for split, (de_gz, en_gz) in FILES.items():
        split_paths = {}
        for gz, lang in ((de_gz, "de"), (en_gz, "en")):
            gz_path = data_dir / gz
            txt_path = data_dir / gz[:-3]
            if not txt_path.exists():
                if not gz_path.exists():
                    print(f"Downloading {gz} ...")
                    urllib.request.urlretrieve(BASE_URL + gz, gz_path)
                with gzip.open(gz_path, "rb") as f_in, open(txt_path, "wb") as f_out:
                    f_out.write(f_in.read())
            split_paths[lang] = str(txt_path)
        paths[split] = split_paths
    return paths


_TOK_RE = re.compile(r"\w+|[^\w\s]", re.UNICODE)


def simple_tokenize(text):
    return _TOK_RE.findall(text.lower())


class Vocabulary:
    def __init__(self, token_iter, min_freq=2):
        counter = Counter()
        for tokens in token_iter:
            counter.update(tokens)
        self.itos = list(SPECIALS) + [
            tok for tok, c in sorted(counter.items(), key=lambda x: (-x[1], x[0]))
            if c >= min_freq and tok not in SPECIALS
        ]
        self.stoi = {tok: i for i, tok in enumerate(self.itos)}

    def __len__(self):
        return len(self.itos)

    def encode(self, tokens, add_bos_eos=False):
        ids = [self.stoi.get(t, UNK_IDX) for t in tokens]
        if add_bos_eos:
            ids = [BOS_IDX] + ids + [EOS_IDX]
        return ids

    def decode(self, ids):
        return [self.itos[i] if 0 <= i < len(self.itos) else "<unk>" for i in ids]


def load_pairs(de_path, en_path):
    with open(de_path, encoding="utf-8") as f:
        de_lines = [simple_tokenize(l.strip()) for l in f if l.strip()]
    with open(en_path, encoding="utf-8") as f:
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
        return (
            torch.tensor(src_ids, dtype=torch.long),
            torch.tensor(tgt_ids, dtype=torch.long),
        )


def collate_fn(batch):
    src_batch, tgt_batch = zip(*batch)
    src_max = max(s.size(0) for s in src_batch)
    tgt_max = max(t.size(0) for t in tgt_batch)
    src_pad = torch.full((len(batch), src_max), PAD_IDX, dtype=torch.long)
    tgt_pad = torch.full((len(batch), tgt_max), PAD_IDX, dtype=torch.long)
    for i, (s, t) in enumerate(zip(src_batch, tgt_batch)):
        src_pad[i, :s.size(0)] = s
        tgt_pad[i, :t.size(0)] = t
    return src_pad, tgt_pad


def get_dataloaders(batch_size=64, data_dir="data/multi30k", min_freq=2, num_workers=0):
    paths = download_multi30k(data_dir)
    train_pairs = load_pairs(paths["train"]["de"], paths["train"]["en"])
    val_pairs   = load_pairs(paths["val"]["de"],   paths["val"]["en"])
    test_pairs  = load_pairs(paths["test"]["de"],  paths["test"]["en"])

    src_vocab = Vocabulary((s for s, _ in train_pairs), min_freq=min_freq)
    tgt_vocab = Vocabulary((t for _, t in train_pairs), min_freq=min_freq)

    train_set = Multi30kDataset(train_pairs, src_vocab, tgt_vocab)
    val_set   = Multi30kDataset(val_pairs,   src_vocab, tgt_vocab)
    test_set  = Multi30kDataset(test_pairs,  src_vocab, tgt_vocab)

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True,
                              collate_fn=collate_fn, num_workers=num_workers)
    val_loader   = DataLoader(val_set,   batch_size=batch_size, shuffle=False,
                              collate_fn=collate_fn, num_workers=num_workers)
    test_loader  = DataLoader(test_set,  batch_size=batch_size, shuffle=False,
                              collate_fn=collate_fn, num_workers=num_workers)
    return {
        "train_loader": train_loader,
        "val_loader": val_loader,
        "test_loader": test_loader,
        "src_vocab": src_vocab,
        "tgt_vocab": tgt_vocab,
    }
