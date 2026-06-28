# Transformer from Scratch

Vaswani et al. (2017) "Attention Is All You Need" の Transformer を PyTorch で実装し、Multi30k 独→英の翻訳タスクで LSTM Seq2Seq と比較した。

## 構成

- `src/` Transformer と LSTM の実装、データ準備、訓練、評価
- `notebooks/` 各モジュールの動作確認、訓練、評価
- `notes/` 数式とコードの対応
- `results/` 比較結果、損失曲線、BLEU グラフ

## 環境

Python 3.10+、PyTorch。Multi30k は `src/data.py` が自動ダウンロード。

```
pip install -r requirements.txt
```

## 実行

訓練は `notebooks/10_train_compare.ipynb` を Colab GPU で実行。重みと履歴が `results/` に保存される。評価は `notebooks/11_evaluation.ipynb` をローカルで実行し、BLEU とグラフを生成。

## 結果

Multi30k 独→英のテストセットでの BLEU:

- Transformer: **36.08**
- LSTM Seq2Seq (Attention 無し): 20.83

文長別 BLEU では source が長くなるほど Transformer の優位が拡大(+5 → +12.9 → +16.1 → +16.3)。

詳細は [results/comparison.md](results/comparison.md) を参照。
