# Transformer from Scratch

Vaswani et al. (2017) "Attention Is All You Need" の Transformer を PyTorch で実装し、Multi30k 独→英の翻訳タスクで LSTM Seq2Seq と比較。

- `src/` 実装本体
- `notebooks/` 各モジュールの動作確認 (01〜07)、訓練 (10)、評価 (11)
- `notes/formula_to_code.md` 数式とコードの対応
- `results/comparison.md` 比較結果

## 結果

Multi30k 独→英 テストセットでの BLEU:

- Transformer: **36.08**
- LSTM Seq2Seq (Attention 無し): 20.83

文長が伸びるほど Transformer の優位が拡大 (+5 → +12.9 → +16.1 → +16.3)。詳細は [results/comparison.md](results/comparison.md) を参照。

## 環境

```
pip install -r requirements.txt
```

訓練は `notebooks/10_train_compare.ipynb` を Colab GPU で、評価は `notebooks/11_evaluation.ipynb` をローカルで実行。
