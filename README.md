# Transformer from Scratch

「Attention Is All You Need」(Vaswani et al., 2017) をPyTorchでスクラッチ実装し、LSTMベースラインと性能比較するプロジェクト。

## 状態

実装中

## ディレクトリ構成

```
transformer-from-scratch/
├── README.md            プロジェクト概要(本ファイル)
├── requirements.txt     依存パッケージ
├── notes/               学習の記録
│   ├── learning_log.md     毎日の学習ログ
│   ├── formula_to_code.md  論文の数式とコードの対応
│   ├── stuck_points.md     詰まった点と解決の記録
│   └── learned_items.md    習得項目リスト(完成後)
├── src/                 実装コード本体
├── notebooks/           Jupyter Notebook(動作確認・可視化)
└── results/             実験結果
    └── comparison.md       LSTMとの性能比較レポート
```

## 使用技術

- Python 3.10+
- PyTorch (Transformer・LSTM 両方の実装に使用)
- NumPy
- Matplotlib (損失曲線・Attention可視化)
- Jupyter

## 実装方針

- `nn.Transformer` / `nn.MultiheadAttention` は使用せず、`nn.Linear` / `nn.LayerNorm` 等の基本部品から自作
- LSTMは比較対象(ベースライン)なので `nn.LSTM` を使用
- 訓練はGoogle Colabを使用予定(ローカルはCPUのみ)
