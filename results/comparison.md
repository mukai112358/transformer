# Transformer vs LSTM Seq2Seq

Multi30k 独→英の翻訳タスクで、Transformer と素の LSTM Encoder-Decoder (Attention 無し) を同条件で訓練し比較した。

## 実験設定

- データ: Multi30k 独→英 (train 29,000 / val 1,014 / test 1,000)
- 語彙: 単語単位、min_freq=2
- 訓練: 20 epoch、batch 64、Adam lr=5e-4、gradient clipping 1.0
- 評価: sacrebleu BLEU、Greedy Decode

## モデル

- Transformer: d_model=256、num_heads=8、d_ff=1024、num_layers=3
- LSTM: embed_dim=256、hidden_dim=320、num_layers=3 (パラメータ数を Transformer に揃えた)

## 結果

| モデル | パラメータ数 | 訓練時間 | val loss | BLEU |
|---|---|---|---|---|
| Transformer | (TBD) | (TBD) | (TBD) | (TBD) |
| LSTM | (TBD) | (TBD) | (TBD) | (TBD) |

### 学習曲線

![](loss_curve.png)

### 文長別 BLEU

![](bleu_by_length.png)

## 訳例

(TBD)

## 考察

(TBD)
