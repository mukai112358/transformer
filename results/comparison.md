# LSTMとの性能比較

下記は記入テンプレート。訓練完了後に実測値で埋める。

---

## 実験設定

- **タスク:** コピータスク(入力系列をそのまま出力に再現する seq2seq タスク)
- **語彙サイズ:** 20
- **系列長:** 10
- **訓練データ:** 8,000 サンプル(ランダム生成)
- **検証データ:** 1,000 サンプル
- **バッチサイズ:** 64
- **エポック数:** 50
- **最適化:** Adam, lr=1e-3
- **損失関数:** Cross Entropy Loss (パディングを `ignore_index` で除外)
- **環境:** Google Colab (GPU: T4)

## モデル構成

### LSTM (ベースライン)

- Encoder: `nn.LSTM` 1層, hidden=128
- Decoder: `nn.LSTM` 1層, hidden=128 (Teacher Forcing)
- パラメータ数: 約 XXX,XXX ← 実測値を入れる

### Transformer (スクラッチ実装)

- d_model: 64
- num_heads: 4
- num_layers (Encoder, Decoder): 2, 2
- d_ff: 256
- パラメータ数: 約 XXX,XXX ← 実測値を入れる

## 結果

| モデル | 最終訓練Loss | 最終検証Loss | トークン一致率 | 訓練時間 |
|---|---|---|---|---|
| LSTM | X.XX | X.XX | XX.X% | XX分 |
| Transformer | X.XX | X.XX | XX.X% | XX分 |

### 損失曲線

![Loss Curve](loss_curve.png)

→ Transformer は LSTM よりも早い段階で loss が低下している(or 安定している)など、観察した事実を書く

### Attention 可視化(Transformer)

![Attention](attention.png)

→ コピータスクなので入力位置 i と出力位置 i が強く対応する「対角線パターン」が見られるはず

## 考察

- **Transformer が有利だった点:**
  - (例)並列計算により訓練時間が短い
  - (例)長距離依存を直接捉えられる
- **LSTM が善戦/劣った点:**
  - (例)短い系列ではほぼ同等の性能
  - (例)系列が長くなると Transformer との差が広がった
- **Attention パターンの観察:**
  - (例)Decoder の Cross-Attention で対角線パターンが鮮明 = コピータスクに必要な対応関係を正しく学習
  - (例)Encoder Self-Attention は全位置を見る分散的なパターン

## 結論

(2-3行で全体所感を書く)
- 同等のパラメータ数で比較した場合の優劣
- 訓練効率の違い
- Transformer の解釈性(Attention可視化)による優位性
