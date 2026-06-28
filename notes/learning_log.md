# 学習ログ

毎日の学習を「やった / 詰まった / 学び」の3行で記録する。
日付ベース(Day番号ではなく)で書くことで、1日空いても自然に飛ばせる。

---

## 2026-05-29

- やった: プロジェクトの初期セットアップ(ディレクトリ作成、Git初期化)
- 詰まった: なし
- 学び: 成果物としての見せ方を意識して、最初から `notes/` と `results/` を分けておく構成にした

## 2026-05-30 ← 例: 実装1日目

- やった: `notebooks/01_scaled_dot_product_attention.ipynb` で記事のStep1を写経完了。`src/attention.py` に整理
- 詰まった: K の転置を忘れて shape mismatch エラー → stuck_points.md #01 に記録
- 学び: PyTorchの `transpose(-2, -1)` は「最後の2次元の入れ替え」専用イディオム。numpyの `swapaxes` に相当

## 2026-05-31 ← 例: Step2の途中で1日終わったケース

- やった: Multi-Head Attention の写経50%まで。Q,K,V を head 数で分割する処理を実装
- 詰まった: `view(batch, seq, num_heads, d_k)` の後の `transpose(1, 2)` の意図がすぐ理解できなかった
- 学び: 「head次元を batch の次に持ってくる」ことで attention 計算をベクトル化できる

## 2026-06-01 ← 例: 詰まりが多い日

- やった: Multi-Head Attention 完成、`src/attention.py` に整理
- 詰まった: 出力の concat 後 W^O を通す部分で次元を間違える → stuck_points.md #03
- 学び: 「分割 → attention → concat → 線形射影」の4ステップが MHA の本質。残りはshape操作の機械的処理


## 2026-05-29
- やった:　Multi-Head Attentionの実装
- 詰まった: ・viewの使い方（次元を分割できること）
・理論上は各ヘッドごとに別々の重み行列を持つけど実装ではまず次元を落とさずに大きな線形層で射影してその圧reshapeでヘッドに分割すること
- 学び: ・.viewはテンソルの形を変えることができる　・