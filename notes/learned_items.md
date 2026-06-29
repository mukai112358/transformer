# 習得項目

まだ作業途中。訓練と比較が終わったら書き直す予定。

---

## 実装したもの

- Transformer を実装。Multi-Head Attention (内部に Scaled Dot-Product Attention) / Position-wise FFN / Positional Encoding / Encoder / Decoder を src/ 配下に分割
- LSTM Seq2Seq (Attention 無し) のベースラインを src/lstm_baseline.py に
- Multi30k 独→英のデータ取得・tokenize・語彙構築・DataLoader を src/data.py にまとめた
- 両モデル共通の訓練ループを src/train.py に統一 (Teacher Forcing, CrossEntropyLoss, Adam, gradient clipping)
- BLEU と文長別精度を計算する評価コードを src/evaluate.py に
- Greedy Decode は Transformer と LSTMSeq2Seq それぞれの greedy_decode メソッドとして実装

## 触った PyTorch

- nn.Module の継承と forward の書き方
- nn.Linear / nn.LayerNorm / nn.Embedding / nn.LSTM / nn.Dropout / nn.ModuleList
- テンソル操作: transpose, reshape, matmul / @, masked_fill, unsqueeze
- register_buffer で学習しない定数テンソルを持つ
- DataLoader / Dataset / collate_fn でバッチを作る

## 開発まわり

- venv で環境を切る
- Git で意味のある単位でコミット
- VSCode + Jupyter で .ipynb 編集
