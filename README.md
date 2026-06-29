# Transformer from Scratch

## 目的

Transformerは主要なLLMに共通する基盤であり、RAGやプロンプト設計の土台でもある。APIの利用者にとどまらず内部を説明できる理解を目的に、論文「Attention Is All You Need」(Vaswani et al., 2017) をPyTorchでゼロから実装した。各モジュール (Attention、Encoder、Decoder など) を一つずつ実装し、内部のテンソルの流れと数式の対応を腑に落とした。
最後に Multi30k 独→英の翻訳タスクで LSTM Seq2Seq と比較し、並列性や長距離依存への強さといった Transformer の理論的特徴が、実測値にも現れるかを確認した。

## 結果

Multi30k のテストセット (1,000 文) で評価した BLEU は以下のとおり。

| モデル | パラメータ数 | BLEU |
|---|---|---|
| Transformer | 10.6M | **36.08** |
| LSTM Seq2Seq (Attention 無し) | 10.2M | 20.83 |

文長別に BLEU を見ると、原文が短いとき (1〜5 単語) では Transformer の優位は約 5 ポイントだったのに対し、長くなる (16 単語以上) と 16 ポイント以上に拡大した。LSTM が長距離依存を扱いにくいという理論的な弱点が、実測値にもはっきり現れている。

![](results/loss_curve.png)
![](results/bleu_by_length.png)

詳しい数字と訳例は [results/comparison.md](results/comparison.md) を参照。

## 実装の流れ

論文を読みながら、Scaled Dot-Product Attention から順に Multi-Head Attention、Position-wise FFN、Positional Encoding、Encoder、Decoder、Transformer 全体まで、各モジュールを実装していった (`notebooks/01〜07`)。動作確認が終わったものは `src/` 配下に整理。

比較対象の LSTM Seq2Seq は `nn.LSTM` を使って Encoder と Decoder を組み立てた。

データの取得は HuggingFace の `datasets`、BLEU 計算は `sacrebleu`、padding は `torch.nn.utils.rnn.pad_sequence` を使った。訓練は Colab GPU、評価とグラフはローカル環境で行った。

## 構成

- `src/` 実装本体 (Transformer / LSTM / データ準備 / 訓練 / 評価)
- `notebooks/01〜07` 各モジュールの実装と動作確認
- `notebooks/10_train_compare.ipynb` 訓練
- `notebooks/11_evaluation.ipynb` 評価 (BLEU 計算と図の生成)
- `notes/formula_to_code.md` 論文の数式とコードの対応
- `results/comparison.md` 比較結果のまとめ

## 参考

- Vaswani et al. (2017) ["Attention Is All You Need"](https://arxiv.org/abs/1706.03762)
- [disassemble-channel — Transformer をゼロから実装する](https://disassemble-channel.com/transformer-from-scratch/)
