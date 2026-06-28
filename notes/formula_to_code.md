# 数式とコードの対応

論文「Attention Is All You Need」の数式と `src/` 実装の対応メモ。

---

## Scaled Dot-Product Attention

論文 (Section 3.2.1, 式 1):

```
Attention(Q, K, V) = softmax(QK^T / √d_k) V
```

`src/attention.py:scaled_dot_product_attention`

実装は Multi-Head Attention から呼ぶ前提で 4D テンソル `(batch, heads, seq_len, d_k)` を扱う。
論文の式自体は次元に依存しない(`matmul` は先頭次元をブロードキャストする)ので、先頭に `heads` が増えても同じ式が成り立つ。

| 数式 | コード | 形状 | 意味 |
|---|---|---|---|
| Q | `Q` | (batch, heads, seq_q, d_k) | クエリ |
| K | `K` | (batch, heads, seq_k, d_k) | キー |
| V | `V` | (batch, heads, seq_k, d_v) | バリュー |
| K^T | `K.transpose(-2, -1)` | (batch, heads, d_k, seq_k) | キーの転置(最後2次元) |
| QK^T | `torch.matmul(Q, K.transpose(-2, -1))` | (batch, heads, seq_q, seq_k) | クエリとキーの内積 |
| √d_k | `math.sqrt(d_k)` | scalar | スケーリング係数 |
| (mask 適用) | `scores.masked_fill(mask == 0, float('-inf'))` | (batch, heads, seq_q, seq_k) | 未来トークンやパディングを無視 |
| softmax(...) | `F.softmax(scores, dim=-1)` | (batch, heads, seq_q, seq_k) | アテンション重み |
| softmax(...) V | `torch.matmul(attn_weights, V)` | (batch, heads, seq_q, d_v) | 出力 |

**コード抜粋:**

```python
def scaled_dot_product_attention(Q, K, V, mask=None):
    d_k = Q.size(-1)
    scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(d_k)
    if mask is not None:
        scores = scores.masked_fill(mask == 0, float('-inf'))
    attn_weights = F.softmax(scores, dim=-1)
    output = torch.matmul(attn_weights, V)
    return output, attn_weights
```

---

## Multi-Head Attention

論文 (Section 3.2.2, 式 2):

```
MultiHead(Q, K, V) = Concat(head_1, ..., head_h) W^O
  where head_i = Attention(Q W_i^Q, K W_i^K, V W_i^V)
```

`src/attention.py:MultiHeadAttention`

| 数式 | コード | 形状 | 意味 |
|---|---|---|---|
| W_i^Q, W_i^K, W_i^V | `self.W_q`, `self.W_k`, `self.W_v` | (d_model, d_model) | 線形射影。実装では h 個まとめて1つの大きな nn.Linear で持つ |
| h | `self.num_heads` | scalar | head 数 |
| d_k | `self.d_k = d_model // num_heads` | scalar | 各 head の次元 |
| head_i = Attention(...) | `reshape` で head 分割 → `scaled_dot_product_attention` | (batch, heads, seq, d_k) | 各 head 独立に Attention 計算 |
| Concat(...) | `context.transpose(1, 2).reshape(batch_size, -1, self.d_model)` | (batch, seq, d_model) | head を結合 |
| W^O | `self.W_o` | (d_model, d_model) | 出力射影 |

---

## Position-wise Feed-Forward Network

論文 (Section 3.3, 式 2):

```
FFN(x) = max(0, xW_1 + b_1) W_2 + b_2
```

`src/feed_forward.py:PositionwiseFeedForward`

| 数式 | コード | 形状 | 意味 |
|---|---|---|---|
| W_1, b_1 | `self.linear1` | (d_model, d_ff) | 1段目の線形層(次元拡大) |
| max(0, ·) | `self.activation` (GELU) | - | 非線形活性化(論文 ReLU、実装 GELU) |
| W_2, b_2 | `self.linear2` | (d_ff, d_model) | 2段目の線形層(次元復元) |

---

## Positional Encoding

論文 (Section 3.5):

```
PE(pos, 2i)   = sin(pos / 10000^(2i/d_model))
PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))
```

`src/positional_encoding.py:PositionalEncoding`

| 数式 | コード | 形状 | 意味 |
|---|---|---|---|
| pos | `position = torch.arange(0, max_seq).unsqueeze(1)` | (max_seq, 1) | 位置インデックス |
| 1 / 10000^(2i/d_model) | `div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))` | (d_model/2,) | 周波数項(対数化して計算) |
| PE(pos, 2i) | `pe[:, 0::2] = torch.sin(position * div_term)` | (max_seq, d_model/2) | 偶数列に sin |
| PE(pos, 2i+1) | `pe[:, 1::2] = torch.cos(position * div_term)` | (max_seq, d_model/2) | 奇数列に cos |
| 入力との加算 | `x + self.pe[:, :x.size(1), :]` | (batch, seq, d_model) | Embedding に位置情報を足す |

---

## Encoder Layer

論文の構造: Self-Attention → 残差 + LayerNorm → FFN → 残差 + LayerNorm を N 層スタック。

`src/encoder.py:EncoderLayer`, `src/encoder.py:Encoder`

| 数式 | コード | 形状 |
|---|---|---|
| Self-Attention | `self.self_attn(x, x, x, src_mask)` | (batch, seq, d_model) |
| 残差 + LayerNorm 1 | `self.norm1(x + self.dropout1(attn_output))` | (batch, seq, d_model) |
| FFN | `self.ffn(x)` | (batch, seq, d_model) |
| 残差 + LayerNorm 2 | `self.norm2(x + self.dropout2(ffn_output))` | (batch, seq, d_model) |
| N 層スタック | `self.layers = nn.ModuleList([EncoderLayer(...) for _ in range(N)])` | - |

---

## Decoder Layer

論文の構造: Masked Self-Attention → 残差 + LayerNorm → Cross-Attention → 残差 + LayerNorm → FFN → 残差 + LayerNorm を N 層スタック。

`src/decoder.py:DecoderLayer`, `src/decoder.py:Decoder`

| 数式 | コード | 形状 |
|---|---|---|
| Masked Self-Attention | `self.self_attn(x, x, x, tgt_mask)` | (batch, tgt_len, d_model) |
| 残差 + LayerNorm 1 | `self.norm1(x + self.dropout1(attn_output))` | (batch, tgt_len, d_model) |
| Cross-Attention | `self.cross_attn(x, enc_output, enc_output, src_mask)` | (batch, tgt_len, d_model) |
| 残差 + LayerNorm 2 | `self.norm2(x + self.dropout2(attn_output))` | (batch, tgt_len, d_model) |
| FFN | `self.ffn(x)` | (batch, tgt_len, d_model) |
| 残差 + LayerNorm 3 | `self.norm3(x + self.dropout3(ffn_output))` | (batch, tgt_len, d_model) |
| Causal mask | `torch.tril(torch.ones(seq, seq)).unsqueeze(0).unsqueeze(0)` | (1, 1, seq, seq) |

---

## Transformer 全体

`src/transformer.py:Transformer`

| 数式 | コード | 形状 |
|---|---|---|
| src Embedding × √d_model | `self.src_embedding(src) * math.sqrt(self.d_model)` | (batch, src_len, d_model) |
| + Positional Encoding | `self.pos_encoding(src_emb)` | (batch, src_len, d_model) |
| Encoder | `self.encoder(src_emb, src_mask)` | (batch, src_len, d_model) |
| tgt Embedding × √d_model + PE | (上と同じ手順を target 側に) | (batch, tgt_len, d_model) |
| Decoder | `self.decoder(tgt_emb, enc_output, src_mask, tgt_mask)` | (batch, tgt_len, d_model) |
| Generator (Linear) | `self.generator` | d_model → tgt_vocab_size |
| logits | 出力 | (batch, tgt_len, tgt_vocab_size) |

**マスク:**

| マスク | コード | 形状 |
|---|---|---|
| src_mask (パディング) | `(src != pad_idx).unsqueeze(1).unsqueeze(2)` | (batch, 1, 1, src_len) |
| tgt_mask (パディング & causal) | `pad_mask & torch.tril(torch.ones(tgt_len, tgt_len)).bool()` | (batch, 1, tgt_len, tgt_len) |
