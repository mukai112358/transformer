import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super().__init__()
        assert d_model % num_heads == 0, "d_modelはnum_headで割り切れる必要があります"
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads

        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)

    def forward(self, Q_input, K_input, V_input, mask=None):
        batch_size = Q_input.size(0)

        Q = self.W_q(Q_input)
        K = self.W_k(K_input)
        V = self.W_v(V_input)

        Q = Q.reshape(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        K = K.reshape(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        V = V.reshape(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)

        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)

        if mask is not None:
            scores = scores.masked_fill(mask == 0, float("-inf"))
        attention_weights = F.softmax(scores, dim=-1)

        context = torch.matmul(attention_weights, V)
        context = context.transpose(1, 2).reshape(batch_size, -1, self.d_model)

        output = self.W_o(context)
        return output
