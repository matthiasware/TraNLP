import torch
import torch.nn as nn
#
import numpy as np


class Embedding(nn.Module):
    def __init__(self, d_vocabulary, d_model, d_l):
        """
            d_vocabulary: size of vocabulary
            d_model: dimension of embedding
            d_l: length of input sentences
        """

        super(Embedding, self).__init__()
        self.d_vocabulary = d_vocabulary
        self.d_model = d_model
        #
        self.tok_emb = nn.Embedding(d_vocabulary, d_model)  # token embedding
        self.pos_emb = nn.Embedding(d_l, d_model)  # position embedding
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x):
        seq_len = x.size(1)
        pos = torch.arange(seq_len, dtype=torch.long)
        # (seq_len,) -> (batch_size, seq_len)
        pos = pos.unsqueeze(0).expand_as(x)
        embedding = self.tok_emb(x) + self.pos_emb(pos)
        return self.norm(embedding)


class ScaledDotProductAttention(nn.Module):
    def __init__(self, d_k):
        super(ScaledDotProductAttention, self).__init__()
        self.d_k = d_k

    def forward(self, Q, K, V, attn_mask):
        # without heads:
        # Q (d_b, d_l, d_k)
        # K (d_b, d_l, d_k)
        # V (d_n, d_l, d_v)
        # attn_mask (d_b, d_l, d_l)
        #
        # with heads
        # Q (d_b, n_h, d_l, d_k)
        # K (d_b, n_h, d_l, d_k)
        # V (d_b, n_h, d_l, d_v)
        # attn_mask (d_b, n_h, d_l, d_v)

        # scores | attn (d_b, (d_h), d_l, d_l)
        scores = torch.matmul(Q, K.transpose(-1, -2)) / np.sqrt(self.d_k)
        # Fills elements of self tensor with value where mask is one.
        scores.masked_fill_(attn_mask, -1e9)
        attn = nn.Softmax(dim=-1)(scores)

        # context = (d_b, (d_h), d_l, d_v)
        context = torch.matmul(attn, V)
        return context, attn


class NormedResidualSubLayerConnection(nn.Module):
    def __init__(self, d_model):
        super(NormedResidualSubLayerConnection, self).__init__()
        self.d_model = d_model
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x, sublayer):
        return x + sublayer(self.norm(x))


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, d_k, d_v, n_heads):
        super(MultiHeadAttention, self).__init__()
        #
        self.d_k = d_k
        self.d_v = d_v
        self.d_model = d_model
        self.n_heads = n_heads
        #
        self.W_Q = nn.Linear(d_model, d_k * n_heads)
        self.W_K = nn.Linear(d_model, d_k * n_heads)
        self.W_V = nn.Linear(d_model, d_v * n_heads)
        #
        self.model_sdpa = ScaledDotProductAttention(d_k)
        self.output_linear = nn.Linear(n_heads * d_v, d_v)

        # here d_v == d_model
        # rework this module to be independent of the dimensions
        # or simplify the dimensions
        self.output_norm = nn.LayerNorm(d_v)

    def forward(self, x, attn_mask):
        # x         (b, d_l, d_model) = (b, s, m)
        # attn_mask (b, d_l, d_model)
        #
        d_b = x.size(0)
        #
        # (b, s, m) x (h, m, k) -> (b, h, m, k)
        q_s = self.W_Q(x).view(d_b, -1, self.n_heads, self.d_k).transpose(1, 2)
        k_s = self.W_K(x).view(d_b, -1, self.n_heads, self.d_k).transpose(1, 2)

        # (b, s, m) x (h, m, v) -> (b, h, m, k)
        v_s = self.W_V(x).view(d_b, -1, self.n_heads, self.d_v).transpose(1, 2)

        # (b, l, l) -> (b, h, l, l)
        attn_mask_headed = attn_mask.unsqueeze(1).repeat(1, self.n_heads, 1, 1)

        context, attn = self.model_sdpa(q_s, k_s, v_s, attn_mask_headed)

        # (b, h, l, v) -> (b, l, h * v)
        context = context.transpose(1, 2).contiguous().view(
            d_b, -1, self.n_heads * self.d_v)

        # (b, l, h*v) - > (b, l, v)
        output = self.output_linear(context)

        # (b, l, v) -> (b, l, v) where v == d_model right now
        output = self.output_norm(x + output)
        return output, attn


class PoswiseFeedForwardNet(nn.Module):
    def __init__(self, d_model, d_ff):
        super(PoswiseFeedForwardNet, self).__init__()
        self.d_model = d_model
        self.d_ff = d_ff
        self.fc1 = nn.Linear(d_model, d_ff)
        self.fc2 = nn.Linear(d_ff, d_model)
        self.gelu = torch.nn.GELU()

    def forward(self, x):
        # (b, l, m) -> (b, l, d_ff) -> (b, l, m)
        out = self.fc1(x)
        out = self.gelu(out)
        out = self.fc2(out)
        return out


class AttentionEncoder(nn.Module):
    def __init__(self, d_model, d_k, d_v, n_heads, d_ff):
        super(AttentionEncoder, self).__init__()
        self.enc_self_attn = MultiHeadAttention(d_model, d_k, d_v, n_heads)
        self.pos_ffn = PoswiseFeedForwardNet(d_model, d_ff)

    def forward(self, x, attn_mask):
        enc_outputs, attn = self.enc_self_attn(x, attn_mask)
        enc_outputs = self.pos_ffn(enc_outputs)
        return enc_outputs, attn
