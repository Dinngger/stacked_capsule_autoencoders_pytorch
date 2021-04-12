import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class QKAttention(nn.Module):
    """Transformer-like minus attention with one-hot output"""

    def __init__(self, d_k):
        super(QKAttention, self).__init__()
        self.conv1 = nn.Conv2d(d_k, 16, 1, 1)
        self.conv2 = nn.Conv2d(16, 1, 1, 1)

    def forward(self, queries, keys, q_pres=None, k_pres=None):
        """Builds the module.

        Args:
          queries: Tensor of shape [B, M, d_k].
          keys: Tensor of shape [B, N, d_k].
          presence: None or tensor of shape [B, N].

        Returns:
          Tensor of shape [B, M, N]
        """

        # [B, M, d] x [B, d, N] = [B, M, N]
        routing = torch.matmul(queries, keys.permute(0, 2, 1))

        # [B, M, N, d]
        # M = int(queries.shape[1])
        # N = int(keys.shape[1])
        # queries_expand = queries.unsqueeze(2).repeat(1, 1, N, 1)
        # keys_expand = keys.unsqueeze(1).repeat(1, M, 1, 1)
        # routing = F.relu(self.conv1((queries_expand - keys_expand).permute(0, 3, 1, 2)))
        # routing = self.conv2(routing).permute(0, 2, 3, 1).squeeze(-1)

        if q_pres is not None:
            q_pres_f = q_pres.float()
            routing = routing * q_pres_f.unsqueeze(-1)
        if k_pres is not None:
            k_pres_f = k_pres.float()
            routing = routing * k_pres_f.unsqueeze(-2)

        return routing


class QKVAttention(nn.Module):
    """Transformer-like attention."""

    def __init__(self, d_k):
        super(QKVAttention, self).__init__()
        self.d_k = d_k
        self.qk = QKAttention(d_k)

    def forward(self, queries, keys, values, q_pres=None, k_pres=None):
        """Builds the module.

        Args:
          queries: Tensor of shape [B, M, d_k].
          keys: Tensor of shape [B, N, d_k].
          values: : Tensor of shape [B, N, d_v].
          presence: None or tensor of shape [B, N].

        Returns:
          Tensor of shape [B, M, d_v]
        """

        routing = self.qk(queries, keys, q_pres, k_pres)

        if k_pres is not None:
            presence = k_pres.unsqueeze(-2).float()
            routing = routing - (1. - presence) * 1e32

        routing = F.softmax(routing / np.sqrt(self.d_k), -1)

        # every output is a linear combination of all inputs
        # [B, M, N] x [B, N, d_v] = [B, M, d_v]
        res = torch.matmul(routing, values)
        return res


class TransformedQKVAttention(nn.Module):
    def __init__(self, d_k, d_v, n_d_k, n_d_v):
        super(TransformedQKVAttention, self).__init__()
        self.conv_q = nn.Conv1d(d_k, n_d_k, 1, 1)
        self.conv_k = nn.Conv1d(d_k, n_d_k, 1, 1)
        self.conv_v = nn.Conv1d(d_v, n_d_v, 1, 1)
        self.qkv = QKVAttention(n_d_k)

    def forward(self, queries, keys, values, q_pres=None, k_pres=None):
        return self.qkv(self.conv_q(queries.transpose(1, 2)).transpose(1, 2),
                        self.conv_k(keys.transpose(1, 2)).transpose(1, 2),
                        self.conv_v(values.transpose(1, 2)).transpose(1, 2), q_pres, k_pres)


class MultiHeadQKVAttention(nn.Module):
    """Multi-head version of Transformer-like attention."""

    def __init__(self, n_heads, d_k, d_v):
        super(MultiHeadQKVAttention, self).__init__()
        self._n_heads = n_heads
        n_d_k = int(np.ceil(float(d_k) / n_heads))
        n_d_v = int(np.ceil(float(d_v) / n_heads))
        self.conv_o = nn.Conv1d(n_d_v * n_heads, d_v, 1, 1)
        self.tqkvs = nn.ModuleList()
        for _ in range(n_heads):
            self.tqkvs.append(QKVAttention(n_d_k))

    def forward(self, queries, keys, values, q_pres=None, k_pres=None):
        """Builds the module.

        Args:
          queries: Tensor of shape [B, M, d_k].
          keys: Tensor of shape [B, N, d_k].
          values: : Tensor of shape [B, N, d_v].
          presence: None or tensor of shape [B, N].

        Returns:
          Tensor of shape [B, M, d_v]
        """

        outputs = []
        for i in range(self._n_heads):
            outputs.append(self.tqkvs[i](queries, keys, values, q_pres, k_pres))

        return self.conv_o(torch.cat(outputs, -1).transpose(1, 2)).transpose(1, 2)


class SelfAttention(nn.Module):
    """Self-attention where keys, values and queries are the same."""

    def __init__(self, n_heads, d_k, layerNorm=True):
        super(SelfAttention, self).__init__()
        self.layerNorm = layerNorm
        self.conv0 = nn.Conv1d(d_k, 2*d_k, 1, 1)
        self.conv1 = nn.Conv1d(2*d_k, d_k, 1, 1)
        if layerNorm:
            self.ln0 = nn.LayerNorm([d_k])
            self.ln1 = nn.LayerNorm([d_k])
        self.mqkv = MultiHeadQKVAttention(n_heads, d_k, d_k)

    def forward(self, x, presence=None):

        y = self.mqkv(x, x, x, presence, presence)
        y = y + x
        if presence is not None:
            y = y * presence.unsqueeze(-1).float()
        if self.layerNorm:
            y = self.ln0(y)

        h = self.conv0(y.transpose(1, 2)).transpose(1, 2)
        h = self.conv1(F.relu(h).transpose(1, 2)).transpose(1, 2)
        h = F.relu(h) + y
        if self.layerNorm:
            h = self.ln1(h)

        return h


class AttentionPooling(nn.Module):
    def __init__(self, d_k, n_heads, n_out, layerNorm=True):
        super(AttentionPooling, self).__init__()
        self.inducing_points = nn.Parameter(torch.Tensor(1, n_out, d_k))
        nn.init.xavier_uniform_(self.inducing_points)
        self.mqkv = MultiHeadQKVAttention(n_heads, d_k, d_k)

    def forward(self, x, presence=None):
        y = self.mqkv(self.inducing_points.repeat(x.size(0), 1, 1), x, x, k_pres=presence)
        return y


class SetTransformer(nn.Module):
    """Permutation-invariant Transformer."""

    def __init__(self,
                 d_x,
                 d_h,
                 d_o,
                 n_layers,
                 n_output,
                 n_heads):
        super(SetTransformer, self).__init__()
        self.conv1 = nn.Conv1d(d_x, d_h, 1, 1)
        self.conv2 = nn.Conv1d(d_h, d_o, 1, 1)
        self.enc = nn.ModuleList()
        for _ in range(n_layers):
            self.enc.append(SelfAttention(n_heads, d_h))
        self.pooling = AttentionPooling(d_o, n_heads, n_output)

    def forward(self, x, presence=None):
        h = self.conv1(x.transpose(1, 2)).transpose(1, 2)
        for m in self.enc:
            h = m(h, presence)
        z = self.conv2(h.transpose(1, 2)).transpose(1, 2)
        return self.pooling(z, presence)
