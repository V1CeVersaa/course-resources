import math

import torch
import torch.nn as nn
import torch.nn.init as init
from einops import einsum
from jaxtyping import Bool, Float, Int
from torch.types import Tensor

from cs336_basics.utils import softmax


class RoPE(nn.Module):
    def __init__(self, theta: float, d_k: int, max_seq_len: int, device: torch.device | None = None):
        super().__init__()
        self.theta = theta
        self.d_k = d_k
        self.max_seq_len = max_seq_len

        positions = torch.arange(max_seq_len, dtype=torch.float32, device=device)  # (max_seq_len, )
        half_dims = torch.arange(0, d_k, 2, dtype=torch.float32, device=device)
        inv_freqs = torch.pow(self.theta, -half_dims / d_k)  # (d_k/2, )
        angles = einsum(positions, inv_freqs, "... max_seq_len, ... d_k -> ... max_seq_len d_k")  # (max_seq_len, d_k/2)

        self.register_buffer("cos_cache", torch.cos(angles), persistent=False)  # (max_seq_len, d_k/2)
        self.register_buffer("sin_cache", torch.sin(angles), persistent=False)

    def forward(self, x: torch.Tensor, token_positions: torch.Tensor) -> torch.Tensor:
        cos_pos = self.cos_cache[token_positions].to(dtype=x.dtype, device=x.device)  # type: ignore
        sin_pos = self.sin_cache[token_positions].to(dtype=x.dtype, device=x.device)  # type: ignore

        x_even = x[..., 0::2]  # (..., seq_len, d_k/2)
        x_odd = x[..., 1::2]

        x_rotated_even = x_even * cos_pos - x_odd * sin_pos  # (..., seq_len, d_k/2)
        x_rotated_odd = x_even * sin_pos + x_odd * cos_pos
        rotated = torch.empty_like(x)
        rotated[..., 0::2] = x_rotated_even
        rotated[..., 1::2] = x_rotated_odd
        return rotated


class MultiheadSelfAttention(nn.Module):
    def __init__(self, d_model: int, num_heads: int, RoPE: RoPE | None = None):
        super().__init__()
        self.d_model = d_model
        self.d_kv = d_model // num_heads
        self.num_heads = num_heads
        self.qkv_proj = Linear(d_model, 3 * d_model)
        self.o_proj = Linear(d_model, d_model)
        self.RoPE = RoPE

    def forward(self, x: torch.Tensor, token_positions: torch.Tensor | None = None) -> torch.Tensor:
        qkv = self.qkv_proj(x)  # (..., seq_len, 3 * d_model)
        q, k, v = qkv.split(self.d_model, dim=-1)

        def _reshape_heads(tensor: torch.Tensor) -> torch.Tensor:
            """Reshape tensor with shape (batch, seq_len, d_model) to (batch, num_heads, seq_len, d_kv)"""
            return tensor.view(*tensor.shape[:-1], self.num_heads, self.d_kv).transpose(-3, -2)

        q_heads, k_heads, v_heads = (_reshape_heads(t) for t in (q, k, v))  # (..., num_heads, seq_len, d_kv)

        def _make_casual_mask(width: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
            mask_raw = torch.ones((width, width), device=device, dtype=dtype)
            mask = mask_raw.tril().view(1, 1, width, width) == 1  # (1, 1, width, width)
            return mask

        seq_len = x.shape[-2]
        mask_raw = _make_casual_mask(seq_len, x.device, x.dtype)  # (1, 1, seq_len, seq_len)
        mask = mask_raw.expand(*x.shape[:-2], self.num_heads, seq_len, seq_len)  # (..., num_heads, seq_len, seq_len)

        if self.RoPE is not None:
            token_positions = torch.arange(seq_len) if token_positions is None else token_positions
            q_heads = self.RoPE(q_heads, token_positions)
            k_heads = self.RoPE(k_heads, token_positions)

        attn = scaled_dot_product_attention(q_heads, k_heads, v_heads, mask=mask)  # (..., num_heads, seq_len, d_kv)
        attn = attn.contiguous().transpose(-3, -2)
        attn = attn.reshape(*x.shape[:-2], seq_len, self.d_model)  # (..., seq_len, d_model)
        attn_out = self.o_proj(attn)

        return attn_out


class TransformerBlock(nn.Module):
    def __init__(self, d_model: int, num_heads: int, d_ff: int, max_seq_len: int, theta: float):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_ff = d_ff
        self.max_seq_len = max_seq_len
        self.theta = theta

        self.rope = RoPE(theta, d_model // num_heads, max_seq_len)
        self.rms_norm_1 = RMSNorm(d_model)
        self.attention = MultiheadSelfAttention(d_model, num_heads, self.rope)
        self.rms_norm_2 = RMSNorm(d_model)
        self.feedforward = SwiGLU(d_model, d_ff)

    def forward(self, x: Float[Tensor, "batch sequence_length d_model"]) -> torch.Tensor:
        layer1 = x + self.attention(self.rms_norm_1(x))
        layer2 = layer1 + self.feedforward(self.rms_norm_2(layer1))
        return layer2


class TransformerLM(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        context_length: int,
        num_layers: int,
        d_model: int,
        num_heads: int,
        d_ff: int,
        rope_theta: float,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.context_length = context_length
        self.num_layers = num_layers
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_ff = d_ff
        self.rope_theta = rope_theta

        self.embeddings = Embedding(vocab_size, d_model)
        self.transformers = nn.ModuleList([
            TransformerBlock(d_model, num_heads, d_ff, context_length, rope_theta) for _ in range(num_layers)
        ])
        self.out_norm = RMSNorm(d_model)
        self.out_embedding = Linear(d_model, vocab_size)

    def forward(self, in_indices: Int[Tensor, " batch_size sequence_length"]):
        in_indices = self.embeddings(in_indices)
        for layer in self.transformers:
            in_indices = layer(in_indices)
        out_normalize = self.out_norm(in_indices)
        out_embedings = self.out_embedding(out_normalize)
        # out_inference = softmax(out_embedings)
        return out_embedings


class Linear(nn.Module):
    def __init__(
        self, in_features: int, out_features: int, device: torch.device | None = None, dtype: torch.dtype | None = None
    ):
        super().__init__()
        std_init = math.sqrt(2 / (in_features + out_features))
        self.weight = nn.Parameter(torch.empty(out_features, in_features, dtype=dtype, device=device))
        init.trunc_normal_(self.weight, 0, std_init, -3 * std_init, 3 * std_init)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return einsum(self.weight, x, "d_out d_in, ... d_in -> ... d_out")


class Embedding(nn.Module):
    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(num_embeddings, embedding_dim, dtype=dtype, device=device))
        init.trunc_normal_(self.weight, 0, 1, -3, 3)

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        return self.weight[token_ids]


class RMSNorm(nn.Module):
    def __init__(
        self, d_model: int, eps: float = 1e-5, device: torch.device | None = None, dtype: torch.dtype | None = None
    ):
        super().__init__()
        self.d_model = d_model
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d_model, device=device, dtype=dtype))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        in_dtype = x.dtype
        x = x.to(torch.float32)  # (... d_model)
        rms = torch.sqrt(torch.mean(x**2, dim=-1, keepdim=True) + self.eps)  # (... 1)
        result = (x / rms) * self.weight
        return result.to(in_dtype)


def silu(in_features: torch.Tensor) -> torch.Tensor:
    sigma = torch.sigmoid(in_features)
    return in_features * sigma


class SwiGLU(nn.Module):
    def __init__(self, d_model: int, d_ff: int, device: torch.device | None = None, dtype: torch.dtype | None = None):
        super().__init__()
        self.d_model = d_model
        self.d_ff = d_ff
        self.w1 = nn.Parameter(torch.empty(d_ff, d_model, device=device, dtype=dtype))
        self.w2 = nn.Parameter(torch.empty(d_model, d_ff, device=device, dtype=dtype))
        self.w3 = nn.Parameter(torch.empty(d_ff, d_model, device=device, dtype=dtype))

        std = 0.02
        for layer in (self.w1, self.w2, self.w3):
            init.trunc_normal_(layer, 0, std, -2 * std, 2 * std)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        silu_out = silu(einsum(self.w1, x, "d_ff d_model, ... d_model -> ... d_ff"))  # SiLU(W_1 x)
        silu_para = einsum(self.w3, x, "d_ff d_model, ... d_model -> ... d_ff")  # W_3 x
        silu_w2_para = silu_out * silu_para  # Silu(W_1 x) otimes W_3 x
        return einsum(self.w2, silu_w2_para, "d_model d_ff, ... d_ff -> ... d_model")  # W_2 (Silu(W_1 x) otimes W_3 x)


def scaled_dot_product_attention(
    Q: Float[Tensor, " ... queries d_k"],
    K: Float[Tensor, " ... keys d_k"],
    V: Float[Tensor, " ... values d_v"],
    mask: Bool[Tensor, " ... queries keys"] | None = None,
) -> Float[Tensor, " ... queries d_v"]:
    scale = math.sqrt(Q.shape[-1])
    scores = einsum(Q, K, "... queries d_k, ... keys d_k -> ... queries keys")  # (..., queries, keys)
    scores /= scale
    scores_masked = scores.masked_fill(~mask, float("-inf")) if mask is not None else scores
    weights = softmax(scores_masked, dim=-1)  # (..., queries, keys)
    attention = einsum(weights, V, "... queries keys, ... keys d_v -> ... queries d_v")  # (..., queries, d_v)
    return attention
