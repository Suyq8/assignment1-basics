from argparse import Namespace
from typing import List, Optional
import torch
import torch.nn as nn
from einops import einsum, rearrange
from math import sqrt

from tqdm import tqdm

from cs336_basics.bpe import Tokenizer


def softmax(x: torch.Tensor, dim: int = -1) -> torch.Tensor:
    x_max = x.max(dim=dim, keepdim=True).values
    x_exp = torch.exp(x - x_max)
    return x_exp / x_exp.sum(dim=dim, keepdim=True)


def scaled_dot_product_attention(
    Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor, mask: torch.Tensor | None = None
) -> torch.Tensor:
    """
    Calculate the scaled dot-product attention.
    This function computes the attention values by performing a scaled dot-product of the queries and keys,
    followed by applying a softmax to obtain the attention probabilities. The resulting probabilities are
    then used to weigh the values. It supports input tensors with additional batch dimensions and allows
    for an optional boolean mask.

    Args:
        Q: torch.Tensor: Query tensor of shape (batch_size, ..., seq_len, d_k)
        K: torch.Tensor: Key tensor of shape (batch_size, ..., seq_len, d_k)
        V: torch.Tensor: Value tensor of shape (batch_size, ..., seq_len, d_v)
        mask: torch.Tensor, optional: A boolean tensor of shape (seq_len, seq_len) that indicates which
        positions to attend to. For each query in the sequence, the attention probabilities corresponding
        to positions with a mask value of False will be set to zero, while those with True values will
        collectively sum to 1. If not provided, no masking is applied.
    """
    QK = einsum(Q, K, "... s_q d_k, ... s_k d_k -> ... s_q s_k")
    QK /= sqrt(Q.shape[-1])
    if mask is not None:
        QK = QK.masked_fill(mask == False, float("-inf"))
    return QK.softmax(dim=-1) @ V


class Linear(nn.Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ):
        """
        Construct a linear transformation module. This function should accept the following parameters:

        Args:
            in_features: int final dimension of the input
            out_features: int final dimension of the output
            device: torch.device | None = None Device to store the parameters on
            dtype: torch.dtype | None = None Data type of the parameters
        """
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.device = device
        self.dtype = dtype
        self.W = nn.Parameter(
            torch.empty((out_features, in_features), device=device, dtype=dtype)
        )
        std = sqrt(2.0 / (in_features + out_features))
        torch.nn.init.trunc_normal_(self.W, std=std, a=-3 * std, b=3 * std)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply the linear transformation to the input.

         Args:
             x: torch.Tensor Input tensor of shape (..., in_features)

         Returns:
             torch.Tensor Output tensor of shape (..., out_features)
        """
        return einsum(self.W, x, "j i, ... i->... j")


class Embedding(nn.Module):
    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ):
        """
        Construct an embedding module. This function should accept the following parameters:

        Args:
            num_embeddings: int Size of the vocabulary
            embedding_dim: int Dimension of the embedding vectors
            device: torch.device | None = None Device to store the parameters on
            dtype: torch.dtype | None = None Data type of the parameters
        """
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.device = device
        self.dtype = dtype
        self.embeddings = nn.Parameter(
            torch.empty((num_embeddings, embedding_dim), device=device, dtype=dtype)
        )
        std = 1
        torch.nn.init.trunc_normal_(self.embeddings, std=std, a=-3 * std, b=3 * std)

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        """Lookup the embedding vectors for the given token IDs."""
        return self.embeddings[token_ids]


class RMSNorm(nn.Module):
    def __init__(
        self,
        d_model: int,
        eps: float = 1e-5,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ):
        """
        Construct a RMS normalization module. This function should accept the following parameters:

        Args:
            d_model: int Hidden dimension of the model
            eps: float = 1e-5 Epsilon value for numerical stability
            device: torch.device | None = None Device to store the parameters on
            dtype: torch.dtype | None = None Data type of the parameters
        """
        super().__init__()
        self.d_model = d_model
        self.eps = eps
        self.device = device
        self.dtype = dtype
        self.g = nn.Parameter(torch.ones((d_model,), device=device, dtype=dtype))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Process an input tensor of shape (batch_size, sequence_length, d_model) and
        return a tensor of the same shape.
        """
        in_dtype = x.dtype
        x = x.to(torch.float32)
        rms = (x.pow(2).mean(dim=-1, keepdim=True) + self.eps).sqrt()
        return (x / rms * self.g).to(in_dtype)


class SiLU(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply the SiLU activation function."""
        return x * x.sigmoid()


class SwiGLU(nn.Module):
    def __init__(
        self,
        d_model: int,
        d_ff: int,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ):
        """
        Construct a SwiGLU module. This function should accept the following parameters:

        Args:
            d_model: int Hidden dimension of the model
            d_ff: int Hidden dimension of the feed-forward network
            device: torch.device | None = None Device to store the parameters on
            dtype: torch.dtype | None = None Data type of the parameters
        """
        super().__init__()
        self.d_model = d_model
        self.d_ff = d_ff
        self.device = device
        self.dtype = dtype
        self.silu = SiLU()
        self.linear1 = Linear(self.d_model, self.d_ff, device=device, dtype=dtype)
        self.linear2 = Linear(self.d_ff, self.d_model, device=device, dtype=dtype)
        self.linear3 = Linear(self.d_model, self.d_ff, device=device, dtype=dtype)

    def forward(self, x: torch.Tensor):
        return self.linear2(self.silu(self.linear1(x)) * self.linear3(x))


class RotaryPositionalEmbedding(nn.Module):
    def __init__(
        self,
        theta: float,
        d_k: int,
        max_seq_len: int,
        device: torch.device | None = None,
    ):
        """
        Construct a rotary positional embedding module. This function should accept the following parameters:

        Args:
            theta: float Θ value for the RoPE
            d_k: int dimension of query and key vectors
            max_seq_len: int Maximum sequence length that will be inputted
            device: torch.device | None = None Device to store the buffer on
        """
        super().__init__()
        self.theta = theta
        self.d_k = d_k
        self.max_seq_len = max_seq_len
        self.device = device

        angles = 1.0 / (
            theta ** (torch.arange(0, d_k, 2, dtype=torch.float32, device=device) / d_k)
        )
        idx = torch.arange(max_seq_len, dtype=torch.float32, device=device).reshape(
            -1, 1
        )
        angles = angles.reshape(1, -1) * idx  # (max_seq_len, d_k//2)
        self.register_buffer(
            "freqs", torch.polar(torch.ones_like(angles), angles), persistent=False
        )  # cos+sin*j

    def forward(self, x: torch.Tensor, token_positions: torch.Tensor) -> torch.Tensor:
        """
        Process an input tensor of shape (..., seq_len, d_k) and return a tensor of the same shape. Note
        that you should tolerate x with an arbitrary number of batch dimensions. You should assume
        that the token positions are a tensor of shape (..., seq_len) specifying the token positions of
        x along the sequence dimension.
        You should use the token positions to slice your (possibly precomputed) cos and sin tensors along
        the sequence dimension.

         Args:
             x: torch.Tensor Input tensor of shape (..., seq_len, d_k)
             token_positions: torch.Tensor Token positions of shape (..., seq_len)

         Returns:
             torch.Tensor Output tensor of shape (..., seq_len, d_k)
        """
        # (x0+x1j) * (cos+sinj) = (x0*cos-x1*sin) + (x0*sin+x1*cos)*j
        _x = torch.view_as_complex(
            x.float().reshape(*x.shape[:-1], x.shape[-1] // 2, 2)
        )
        r = self.freqs[token_positions]
        out = _x * r
        return torch.view_as_real(out).reshape(*x.shape).to(x.dtype)


class CausalMultiheadSelfAttention(nn.Module):
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        rope: RotaryPositionalEmbedding | None = None,
        token_positions: torch.Tensor | None = None,
        dtype: torch.dtype | None = None,
        device: torch.device | None = None,
    ):
        """
        Construct a casual multi-head self-attention module. This function should accept the following parameters:

        Args:
            d_model: int Dimensionality of the Transformer block inputs.
            num_heads: int Number of heads to use in multi-head self-attention.
            device: torch.device | None = None Device to store the parameters on
        """
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.rope = rope
        self.token_positions = token_positions
        self.d_head = d_model // num_heads

        self.linear_qkv = Linear(d_model, d_model * 3, device=device, dtype=dtype)
        self.linear_o = Linear(d_model, d_model, device=device, dtype=dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.shape
        qkv = (
            self.linear_qkv(x)
            .view(B, T, 3, self.num_heads, self.d_head)
            .permute(2, 0, 3, 1, 4)
        )
        q, k, v = (x[0] for x in qkv.split(1, dim=0))

        if self.rope is not None:
            if self.token_positions is not None:
                token_positions = self.token_positions
            else:
                token_positions = torch.arange(T, device=x.device)
            q = self.rope(q.contiguous(), token_positions)
            k = self.rope(k.contiguous(), token_positions)

        mask = torch.full((T, T), True, device=x.device, dtype=torch.bool)
        mask = torch.triu(mask, diagonal=1)
        attn = scaled_dot_product_attention(q, k, v, mask=~mask)

        attn = rearrange(attn, "b h t d -> b t (h d)")
        return self.linear_o(attn)


class TransformerBlock(nn.Module):
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        d_ff: int,
        rope: RotaryPositionalEmbedding | None = None,
        dtype: torch.dtype | None = None,
        device: torch.device | None = None,
    ):
        """
        Construct a Transformer block. This function should accept the following parameters:

        Args:
            d_model: int Dimensionality of the Transformer block inputs.
            num_heads: int Number of heads to use in multi-head self-attention.
            d_ff: int Dimensionality of the position-wise feed-forward inner layer.
            rope: RotaryPositionalEmbedding | None = None Optional rotary positional embedding module.
            dtype: torch.dtype | None = None Data type of the parameters
            device: torch.device | None = None Device to store the parameters on
        """
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_ff = d_ff
        self.device = device
        self.dtype = dtype

        self.rope = rope
        self.attention = CausalMultiheadSelfAttention(d_model, num_heads, rope=rope, device=device, dtype=dtype)
        self.norm1 = RMSNorm(d_model, device=device, dtype=dtype)
        self.norm2 = RMSNorm(d_model, device=device, dtype=dtype)
        self.ffn = SwiGLU(d_model, d_ff, device=device, dtype=dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attention(self.norm1(x))
        x = x + self.ffn(self.norm2(x))
        return x
    

class Transformer(nn.Module):
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        d_ff: int,
        num_layers: int,
        vocab_size: int,
        context_length: int,
        rope_theta: float = 10000.0,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ):

        """
        Construct a Transformer model. This function should accept the following parameters:

        Args:
            d_model: int Dimensionality of the Transformer block inputs.
            num_heads: int Number of heads to use in multi-head self-attention.
            d_ff: int Dimensionality of the position-wise feed-forward inner layer.
            vocab_size: int The size of the vocabulary, necessary for determining the dimensionality of the token
            embedding matrix.
            context_length: int The maximum context length, necessary for determining the dimensionality of
            the position embedding matrix.
            num_layers: int The number of Transformer blocks to use.
            rope: RotaryPositionalEmbedding | None = None Optional rotary positional embedding module.
            dtype: torch.dtype | None = None Data type of the parameters
            device: torch.device | None = None Device to store the parameters on
        """
        super().__init__()
        assert (
            d_model % num_heads == 0
        ), f"d_model {d_model} must be divisible by num_heads {num_heads}"

        self.d_model = d_model
        self.num_heads = num_heads
        self.d_ff = d_ff
        self.num_layers = num_layers
        self.rope = RotaryPositionalEmbedding(
            theta=rope_theta,
            d_k=d_model//num_heads,
            max_seq_len=context_length,
            device=device,
        )
        self.device = device
        self.dtype = dtype
        self.context_length = context_length

        self.token_embedding = Embedding(
            num_embeddings=vocab_size,
            embedding_dim=d_model,
            device=device,
            dtype=dtype,
        )
        self.layers = nn.ModuleList(
            [
                TransformerBlock(d_model, num_heads, d_ff, rope=self.rope, device=device, dtype=dtype)
                for _ in range(num_layers)
            ]
        )
        self.norm = RMSNorm(d_model, device=device, dtype=dtype)
        self.linear_o = Linear(d_model, vocab_size, device=device, dtype=dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.size(1)>self.context_length:
            x = x[:, :self.context_length]

        x = self.token_embedding(x)
        for layer in self.layers:
            x = layer(x)
        
        return self.linear_o(self.norm(x))
    
    def sample_top_p(self, probs: torch.Tensor, top_p: float) -> torch.Tensor:
        # probs: (batch_size, vocab_size)
        probs_sort, probs_idx = torch.sort(probs, dim=-1, descending=True)
        probs_sum = torch.cumsum(probs_sort, dim=-1)
        mask = probs_sum - probs_sort > top_p
        probs_sort[mask] = 0.0
        probs_sort.div_(probs_sort.sum(dim=-1, keepdim=True))
        next_token = torch.multinomial(probs_sort, num_samples=1)
        next_token = torch.gather(probs_idx, -1, next_token)
        return next_token

    #@torch.compile(dynamic=True, fullgraph=True)
    def generate_one_next_token(self, x: torch.Tensor, temperature: float=0.7, top_p: float=0.9) -> torch.Tensor:
        logits = self.forward(x)  # (batch_size, seq_len, vocab_size)

        if temperature > 0:
            probs = torch.softmax(logits[:, -1] / temperature, dim=-1)
            next_token = self.sample_top_p(probs, top_p).flatten()
        else:
            next_token = torch.argmax(logits[:, -1], dim=-1).flatten()

        return next_token

    @torch.inference_mode()
    def generate(
        self,
        prompt: List[int],
        max_seq_len: int,
        stop_token: List[int],
        max_gen_len: int = 256,
    ) -> List[int]:
        stop_token = stop_token[0]
        x = torch.tensor(prompt, dtype=torch.long, device=self.device).unsqueeze(0)
        for _ in range(max_gen_len):
            n = x.shape[2]
            nxt_token = self.generate_one_next_token(x[max(0, n-max_seq_len):])
            x = torch.cat([x, nxt_token.unsqueeze(0)], dim=2)
            if nxt_token[0].item() == stop_token:
                break
        return x[0].tolist()
    
def text_completion(
    prompt: str,
    tokenizer: Tokenizer,
    model: Transformer,
    args: Namespace,
    max_gen_len: Optional[int] = None,
) -> List[str]:
    if max_gen_len is None:
        max_gen_len = args.max_seq_len - 1

    tokens = tokenizer.encode(prompt)
    stop_token = tokenizer.encode("<|endoftext|>")

    model.eval()
    generated_token = model.generate(tokens, max_gen_len, stop_token)

    return tokenizer.decode(generated_token)