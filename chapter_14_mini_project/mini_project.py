import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch
import triton
import triton.language as tl

from common.benchmark import bench
from common.check import assert_close
from common.utils import get_device, set_seed


BLOCK_M = 32
BLOCK_N = 32
HEAD_DIM = 64


@triton.jit
def flash_attention_kernel(q_ptr, k_ptr, v_ptr, output_ptr, S, scale, CAUSAL: tl.constexpr, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, HEAD_DIM: tl.constexpr):
    query_block = tl.program_id(0)
    batch_head = tl.program_id(1)
    head_offset = batch_head * S * HEAD_DIM
    query_indices = query_block * BLOCK_M + tl.arange(0, BLOCK_M)
    key_indices = tl.arange(0, BLOCK_N)
    dims = tl.arange(0, HEAD_DIM)
    query_valid = query_indices < S

    q_ptrs = q_ptr + head_offset + query_indices[:, None] * HEAD_DIM + dims[None, :]
    q = tl.load(q_ptrs, mask=query_valid[:, None], other=0.0)
    m_i = tl.where(query_valid, -float("inf"), 0.0)
    l_i = tl.zeros((BLOCK_M,), dtype=tl.float32)
    acc = tl.zeros((BLOCK_M, HEAD_DIM), dtype=tl.float32)

    for key_block in range(0, tl.cdiv(S, BLOCK_N)):
        keys = key_block * BLOCK_N + key_indices
        key_valid = keys < S
        k_ptrs = k_ptr + head_offset + dims[:, None] + keys[None, :] * HEAD_DIM
        v_ptrs = v_ptr + head_offset + keys[:, None] * HEAD_DIM + dims[None, :]
        k = tl.load(k_ptrs, mask=key_valid[None, :], other=0.0)
        v = tl.load(v_ptrs, mask=key_valid[:, None], other=0.0)
        qk = tl.dot(q, k) * scale
        valid_scores = query_valid[:, None] & key_valid[None, :]
        if CAUSAL:
            valid_scores &= keys[None, :] <= query_indices[:, None]
        qk = tl.where(valid_scores, qk, -float("inf"))

        block_max = tl.max(qk, axis=1)
        m_new = tl.where(query_valid, tl.maximum(m_i, block_max), 0.0)
        alpha = tl.exp(m_i - m_new)
        p = tl.exp(qk - m_new[:, None])
        p = tl.where(valid_scores, p, 0.0)
        l_i = l_i * alpha + tl.sum(p, axis=1)
        acc = acc * alpha[:, None] + tl.dot(p.to(v.dtype), v)
        m_i = m_new

    output = acc / l_i[:, None]
    output_ptrs = output_ptr + head_offset + query_indices[:, None] * HEAD_DIM + dims[None, :]
    tl.store(output_ptrs, output, mask=query_valid[:, None])


def _flash_attention_forward(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, causal: bool) -> torch.Tensor:
    if q.ndim != 4 or q.shape != k.shape or q.shape != v.shape:
        raise ValueError("q, k, and v must share shape [B, H, S, D]")
    if not q.is_cuda or not k.is_cuda or not v.is_cuda:
        raise ValueError("q, k, and v must be CUDA tensors")
    if not q.is_contiguous() or not k.is_contiguous() or not v.is_contiguous():
        raise ValueError("q, k, and v must be contiguous")
    if q.dtype != torch.float16 or k.dtype != torch.float16 or v.dtype != torch.float16:
        raise ValueError("this teaching kernel supports fp16 inputs only")
    B, H, S, D = q.shape
    if S == 0:
        raise ValueError("sequence length must be positive")
    if D != HEAD_DIM:
        raise ValueError(f"this teaching kernel supports D={HEAD_DIM} only")
    output = torch.empty_like(q)
    grid = (triton.cdiv(S, BLOCK_M), B * H)
    flash_attention_kernel[grid](q, k, v, output, S, 1.0 / math.sqrt(D), CAUSAL=causal, BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, HEAD_DIM=HEAD_DIM, num_warps=4, num_stages=2)
    return output


def _reshape_heads(x: torch.Tensor, num_heads: int) -> torch.Tensor:
    B, S, C = x.shape
    head_dim = C // num_heads
    return x.view(B, S, num_heads, head_dim).transpose(1, 2).contiguous()


def _merge_heads(x: torch.Tensor) -> torch.Tensor:
    B, H, S, D = x.shape
    return x.transpose(1, 2).contiguous().view(B, S, H * D)


def _torch_attention(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, causal: bool) -> torch.Tensor:
    scores = torch.matmul(q.float(), k.float().transpose(-2, -1)) / math.sqrt(q.shape[-1])
    if causal:
        positions = torch.arange(q.shape[-2], device=q.device)
        scores = scores.masked_fill(positions[None, :] > positions[:, None], -float("inf"))
    return torch.matmul(torch.softmax(scores, dim=-1), v.float()).to(q.dtype)


def tiny_self_attention_torch(x: torch.Tensor, weights: dict[str, torch.Tensor], causal: bool = False) -> torch.Tensor:
    num_heads = weights["num_heads"]
    q = _reshape_heads(torch.matmul(x, weights["wq"]), num_heads)
    k = _reshape_heads(torch.matmul(x, weights["wk"]), num_heads)
    v = _reshape_heads(torch.matmul(x, weights["wv"]), num_heads)
    return torch.matmul(_merge_heads(_torch_attention(q, k, v, causal)), weights["wo"])


def tiny_self_attention_triton(x: torch.Tensor, weights: dict[str, torch.Tensor], causal: bool = False) -> torch.Tensor:
    num_heads = weights["num_heads"]
    q = _reshape_heads(torch.matmul(x, weights["wq"]), num_heads)
    k = _reshape_heads(torch.matmul(x, weights["wk"]), num_heads)
    v = _reshape_heads(torch.matmul(x, weights["wv"]), num_heads)
    return torch.matmul(_merge_heads(_flash_attention_forward(q, k, v, causal)), weights["wo"])


def main() -> None:
    device = get_device()
    set_seed(0)
    B, S, C, H = 2, 128, 256, 4
    x = torch.randn(B, S, C, device=device, dtype=torch.float16)
    weights = {
        "wq": torch.randn(C, C, device=device, dtype=torch.float16) / math.sqrt(C),
        "wk": torch.randn(C, C, device=device, dtype=torch.float16) / math.sqrt(C),
        "wv": torch.randn(C, C, device=device, dtype=torch.float16) / math.sqrt(C),
        "wo": torch.randn(C, C, device=device, dtype=torch.float16) / math.sqrt(C),
        "num_heads": H,
    }
    for causal in (False, True):
        expected = tiny_self_attention_torch(x, weights, causal)
        actual = tiny_self_attention_triton(x, weights, causal)
        assert_close(f"tiny self-attention causal={causal}", actual.float(), expected.float(), rtol=2e-2, atol=2e-2)
        print(f"causal={causal}: PyTorch={bench(lambda: tiny_self_attention_torch(x, weights, causal)):.3f} ms, Triton={bench(lambda: tiny_self_attention_triton(x, weights, causal)):.3f} ms")


if __name__ == "__main__":
    main()
