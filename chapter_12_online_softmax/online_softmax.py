import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch

from common.benchmark import bench
from common.check import assert_close
from common.utils import get_device, set_seed


def _validate_qkv(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> None:
    if q.ndim != 4 or q.shape != k.shape or q.shape != v.shape:
        raise ValueError("q, k, and v must share shape [B, H, S, D]")
    if not q.is_cuda or not k.is_cuda or not v.is_cuda:
        raise ValueError("q, k, and v must be CUDA tensors")


def torch_attention_reference(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, causal: bool = False) -> torch.Tensor:
    _validate_qkv(q, k, v)
    scores = torch.matmul(q.float(), k.float().transpose(-2, -1)) / math.sqrt(q.shape[-1])
    if causal:
        positions = torch.arange(q.shape[-2], device=q.device)
        scores = scores.masked_fill(positions[None, :] > positions[:, None], -float("inf"))
    return torch.matmul(torch.softmax(scores, dim=-1), v.float()).to(q.dtype)


def torch_online_attention(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, block_m: int = 16, block_n: int = 32, causal: bool = False) -> torch.Tensor:
    _validate_qkv(q, k, v)
    if block_m <= 0 or block_n <= 0:
        raise ValueError("block_m and block_n must be positive")
    B, H, S, D = q.shape
    scale = 1.0 / math.sqrt(D)
    output = torch.empty_like(q)

    for q_start in range(0, S, block_m):
        q_end = min(q_start + block_m, S)
        q_block = q[:, :, q_start:q_end].float()
        rows = q_end - q_start
        m = torch.full((B, H, rows), -float("inf"), device=q.device, dtype=torch.float32)
        l = torch.zeros((B, H, rows), device=q.device, dtype=torch.float32)
        acc = torch.zeros((B, H, rows, D), device=q.device, dtype=torch.float32)
        query_positions = torch.arange(q_start, q_end, device=q.device)[:, None]

        for k_start in range(0, S, block_n):
            k_end = min(k_start + block_n, S)
            k_block = k[:, :, k_start:k_end].float()
            v_block = v[:, :, k_start:k_end].float()
            block_scores = torch.matmul(q_block, k_block.transpose(-2, -1)) * scale
            if causal:
                key_positions = torch.arange(k_start, k_end, device=q.device)[None, :]
                block_scores = block_scores.masked_fill(key_positions > query_positions, -float("inf"))

            block_max = block_scores.max(dim=-1).values
            m_new = torch.maximum(m, block_max)
            alpha = torch.exp(m - m_new)
            alpha = torch.where(torch.isfinite(m_new), alpha, torch.zeros_like(alpha))
            p = torch.exp(block_scores - m_new[..., None])
            p = torch.where(torch.isfinite(block_scores), p, torch.zeros_like(p))
            l = alpha * l + p.sum(dim=-1)
            acc = alpha[..., None] * acc + torch.matmul(p, v_block)
            m = m_new

        # Standard causal self-attention always includes each query's own key.
        # This guard also prevents NaN if future masking changes leave a row empty.
        safe_l = torch.where(l > 0, l, torch.ones_like(l))
        normalized = acc / safe_l[..., None]
        output[:, :, q_start:q_end] = torch.where(
            (l > 0)[..., None], normalized, torch.zeros_like(normalized)
        ).to(q.dtype)
    return output


def main() -> None:
    device = get_device()
    set_seed(0)
    q = torch.randn(2, 4, 128, 64, device=device, dtype=torch.float16)
    k = torch.randn_like(q)
    v = torch.randn_like(q)
    for causal in (False, True):
        expected = torch_attention_reference(q, k, v, causal)
        actual = torch_online_attention(q, k, v, causal=causal)
        assert_close(f"online attention causal={causal}", actual.float(), expected.float(), rtol=1e-2, atol=1e-2)
        print(f"causal={causal}: full={bench(lambda: torch_attention_reference(q, k, v, causal)):.3f} ms, online={bench(lambda: torch_online_attention(q, k, v, causal=causal), warmup=5, rep=20):.3f} ms")
    print(f"Full score elements per head: {128 * 128:,}")
    print(f"One 16x32 score block:        {16 * 32:,}")


if __name__ == "__main__":
    main()
