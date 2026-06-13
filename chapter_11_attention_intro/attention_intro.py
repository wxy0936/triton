import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch
import torch.nn.functional as F

from common.benchmark import bench
from common.check import assert_close
from common.utils import get_device, set_seed


def _validate_qkv(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> None:
    if q.ndim != 4 or q.shape != k.shape or q.shape != v.shape:
        raise ValueError("q, k, and v must share shape [B, H, S, D]")
    if not q.is_cuda or not k.is_cuda or not v.is_cuda:
        raise ValueError("q, k, and v must be CUDA tensors")
    if q.dtype != k.dtype or q.dtype != v.dtype:
        raise ValueError("q, k, and v must share a dtype")


def torch_attention_reference(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, causal: bool = False) -> torch.Tensor:
    _validate_qkv(q, k, v)
    scale = 1.0 / math.sqrt(q.shape[-1])
    scores = torch.matmul(q.float(), k.float().transpose(-2, -1)) * scale
    if causal:
        query_positions = torch.arange(q.shape[-2], device=q.device)[:, None]
        key_positions = torch.arange(k.shape[-2], device=k.device)[None, :]
        scores = scores.masked_fill(key_positions > query_positions, -float("inf"))
    probabilities = torch.softmax(scores, dim=-1)
    return torch.matmul(probabilities, v.float()).to(q.dtype)


def maybe_compare_with_torch_sdpa(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, causal: bool = False) -> None:
    if not hasattr(F, "scaled_dot_product_attention"):
        print("scaled_dot_product_attention is unavailable in this PyTorch version")
        return
    expected = F.scaled_dot_product_attention(q, k, v, is_causal=causal)
    actual = torch_attention_reference(q, k, v, causal)
    assert_close(f"attention vs SDPA causal={causal}", actual.float(), expected.float(), rtol=1e-2, atol=1e-2)


def main() -> None:
    device = get_device()
    set_seed(0)
    q = torch.randn(2, 4, 128, 64, device=device, dtype=torch.float16)
    k = torch.randn_like(q)
    v = torch.randn_like(q)
    for causal in (False, True):
        output = torch_attention_reference(q, k, v, causal)
        print(f"causal={causal}, output shape={tuple(output.shape)}")
        maybe_compare_with_torch_sdpa(q, k, v, causal)
        print(f"explicit PyTorch attention: {bench(lambda: torch_attention_reference(q, k, v, causal)):.3f} ms")


if __name__ == "__main__":
    main()

