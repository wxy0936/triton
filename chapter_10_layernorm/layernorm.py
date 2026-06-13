import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch
import torch.nn.functional as F
import triton
import triton.language as tl

from common.benchmark import bench
from common.check import assert_close
from common.utils import get_device, set_seed


MAX_BLOCK_SIZE = 65_536


@triton.jit
def layernorm_kernel(x_ptr, weight_ptr, bias_ptr, output_ptr, n_cols, eps: tl.constexpr, BLOCK_SIZE: tl.constexpr):
    # One program normalizes one row; reductions and affine math use fp32.
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK_SIZE)
    mask = cols < n_cols
    x = tl.load(x_ptr + row * n_cols + cols, mask=mask, other=0.0).to(tl.float32)
    mean = tl.sum(x, axis=0) / n_cols
    centered = tl.where(mask, x - mean, 0.0)
    variance = tl.sum(centered * centered, axis=0) / n_cols
    inv_std = tl.rsqrt(variance + eps)
    weight = tl.load(weight_ptr + cols, mask=mask, other=0.0).to(tl.float32)
    bias = tl.load(bias_ptr + cols, mask=mask, other=0.0).to(tl.float32)
    output = centered * inv_std * weight + bias
    tl.store(output_ptr + row * n_cols + cols, output, mask=mask)


def layernorm(x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor, eps: float = 1e-5) -> torch.Tensor:
    if x.ndim != 2 or weight.ndim != 1 or bias.ndim != 1:
        raise ValueError("expected x[M,N], weight[N], and bias[N]")
    if weight.shape != (x.shape[1],) or bias.shape != (x.shape[1],):
        raise ValueError("weight and bias must match the last dimension of x")
    if not x.is_cuda or not weight.is_cuda or not bias.is_cuda:
        raise ValueError("x, weight, and bias must be CUDA tensors")
    if not x.is_contiguous() or not weight.is_contiguous() or not bias.is_contiguous():
        raise ValueError("x, weight, and bias must be contiguous")
    if x.dtype not in (torch.float16, torch.float32) or weight.dtype != x.dtype or bias.dtype != x.dtype:
        raise ValueError("all inputs must share fp16 or fp32 dtype")
    M, N = x.shape
    if M == 0 or N == 0:
        raise ValueError("M and N must be positive")
    block_size = triton.next_power_of_2(N)
    if block_size > MAX_BLOCK_SIZE:
        raise ValueError(f"N={N} is too large; padded width must be <= {MAX_BLOCK_SIZE}")
    output = torch.empty_like(x)
    layernorm_kernel[(M,)](x, weight, bias, output, N, eps=eps, BLOCK_SIZE=block_size, num_warps=8 if block_size >= 2048 else 4)
    return output


def main() -> None:
    device = get_device()
    set_seed(0)
    for N in (127, 513, 1024):
        x = torch.randn(1024, N, device=device, dtype=torch.float16)
        weight = torch.randn(N, device=device, dtype=torch.float16)
        bias = torch.randn(N, device=device, dtype=torch.float16)
        expected = F.layer_norm(x, (N,), weight, bias, 1e-5)
        actual = layernorm(x, weight, bias)
        assert_close(f"layernorm N={N}", actual.float(), expected.float(), rtol=1e-2, atol=1e-2)
        print(f"N={N}: PyTorch={bench(lambda: F.layer_norm(x, (N,), weight, bias, 1e-5)):.3f} ms, Triton={bench(lambda: layernorm(x, weight, bias)):.3f} ms")


if __name__ == "__main__":
    main()
