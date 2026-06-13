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


@triton.jit
def matmul_kernel(
    a_ptr,
    b_ptr,
    c_ptr,
    M,
    N,
    K,
    stride_am,
    stride_ak,
    stride_bk,
    stride_bn,
    stride_cm,
    stride_cn,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    # One program computes one BLOCK_M x BLOCK_N output tile.
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    offsets_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offsets_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offsets_k = tl.arange(0, BLOCK_K)
    accumulator = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for k_start in range(0, tl.cdiv(K, BLOCK_K)):
        k = k_start * BLOCK_K + offsets_k
        a_ptrs = a_ptr + offsets_m[:, None] * stride_am + k[None, :] * stride_ak
        b_ptrs = b_ptr + k[:, None] * stride_bk + offsets_n[None, :] * stride_bn
        a = tl.load(a_ptrs, mask=(offsets_m[:, None] < M) & (k[None, :] < K), other=0.0)
        b = tl.load(b_ptrs, mask=(k[:, None] < K) & (offsets_n[None, :] < N), other=0.0)
        accumulator += tl.dot(a, b)

    c_ptrs = c_ptr + offsets_m[:, None] * stride_cm + offsets_n[None, :] * stride_cn
    c_mask = (offsets_m[:, None] < M) & (offsets_n[None, :] < N)
    tl.store(c_ptrs, accumulator, mask=c_mask)


def matmul(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    if a.ndim != 2 or b.ndim != 2 or a.shape[1] != b.shape[0]:
        raise ValueError("expected A[M, K] and B[K, N]")
    if not a.is_cuda or not b.is_cuda or not a.is_contiguous() or not b.is_contiguous():
        raise ValueError("a and b must be contiguous CUDA tensors")
    if a.dtype != torch.float16 or b.dtype != torch.float16:
        raise ValueError("this teaching kernel expects fp16 inputs")

    M, K = a.shape
    _, N = b.shape
    c = torch.empty((M, N), device=a.device, dtype=torch.float16)
    if M == 0 or N == 0 or K == 0:
        return c
    grid = (triton.cdiv(M, 32), triton.cdiv(N, 32))
    matmul_kernel[grid](
        a, b, c, M, N, K,
        a.stride(0), a.stride(1), b.stride(0), b.stride(1), c.stride(0), c.stride(1),
        BLOCK_M=32, BLOCK_N=32, BLOCK_K=32, num_warps=4,
    )
    return c


def main() -> None:
    device = get_device()
    set_seed(0)
    for shape in ((127, 193, 89), (512, 512, 512)):
        M, N, K = shape
        a = torch.randn(M, K, device=device, dtype=torch.float16)
        b = torch.randn(K, N, device=device, dtype=torch.float16)
        expected = torch.matmul(a, b)
        actual = matmul(a, b)
        assert_close(f"matmul M={M} N={N} K={K}", actual.float(), expected.float(), rtol=1e-2, atol=1e-2)
        print(f"PyTorch: {bench(lambda: torch.matmul(a, b)):.3f} ms")
        print(f"Triton:  {bench(lambda: matmul(a, b)):.3f} ms")


if __name__ == "__main__":
    main()

