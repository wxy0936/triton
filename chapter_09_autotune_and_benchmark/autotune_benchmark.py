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


@triton.autotune(
    configs=[
        triton.Config({"BLOCK_M": 32, "BLOCK_N": 32, "BLOCK_K": 32, "GROUP_SIZE_M": 4}, num_warps=4, num_stages=2),
        triton.Config({"BLOCK_M": 64, "BLOCK_N": 64, "BLOCK_K": 32, "GROUP_SIZE_M": 8}, num_warps=4, num_stages=3),
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 64, "BLOCK_K": 32, "GROUP_SIZE_M": 8}, num_warps=8, num_stages=3),
    ],
    key=["M", "N", "K"],
)
@triton.jit
def autotuned_matmul_kernel(a_ptr, b_ptr, c_ptr, M, N, K, stride_am, stride_ak, stride_bk, stride_bn, stride_cm, stride_cn, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr, GROUP_SIZE_M: tl.constexpr):
    pid = tl.program_id(0)
    num_pid_m = tl.cdiv(M, BLOCK_M)
    num_pid_n = tl.cdiv(N, BLOCK_N)
    width = GROUP_SIZE_M * num_pid_n
    group_id = pid // width
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = tl.minimum(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + pid % group_size_m
    pid_n = (pid % width) // group_size_m
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)
    acc = tl.zeros((BLOCK_M, BLOCK_N), tl.float32)
    for k_block in range(0, tl.cdiv(K, BLOCK_K)):
        k = k_block * BLOCK_K + offs_k
        a = tl.load(a_ptr + offs_m[:, None] * stride_am + k[None, :] * stride_ak, mask=(offs_m[:, None] < M) & (k[None, :] < K), other=0.0)
        b = tl.load(b_ptr + k[:, None] * stride_bk + offs_n[None, :] * stride_bn, mask=(k[:, None] < K) & (offs_n[None, :] < N), other=0.0)
        acc += tl.dot(a, b)
    tl.store(c_ptr + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn, acc, mask=(offs_m[:, None] < M) & (offs_n[None, :] < N))


def autotuned_matmul(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    if a.ndim != 2 or b.ndim != 2 or a.shape[1] != b.shape[0]:
        raise ValueError("expected A[M, K] and B[K, N]")
    if not a.is_cuda or not b.is_cuda or not a.is_contiguous() or not b.is_contiguous():
        raise ValueError("a and b must be contiguous CUDA tensors")
    if a.dtype != torch.float16 or b.dtype != torch.float16:
        raise ValueError("autotuned_matmul expects fp16 inputs")
    M, K = a.shape
    N = b.shape[1]
    c = torch.empty((M, N), device=a.device, dtype=torch.float16)
    grid = lambda meta: (triton.cdiv(M, meta["BLOCK_M"]) * triton.cdiv(N, meta["BLOCK_N"]),)
    autotuned_matmul_kernel[grid](a, b, c, M, N, K, a.stride(0), a.stride(1), b.stride(0), b.stride(1), c.stride(0), c.stride(1))
    return c


def benchmark_matmul_shapes() -> None:
    device = get_device()
    print(f"{'M,N,K':>18} | {'PyTorch ms':>10} | {'Triton ms':>10}")
    print("-" * 46)
    for M, N, K in ((256, 256, 256), (512, 512, 512), (768, 512, 256)):
        a = torch.randn(M, K, device=device, dtype=torch.float16)
        b = torch.randn(K, N, device=device, dtype=torch.float16)
        actual = autotuned_matmul(a, b)
        expected = torch.matmul(a, b)
        assert_close(f"autotuned {M}x{N}x{K}", actual.float(), expected.float(), rtol=1e-2, atol=1e-2)
        print(f"{str((M, N, K)):>18} | {bench(lambda: torch.matmul(a, b)):10.3f} | {bench(lambda: autotuned_matmul(a, b)):10.3f}")


if __name__ == "__main__":
    set_seed(0)
    benchmark_matmul_shapes()

