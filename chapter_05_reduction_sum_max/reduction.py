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
def row_sum_kernel(x_ptr, output_ptr, n_cols, BLOCK_SIZE: tl.constexpr):
    # One program reduces one row.
    row = tl.program_id(axis=0)
    cols = tl.arange(0, BLOCK_SIZE)
    mask = cols < n_cols
    values = tl.load(x_ptr + row * n_cols + cols, mask=mask, other=0.0)
    tl.store(output_ptr + row, tl.sum(values, axis=0))


@triton.jit
def row_max_kernel(x_ptr, output_ptr, n_cols, BLOCK_SIZE: tl.constexpr):
    row = tl.program_id(axis=0)
    cols = tl.arange(0, BLOCK_SIZE)
    mask = cols < n_cols
    values = tl.load(x_ptr + row * n_cols + cols, mask=mask, other=-float("inf"))
    tl.store(output_ptr + row, tl.max(values, axis=0))


def _validate_matrix(x: torch.Tensor) -> None:
    if x.ndim != 2 or not x.is_cuda or not x.is_contiguous():
        raise ValueError("expected a contiguous 2D CUDA tensor")
    if x.shape[1] == 0:
        raise ValueError("the reduction dimension N must be positive")


def _launch_reduction(kernel, x: torch.Tensor) -> torch.Tensor:
    _validate_matrix(x)
    n_rows, n_cols = x.shape
    output = torch.empty(n_rows, device=x.device, dtype=x.dtype)
    if n_rows > 0:
        block_size = triton.next_power_of_2(n_cols)
        num_warps = 8 if block_size >= 2048 else 4
        kernel[(n_rows,)](x, output, n_cols, BLOCK_SIZE=block_size, num_warps=num_warps)
    return output


def row_sum(x: torch.Tensor) -> torch.Tensor:
    return _launch_reduction(row_sum_kernel, x)


def row_max(x: torch.Tensor) -> torch.Tensor:
    return _launch_reduction(row_max_kernel, x)


def main() -> None:
    device = get_device()
    set_seed(0)
    for n_cols in (127, 513, 1024):
        x = torch.randn(1024, n_cols, device=device)
        assert_close(f"row sum N={n_cols}", row_sum(x), torch.sum(x, dim=1), rtol=1e-3, atol=1e-3)
        assert_close(f"row max N={n_cols}", row_max(x), torch.max(x, dim=1).values)
        print(
            f"N={n_cols:4d} sum: torch={bench(lambda: torch.sum(x, dim=1)):.3f} ms, "
            f"triton={bench(lambda: row_sum(x)):.3f} ms"
        )
        print(
            f"N={n_cols:4d} max: torch={bench(lambda: torch.max(x, dim=1).values):.3f} ms, "
            f"triton={bench(lambda: row_max(x)):.3f} ms"
        )


if __name__ == "__main__":
    main()

