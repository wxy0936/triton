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
def copy_kernel(x_ptr, y_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    # tl.arange creates a vector of offsets; it is not a Python loop.
    program_id = tl.program_id(axis=0)
    offsets = program_id * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    values = tl.load(x_ptr + offsets, mask=mask)
    tl.store(y_ptr + offsets, values, mask=mask)


def copy(x: torch.Tensor, block_size: int = 256) -> torch.Tensor:
    if x.ndim != 1 or not x.is_cuda or not x.is_contiguous():
        raise ValueError("copy expects a contiguous 1D CUDA tensor")
    if block_size <= 0 or block_size & (block_size - 1):
        raise ValueError("block_size must be a positive power of two")
    y = torch.empty_like(x)
    if x.numel() > 0:
        grid = (triton.cdiv(x.numel(), block_size),)
        copy_kernel[grid](x, y, x.numel(), BLOCK_SIZE=block_size)
    return y


def demo_1d_grid(n_elements: int = 20, block_size: int = 8) -> None:
    grid = triton.cdiv(n_elements, block_size)
    print(f"1D grid: ({grid},), N={n_elements}, BLOCK_SIZE={block_size}")
    for program_id in range(grid):
        start = program_id * block_size
        stop = min(start + block_size, n_elements)
        print(f"  program {program_id}: offsets [{start}, {stop})")


def demo_2d_grid(rows: int = 3, cols: int = 10, block_cols: int = 4) -> None:
    grid = (rows, triton.cdiv(cols, block_cols))
    print(f"2D grid: {grid}, shape=({rows}, {cols})")
    for row_program in range(grid[0]):
        for col_program in range(grid[1]):
            start = col_program * block_cols
            stop = min(start + block_cols, cols)
            print(f"  program ({row_program}, {col_program}): row {row_program}, cols [{start}, {stop})")


def main() -> None:
    device = get_device()
    set_seed(0)
    demo_1d_grid()
    demo_2d_grid()

    x = torch.randn(1003, device=device)
    for block_size in (64, 128, 256):
        actual = copy(x, block_size)
        assert_close(f"copy BLOCK_SIZE={block_size}", actual, x)

    torch_ms = bench(lambda: x.clone())
    triton_ms = bench(lambda: copy(x, 256))
    print(f"PyTorch clone: {torch_ms:.3f} ms")
    print(f"Triton copy:   {triton_ms:.3f} ms")


if __name__ == "__main__":
    main()
