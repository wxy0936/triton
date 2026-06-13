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


MAX_FUSED_SIZE = 65_536


@triton.jit
def softmax_kernel(x_ptr, output_ptr, n_cols, BLOCK_SIZE: tl.constexpr):
    # One program loads, normalizes, and stores one complete row.
    row = tl.program_id(axis=0)
    cols = tl.arange(0, BLOCK_SIZE)
    mask = cols < n_cols
    row_values = tl.load(x_ptr + row * n_cols + cols, mask=mask, other=-float("inf"))
    row_values = row_values - tl.max(row_values, axis=0)
    numerator = tl.exp(row_values)
    denominator = tl.sum(numerator, axis=0)
    output = numerator / denominator
    tl.store(output_ptr + row * n_cols + cols, output, mask=mask)


def softmax(x: torch.Tensor) -> torch.Tensor:
    if x.ndim != 2 or not x.is_cuda or not x.is_contiguous():
        raise ValueError("softmax expects a contiguous 2D CUDA tensor")
    n_rows, n_cols = x.shape
    if n_rows == 0 or n_cols == 0:
        raise ValueError("softmax expects non-empty M and N dimensions")
    block_size = triton.next_power_of_2(n_cols)
    if block_size > MAX_FUSED_SIZE:
        raise ValueError(
            f"N={n_cols} is too large for this teaching kernel; "
            f"the next power of two must be <= {MAX_FUSED_SIZE}"
        )

    output = torch.empty_like(x)
    num_warps = 8 if block_size >= 2048 else 4
    softmax_kernel[(n_rows,)](
        x,
        output,
        n_cols,
        BLOCK_SIZE=block_size,
        num_warps=num_warps,
    )
    return output


def main() -> None:
    device = get_device()
    set_seed(0)
    for shape in ((256, 127), (1024, 512), (512, 1024)):
        x = torch.randn(*shape, device=device)
        expected = torch.softmax(x, dim=1)
        actual = softmax(x)
        assert_close(f"softmax shape={shape}", actual, expected, rtol=1e-3, atol=1e-5)
        torch_ms = bench(lambda: torch.softmax(x, dim=1))
        triton_ms = bench(lambda: softmax(x))
        print(f"shape={shape}: PyTorch={torch_ms:.3f} ms, Triton={triton_ms:.3f} ms")


if __name__ == "__main__":
    main()
