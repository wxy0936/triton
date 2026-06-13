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
def vector_add_kernel(x_ptr, y_ptr, output_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    # Each program owns one block of consecutive elements.
    program_id = tl.program_id(axis=0)
    offsets = program_id * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask)
    y = tl.load(y_ptr + offsets, mask=mask)
    tl.store(output_ptr + offsets, x + y, mask=mask)


def vector_add(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    if x.ndim != 1 or y.ndim != 1:
        raise ValueError("vector_add expects two 1D tensors")
    if x.shape != y.shape:
        raise ValueError("x and y must have the same shape")
    if not x.is_cuda or not y.is_cuda:
        raise ValueError("x and y must be CUDA tensors")
    if not x.is_contiguous() or not y.is_contiguous():
        raise ValueError("vector_add expects contiguous tensors")
    if x.dtype != y.dtype:
        raise ValueError("x and y must have the same dtype")

    output = torch.empty_like(x)
    if x.numel() == 0:
        return output
    grid = lambda meta: (triton.cdiv(x.numel(), meta["BLOCK_SIZE"]),)
    vector_add_kernel[grid](x, y, output, x.numel(), BLOCK_SIZE=256)
    return output


def main() -> None:
    device = get_device()
    set_seed(0)
    n_elements = 1_000_003
    x = torch.randn(n_elements, device=device)
    y = torch.randn(n_elements, device=device)

    expected = x + y
    actual = vector_add(x, y)
    assert_close("vector add", actual, expected)

    torch_ms = bench(lambda: x + y)
    triton_ms = bench(lambda: vector_add(x, y))
    print(f"PyTorch: {torch_ms:.3f} ms")
    print(f"Triton:  {triton_ms:.3f} ms")


if __name__ == "__main__":
    main()

