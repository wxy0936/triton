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
def add_contiguous_kernel(x_ptr, y_ptr, output_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    program_id = tl.program_id(axis=0)
    offsets = program_id * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    y = tl.load(y_ptr + offsets, mask=mask, other=0.0)
    tl.store(output_ptr + offsets, x + y, mask=mask)


@triton.jit
def add_strided_kernel(
    x_ptr,
    y_ptr,
    output_ptr,
    n_elements,
    x_stride,
    y_stride,
    BLOCK_SIZE: tl.constexpr,
):
    # Logical offsets become physical addresses through each input stride.
    program_id = tl.program_id(axis=0)
    offsets = program_id * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets * x_stride, mask=mask, other=0.0)
    y = tl.load(y_ptr + offsets * y_stride, mask=mask, other=0.0)
    tl.store(output_ptr + offsets, x + y, mask=mask)


def _validate_pair(x: torch.Tensor, y: torch.Tensor) -> None:
    if x.ndim != 1 or y.ndim != 1 or x.shape != y.shape:
        raise ValueError("x and y must be 1D tensors with the same shape")
    if not x.is_cuda or not y.is_cuda:
        raise ValueError("x and y must be CUDA tensors")
    if x.dtype != y.dtype:
        raise ValueError("x and y must have the same dtype")
    if x.stride(0) <= 0 or y.stride(0) <= 0:
        raise ValueError("only positive one-dimensional strides are supported")


def add_contiguous(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    _validate_pair(x, y)
    if not x.is_contiguous() or not y.is_contiguous():
        raise ValueError("add_contiguous expects contiguous tensors")
    output = torch.empty_like(x)
    if x.numel() > 0:
        grid = (triton.cdiv(x.numel(), 256),)
        add_contiguous_kernel[grid](x, y, output, x.numel(), BLOCK_SIZE=256)
    return output


def add_strided(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    _validate_pair(x, y)
    output = torch.empty_like(x, memory_format=torch.contiguous_format)
    if x.numel() > 0:
        grid = (triton.cdiv(x.numel(), 256),)
        add_strided_kernel[grid](
            x,
            y,
            output,
            x.numel(),
            x.stride(0),
            y.stride(0),
            BLOCK_SIZE=256,
        )
    return output


def main() -> None:
    device = get_device()
    set_seed(0)
    n_elements = 500_003

    x = torch.randn(n_elements, device=device)
    y = torch.randn(n_elements, device=device)
    assert_close("contiguous add", add_contiguous(x, y), x + y)

    x_base = torch.randn(n_elements * 2, device=device)
    y_base = torch.randn(n_elements * 3, device=device)
    x_slice = x_base[::2]
    y_slice = y_base[::3]
    print(f"x_slice contiguous={x_slice.is_contiguous()}, stride={x_slice.stride()}")
    print(f"y_slice contiguous={y_slice.is_contiguous()}, stride={y_slice.stride()}")
    assert_close("strided add", add_strided(x_slice, y_slice), x_slice + y_slice)

    print(f"Contiguous PyTorch: {bench(lambda: x + y):.3f} ms")
    print(f"Contiguous Triton:  {bench(lambda: add_contiguous(x, y)):.3f} ms")
    print(f"Strided PyTorch:    {bench(lambda: x_slice + y_slice):.3f} ms")
    print(f"Strided Triton:     {bench(lambda: add_strided(x_slice, y_slice)):.3f} ms")


if __name__ == "__main__":
    main()

