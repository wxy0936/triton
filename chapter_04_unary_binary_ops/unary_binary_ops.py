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
def square_kernel(x_ptr, output_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    offsets = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask)
    tl.store(output_ptr + offsets, x * x, mask=mask)


@triton.jit
def relu_kernel(x_ptr, output_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    offsets = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask)
    tl.store(output_ptr + offsets, tl.maximum(x, 0.0), mask=mask)


@triton.jit
def add_relu_kernel(x_ptr, y_ptr, output_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    # Addition and ReLU happen before one final store, with no intermediate tensor.
    offsets = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask)
    y = tl.load(y_ptr + offsets, mask=mask)
    tl.store(output_ptr + offsets, tl.maximum(x + y, 0.0), mask=mask)


def _validate_1d(x: torch.Tensor) -> None:
    if x.ndim != 1 or not x.is_cuda or not x.is_contiguous():
        raise ValueError("expected a contiguous 1D CUDA tensor")


def _launch_unary(kernel, x: torch.Tensor) -> torch.Tensor:
    _validate_1d(x)
    output = torch.empty_like(x)
    if x.numel() > 0:
        grid = (triton.cdiv(x.numel(), 256),)
        kernel[grid](x, output, x.numel(), BLOCK_SIZE=256)
    return output


def square(x: torch.Tensor) -> torch.Tensor:
    return _launch_unary(square_kernel, x)


def relu(x: torch.Tensor) -> torch.Tensor:
    return _launch_unary(relu_kernel, x)


def add_relu(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    _validate_1d(x)
    _validate_1d(y)
    if x.shape != y.shape or x.dtype != y.dtype:
        raise ValueError("x and y must have the same shape and dtype")
    output = torch.empty_like(x)
    if x.numel() > 0:
        grid = (triton.cdiv(x.numel(), 256),)
        add_relu_kernel[grid](x, y, output, x.numel(), BLOCK_SIZE=256)
    return output


def main() -> None:
    device = get_device()
    set_seed(0)
    x = torch.randn(1_000_003, device=device)
    y = torch.randn_like(x)

    assert_close("square", square(x), x * x)
    assert_close("relu", relu(x), torch.relu(x))
    assert_close("add + relu", add_relu(x, y), torch.relu(x + y))

    torch_ms = bench(lambda: torch.relu(x + y))
    triton_ms = bench(lambda: add_relu(x, y))
    print(f"PyTorch add then ReLU: {torch_ms:.3f} ms")
    print(f"Triton fused add+ReLU: {triton_ms:.3f} ms")


if __name__ == "__main__":
    main()

