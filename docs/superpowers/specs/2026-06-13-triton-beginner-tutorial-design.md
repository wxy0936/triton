# Triton Beginner Tutorial 00-06 Design

## Goal

Create a self-contained beginner tutorial for Triton covering chapters 00 through 06. Each teaching chapter uses a real Jupyter notebook for step-by-step learning and an equivalent standalone Python program for direct execution.

## Audience and Scope

The audience is Python and PyTorch users who are new to Triton. The tutorial assumes Python 3.10+, PyTorch, Triton, a CUDA GPU, and Jupyter.

The project covers only:

- Chapter 00: environment setup and validation
- Chapter 01: vector addition
- Chapter 02: Triton's program, block, and grid model
- Chapter 03: masks, memory access, and one-dimensional strides
- Chapter 04: unary, binary, and fused elementwise operations
- Chapter 05: row-wise sum and max reductions
- Chapter 06: row-wise fused softmax

Backward implementations, production-level tuning, and chapter 07 or later code are out of scope.

## Project Structure

The repository root is `triton-tutorial/` and contains:

```text
triton-tutorial/
  README.md
  requirements.txt
  common/
    __init__.py
    benchmark.py
    check.py
    utils.py
  chapter_00_setup/
    00_setup.ipynb
    setup_check.py
  chapter_01_vector_add/
    01_vector_add.ipynb
    vector_add.py
  chapter_02_triton_programming_model/
    02_programming_model.ipynb
    programming_model.py
  chapter_03_mask_and_memory/
    03_mask_and_memory.ipynb
    mask_and_memory.py
  chapter_04_unary_binary_ops/
    04_unary_binary_ops.ipynb
    unary_binary_ops.py
  chapter_05_reduction_sum_max/
    05_reduction.ipynb
    reduction.py
  chapter_06_fused_softmax/
    06_fused_softmax.ipynb
    fused_softmax.py
  tests/
    test_project_structure.py
```

`common/__init__.py` is included so imports remain predictable when examples are launched from the project root. `tests/test_project_structure.py` is an internal static acceptance test and does not add another tutorial chapter.

## Teaching Format

Every notebook is a genuine `.ipynb` JSON document whose cells run from top to bottom. Each notebook contains the complete teaching implementation rather than importing the finished implementation from the matching `.py` file.

For chapters 01 through 06, the notebook sequence is:

1. State the chapter's one or two new concepts.
2. Define and run the PyTorch reference.
3. Introduce the Triton kernel in small code cells.
4. Define the Python wrapper.
5. Run correctness checks.
6. Run an accurately synchronized benchmark.
7. Summarize the behavior and trade-offs.

The matching `.py` file contains the same final kernel and wrapper behavior in a concise standalone form. Minor presentation differences are allowed, but inputs, outputs, supported shapes, correctness behavior, and benchmark methodology must remain equivalent.

Chapter 00 is the documented exception: it checks PyTorch, CUDA, Triton, and a simple CUDA tensor operation, but defines no Triton kernel.

## Shared Modules

### `common/benchmark.py`

Expose `bench(fn, warmup=25, rep=100) -> float`, returning median execution time in milliseconds. It uses `triton.testing.do_bench`, which correctly synchronizes CUDA work. Inputs are callable closures so chapters can benchmark PyTorch and Triton with the same API.

### `common/check.py`

Expose `assert_close(name, actual, expected, rtol=1e-4, atol=1e-4)`. It validates shape, computes and prints the maximum absolute error, calls `torch.allclose`, and raises an informative `AssertionError` when values differ.

### `common/utils.py`

Expose:

- `get_device()` to require CUDA and return `torch.device("cuda")`
- `print_gpu_info()` to print PyTorch, Triton, CUDA, and GPU information
- `set_seed(seed=0)` to seed PyTorch CPU and CUDA random generators

CUDA failures use clear `RuntimeError` messages that state a CUDA-capable GPU and CUDA-enabled PyTorch installation are required.

## Chapter Designs

### Chapter 00: Setup

The notebook checks package versions, `torch.cuda.is_available()`, GPU identity, CUDA tensor creation, and a simple PyTorch addition. It also explains that notebooks are expanded teaching versions while `.py` files are equivalent standalone programs.

`setup_check.py` performs the same checks under a main guard and prints `Triton tutorial environment is ready` only after successful CUDA computation.

### Chapter 01: Vector Add

The chapter introduces `@triton.jit`, `tl.program_id(0)`, `tl.arange`, offsets, masks, loads, stores, launch grids, and `BLOCK_SIZE`. One Triton program handles one block of a one-dimensional tensor. The wrapper validates equal shape, CUDA placement, one-dimensional layout, and contiguity.

The kernel uses a mask so arbitrary positive lengths are supported. Correctness compares against `x + y`; the benchmark compares PyTorch and Triton on a moderate one-million-element tensor.

### Chapter 02: Programming Model

The chapter uses a copy kernel to isolate the grid/program model from arithmetic. It demonstrates multiple block sizes and prints the logical offset interval assigned to each program. A lightweight two-dimensional grid demonstration prints program coordinates and ownership ranges on the host without introducing a second complex kernel.

`copy(x, block_size=...)` supports arbitrary one-dimensional contiguous CUDA tensors and checks against `x.clone()`. A small PyTorch-versus-Triton copy benchmark is included to satisfy the common chapter format while keeping performance secondary.

### Chapter 03: Mask and Memory Access

The chapter contrasts a contiguous add kernel with a stride-aware add kernel. The stride-aware kernel receives element strides for both inputs and the output. It supports one-dimensional sliced tensors such as `base[::2]`; it does not attempt arbitrary multidimensional layouts.

The output is newly allocated and contiguous, while input addresses are computed as `offsets * stride`. Both contiguous and sliced inputs are checked against PyTorch. Benchmarks compare PyTorch and Triton separately for contiguous and sliced cases.

### Chapter 04: Unary and Binary Operations

The chapter defines square, ReLU, and fused add-plus-ReLU kernels. Sigmoid is explained briefly as another unary operation but is not implemented, keeping the chapter focused on the two new concepts: reusable elementwise structure and fusion.

All wrappers accept arbitrary one-dimensional contiguous CUDA tensors, use masks, and return new tensors. Correctness covers all three operations. The benchmark compares `torch.relu(x + y)` against the fused Triton kernel and explains intermediate tensor and memory-traffic reduction.

### Chapter 05: Row-wise Reduction

The input is a contiguous two-dimensional CUDA tensor `[M, N]`. One Triton program handles one row. The wrapper chooses `BLOCK_SIZE = triton.next_power_of_2(N)`, and masks columns beyond `N`.

Sum loads masked values with `other=0.0`; max loads with `other=-float("inf")`. The kernels use `tl.sum` and `tl.max`. Correctness covers non-power-of-two widths. Benchmarks run several moderate widths while keeping memory use small.

### Chapter 06: Fused Softmax

One program processes one row of a contiguous `[M, N]` CUDA tensor. It loads a power-of-two block, masks invalid columns with negative infinity, subtracts the row maximum for numerical stability, exponentiates, sums, divides, and stores valid outputs.

The wrapper chooses `BLOCK_SIZE = triton.next_power_of_2(N)` and rejects empty dimensions. It also enforces a documented maximum block size of 65,536 elements per row to keep this teaching kernel's resource use bounded and produce a clear error for overly wide rows.

Correctness compares with `torch.softmax(x, dim=1)`. Benchmarks use several moderate `(M, N)` shapes and explain why fusion avoids materializing intermediate tensors.

## Input Validation and Errors

Wrappers reject unsupported inputs before launching kernels. Checks cover CUDA placement, expected dimensionality, matching shapes where applicable, contiguity for chapters that require it, and non-empty dimensions. Chapter 03 intentionally accepts positive-stride one-dimensional slices.

The examples use floating-point CUDA tensors. Default demonstrations use `torch.float32` for clear tolerances and predictable beginner behavior.

## Benchmark Policy

All GPU timing goes through `common.benchmark.bench` and `triton.testing.do_bench`. No chapter uses `time.time()` for CUDA timing. Benchmarks allocate inputs and outputs before timing where practical, and closures contain only the operation being measured.

Default sizes are intentionally moderate: about one million elements for one-dimensional kernels and at most roughly one million elements for two-dimensional examples. Benchmark results are educational observations, not universal performance claims.

## Verification

Static verification will check:

- Every required file exists and no chapter 07+ directory exists.
- Every notebook parses as JSON and contains both Markdown and code cells.
- Every Python file parses with `compileall`.
- Every chapter Python program has a main guard.
- Required kernel and wrapper names are present.
- CUDA timing does not use `time.time()`.

Where a CUDA/Triton environment is available, each chapter program should also be executed directly. In an environment without CUDA or Triton, static checks still run, and the inability to execute GPU kernels is reported explicitly rather than hidden.

## Documentation

`README.md` introduces the tutorial, installation, notebook startup, standalone script execution, chapter index, learning path, and the teaching-oriented performance disclaimer. It may preview future topics by name but will not include code or directories beyond chapter 06.

`requirements.txt` includes `torch`, `triton`, `jupyter`, and `notebook` without restrictive pins so learners can install versions appropriate for their CUDA platform.

## Acceptance Criteria

The project is complete when all requested files exist, notebooks are valid and sequentially runnable, Python files are standalone from the project root, chapters 01-06 include references, Triton implementations, correctness checks, and synchronized benchmarks, and the static acceptance test passes.
