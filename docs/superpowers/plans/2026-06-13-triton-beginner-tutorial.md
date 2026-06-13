# Triton Beginner Tutorial Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a complete, beginner-friendly Triton tutorial for chapters 00-06 with self-contained notebooks, equivalent standalone scripts, shared utilities, and static acceptance checks.

**Architecture:** Keep each chapter independent except for small helpers in `common/`. Put complete teaching code in every notebook and equivalent final code in the matching Python script. Use `triton.testing.do_bench` for all CUDA benchmarks and static tests to validate structure and notebook integrity when a CUDA runtime is unavailable.

**Tech Stack:** Python 3.10+, PyTorch, Triton, Jupyter Notebook, standard-library `unittest`

---

## File Map

- `README.md`: installation, execution, chapter index, learning path, disclaimer.
- `requirements.txt`: runtime and notebook dependencies.
- `common/benchmark.py`: synchronized benchmark helper.
- `common/check.py`: numerical correctness helper.
- `common/utils.py`: CUDA validation, GPU information, and deterministic seeding.
- `chapter_00_setup/*`: environment validation without a Triton kernel.
- `chapter_01_vector_add/*`: first masked one-dimensional kernel.
- `chapter_02_triton_programming_model/*`: copy kernel and grid demonstrations.
- `chapter_03_mask_and_memory/*`: contiguous and stride-aware addition.
- `chapter_04_unary_binary_ops/*`: square, ReLU, and fused add-ReLU.
- `chapter_05_reduction_sum_max/*`: row-wise sum and max.
- `chapter_06_fused_softmax/*`: stable row-wise fused softmax.
- `tests/test_project_structure.py`: static acceptance tests.

### Task 1: Static Acceptance Tests

**Files:**
- Create: `tests/test_project_structure.py`

- [ ] Write tests that assert every requested file exists, notebooks parse with Markdown and code cells, scripts contain main guards, required symbols are present, no chapter 07+ directory exists, and no CUDA timing uses `time.time()`.
- [ ] Run `python -m unittest tests.test_project_structure -v` and confirm it fails because tutorial files are not yet present.

### Task 2: Shared Utilities and Project Documentation

**Files:**
- Create: `README.md`
- Create: `requirements.txt`
- Create: `common/__init__.py`
- Create: `common/benchmark.py`
- Create: `common/check.py`
- Create: `common/utils.py`

- [ ] Implement `bench`, `assert_close`, `get_device`, `print_gpu_info`, and `set_seed` with clear CUDA errors.
- [ ] Document installation, notebook and script commands, chapter contents, learning path, and teaching-performance scope.
- [ ] Run `python -m compileall common` and confirm success.

### Task 3: Chapter 00 Environment Check

**Files:**
- Create: `chapter_00_setup/setup_check.py`
- Create: `chapter_00_setup/00_setup.ipynb`

- [ ] Implement the standalone CUDA/Triton environment check and exact success message.
- [ ] Create a sequential notebook covering package checks, versions, GPU identity, CUDA tensors, a PyTorch operation, and notebook/script roles.
- [ ] Compile the script and parse the notebook JSON.

### Task 4: Chapter 01 Vector Add

**Files:**
- Create: `chapter_01_vector_add/vector_add.py`
- Create: `chapter_01_vector_add/01_vector_add.ipynb`

- [ ] Implement a masked vector-add kernel and validated wrapper for arbitrary one-dimensional lengths.
- [ ] Add PyTorch reference, correctness checks, and synchronized PyTorch-versus-Triton benchmark.
- [ ] Create the complete step-by-step notebook explaining `BLOCK_SIZE` and launch grids.
- [ ] Compile the script and parse the notebook JSON.

### Task 5: Chapter 02 Programming Model

**Files:**
- Create: `chapter_02_triton_programming_model/programming_model.py`
- Create: `chapter_02_triton_programming_model/02_programming_model.ipynb`

- [ ] Implement `copy_kernel`, `copy`, `demo_1d_grid`, and `demo_2d_grid`.
- [ ] Add correctness and synchronized copy benchmarks across block sizes.
- [ ] Create the notebook explaining grids, programs, vector offsets, and a lightweight two-dimensional grid.
- [ ] Compile the script and parse the notebook JSON.

### Task 6: Chapter 03 Mask and Memory Access

**Files:**
- Create: `chapter_03_mask_and_memory/mask_and_memory.py`
- Create: `chapter_03_mask_and_memory/03_mask_and_memory.ipynb`

- [ ] Implement contiguous and stride-aware one-dimensional add kernels and wrappers.
- [ ] Test contiguous and positive-stride sliced tensors against PyTorch and benchmark both cases.
- [ ] Create the notebook explaining masks, `other`, strides, contiguous storage, and sliced tensors.
- [ ] Compile the script and parse the notebook JSON.

### Task 7: Chapter 04 Elementwise Operations and Fusion

**Files:**
- Create: `chapter_04_unary_binary_ops/unary_binary_ops.py`
- Create: `chapter_04_unary_binary_ops/04_unary_binary_ops.ipynb`

- [ ] Implement square, ReLU, and fused add-ReLU kernels and wrappers.
- [ ] Add correctness coverage and synchronized fused-versus-PyTorch benchmark.
- [ ] Create the notebook explaining elementwise structure and memory-traffic reduction through fusion.
- [ ] Compile the script and parse the notebook JSON.

### Task 8: Chapter 05 Row-wise Reductions

**Files:**
- Create: `chapter_05_reduction_sum_max/reduction.py`
- Create: `chapter_05_reduction_sum_max/05_reduction.ipynb`

- [ ] Implement row sum and max using one program per row, power-of-two blocks, masks, and correct neutral values.
- [ ] Add correctness checks and synchronized benchmarks for multiple non-power-of-two widths.
- [ ] Create the notebook explaining `tl.sum`, `tl.max`, row ownership, and `BLOCK_SIZE` versus `N`.
- [ ] Compile the script and parse the notebook JSON.

### Task 9: Chapter 06 Fused Softmax

**Files:**
- Create: `chapter_06_fused_softmax/fused_softmax.py`
- Create: `chapter_06_fused_softmax/06_fused_softmax.ipynb`

- [ ] Implement stable row-wise fused softmax with a power-of-two block and a clear 65,536-column teaching limit.
- [ ] Add correctness checks and synchronized benchmarks over moderate shapes.
- [ ] Create the notebook explaining numerical stability and avoided intermediate tensors.
- [ ] Compile the script and parse the notebook JSON.

### Task 10: Final Verification

**Files:**
- Verify all project files.

- [ ] Run `python -m compileall common chapter_00_setup chapter_01_vector_add chapter_02_triton_programming_model chapter_03_mask_and_memory chapter_04_unary_binary_ops chapter_05_reduction_sum_max chapter_06_fused_softmax tests`.
- [ ] Run `python -m unittest tests.test_project_structure -v` and require all tests to pass.
- [ ] If CUDA, PyTorch, and Triton are available, run every chapter script; otherwise report the unavailable runtime and retain static verification evidence.
- [ ] Inspect the final file tree and confirm there is no chapter 07+ code.
