# Triton Tutorial Chapters 07-14 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Extend the existing Triton tutorial with complete chapters 07-14, preserving chapter independence, teaching style, synchronized benchmarks, and forward-only scope.

**Architecture:** Chapters 07-10 build matmul, autotune, and LayerNorm kernels; Chapters 11-12 establish attention and online-softmax references in PyTorch; Chapters 13-14 each contain the same teaching FlashAttention kernel and core wrapper helper. Static tests compare only normalized AST for the copied kernel and core validation/launch helper.

**Tech Stack:** Python 3.10+, PyTorch, Triton, Jupyter Notebook, standard-library `unittest`

---

### Task 1: Extend Static Acceptance Tests

**Files:**
- Modify: `tests/test_project_structure.py`

- [ ] Require all Chapter 07-14 notebooks and scripts and their specified symbols.
- [ ] Replace the old Chapter 07+ prohibition with a Chapter 15+ prohibition.
- [ ] Add normalized AST equality checks for `flash_attention_kernel` and `_flash_attention_forward` only.
- [ ] Reject cross-chapter imports and `common.attention` in Chapters 13-14.
- [ ] Run the suite and confirm failure because Chapter 07-14 files do not exist.

### Task 2: Implement Chapters 07-10 Scripts

**Files:**
- Create: `chapter_07_matmul_baseline/matmul_baseline.py`
- Create: `chapter_08_matmul_optimized/matmul_optimized.py`
- Create: `chapter_09_autotune_and_benchmark/autotune_benchmark.py`
- Create: `chapter_10_layernorm/layernorm.py`

- [ ] Implement baseline tiled matmul with masked non-divisible boundaries and fp32 accumulation.
- [ ] Implement grouped matmul plus an intentionally duplicated local baseline for standalone comparison.
- [ ] Implement a three-configuration autotuned matmul and clear benchmark table.
- [ ] Implement row-wise LayerNorm with fp32 mean/variance and a documented width limit.
- [ ] Compile all four scripts.

### Task 3: Implement Chapters 11-12 Scripts

**Files:**
- Create: `chapter_11_attention_intro/attention_intro.py`
- Create: `chapter_12_online_softmax/online_softmax.py`

- [ ] Implement explicit PyTorch attention and optional SDPA comparison for causal and non-causal modes.
- [ ] Implement blockwise online attention with correct absolute causal indices, self-attention availability, and an `l == 0` guard.
- [ ] Add synchronized benchmarks and compile both scripts.

### Task 4: Implement Chapter 13 FlashAttention

**Files:**
- Create: `chapter_13_flashattention_v1/flashattention_v1.py`

- [ ] Implement explicit PyTorch reference.
- [ ] Implement stable fp32 online-softmax Triton kernel for contiguous fp16 `[B,H,S,64]` inputs.
- [ ] Implement `_flash_attention_forward` as the shared core validation/launch helper copied by Chapter 14.
- [ ] Compare `output.float()` with `reference.float()` for all required causal and non-causal shapes.
- [ ] Add moderate sequence-length benchmarks and compile the script.

### Task 5: Implement Chapter 14 Mini Project

**Files:**
- Create: `chapter_14_mini_project/mini_project.py`

- [ ] Copy the final Chapter 13 `flash_attention_kernel` and `_flash_attention_forward` without changes.
- [ ] Implement tiny attention weights and complete PyTorch/Triton self-attention paths.
- [ ] Add correctness and synchronized end-to-end benchmarks.
- [ ] Compile the script and run AST drift checks.

### Task 6: Create Teaching Notebooks

**Files:**
- Create all Chapter 07-14 `.ipynb` files.

- [ ] Scaffold each notebook from the tutorial template.
- [ ] Add complete, sequential code and focused Markdown explanations matching each script.
- [ ] Explain the Chapter 12 `l == 0` guard and Chapter 14 kernel duplication.
- [ ] Compile every code cell.

### Task 7: Update README and Verify

**Files:**
- Modify: `README.md`

- [ ] Extend chapter table and learning path through Chapter 14.
- [ ] Mark Chapters 11-14 as advanced attention content and document all FlashAttention exclusions.
- [ ] Explain Chapter 13/14 duplication and add the four requested run commands.
- [ ] Run full `compileall`, static tests, notebook cell compilation, and final tree inspection.
- [ ] Report that GPU execution remains unverified on the current non-CUDA machine.
