# Triton Tutorial Chapters 07-14 Design

## Goal

Extend the existing beginner tutorial with chapters 07 through 14 while preserving the structure, teaching style, standalone scripts, synchronized benchmarks, and notebook/script equivalence established by chapters 00 through 06.

## Scope

The extension adds:

- Chapter 07: baseline blocked matrix multiplication
- Chapter 08: grouped-order matrix multiplication
- Chapter 09: Triton autotune and reliable benchmarking
- Chapter 10: row-wise LayerNorm forward
- Chapter 11: PyTorch scaled dot-product attention foundations
- Chapter 12: PyTorch online-softmax attention reference
- Chapter 13: teaching-oriented Triton FlashAttention v1 forward
- Chapter 14: tiny self-attention mini project using a copied Chapter 13 kernel

All implementations are forward-only. Backward, dropout, variable lengths, paged attention, KV cache, GQA, and MQA are explicitly out of scope.

## Project Boundaries

Each chapter contains a real, sequentially runnable notebook and an equivalent standalone Python script. A chapter may import only helpers from `common/`; it may not import another chapter.

Chapter 11 and Chapter 12 are deliberate exceptions to the usual Triton-kernel requirement. Chapter 11 establishes the attention equations and tensor shapes with PyTorch. Chapter 12 establishes the online-softmax recurrence with a clear PyTorch implementation. Both still contain a reference, correctness checks, and synchronized benchmarks.

## Shared FlashAttention Strategy

Chapter 13 and Chapter 14 each contain their own copy of the same teaching FlashAttention forward kernel and wrapper logic. Chapter 14's copy is taken from the final Chapter 13 implementation rather than designed independently.

The two kernels must remain behaviorally identical for shape validation, supported dtype and head dimension, causal masking, numerical stability, launch configuration, and output layout. Static acceptance tests compare normalized AST representations of the kernel functions so accidental drift is detected. Any bug fix affecting causal masking, numerical stability, dtype, shape handling, or indexing must be applied to both copies.

The README explains that this duplication exists for chapter-level teaching independence. A production project should extract the shared implementation into a common module.

## Chapter Designs

### Chapter 07: Matmul Baseline

The kernel maps a two-dimensional grid to output tiles. Each program owns one `[BLOCK_M, BLOCK_N]` tile, builds row and column offsets, and loops over K in `BLOCK_K` chunks. Loads are masked independently for M, N, and K boundaries. `tl.dot` accumulates into fp32 and the output is stored as fp16.

The wrapper accepts contiguous fp16 CUDA matrices `A[M, K]` and `B[K, N]`, validates dimensions, and supports non-divisible shapes. Correctness uses relaxed fp16 tolerances. The default benchmark uses `512 x 512 x 512`.

### Chapter 08: Matmul Optimized

The arithmetic remains the same as Chapter 07, but program IDs use grouped M ordering to encourage L2 reuse. The chapter introduces `GROUP_SIZE_M`, `num_warps`, and `num_stages` without autotune. The script contains a small baseline kernel locally so its benchmark can compare baseline Triton, grouped Triton, and PyTorch without importing Chapter 07.

### Chapter 09: Autotune and Benchmark

The chapter explains asynchronous CUDA timing, synchronization, warmup, repeat, `triton.testing.do_bench`, and `@triton.autotune`. Three moderate configurations cover different tile sizes and warp/stage counts. The autotune key is `M`, `N`, and `K`. A result table compares PyTorch and autotuned Triton over several moderate shapes.

### Chapter 10: LayerNorm

One program handles one row of a contiguous `[M, N]` input. The kernel loads a power-of-two block, computes mean and variance in fp32, applies reciprocal square root with epsilon, then applies one-dimensional affine weight and bias. Invalid columns are masked.

The wrapper accepts fp16 or fp32 input with matching weight and bias, chooses `next_power_of_2(N)`, and rejects rows whose padded width exceeds 65,536. Correctness compares against `torch.nn.functional.layer_norm` with dtype-appropriate tolerance.

### Chapter 11: Attention Foundations

The chapter defines `q`, `k`, and `v` as `[B, H, S, D]`, computes scaled scores, optionally applies a causal upper-triangular mask, applies softmax, and multiplies probabilities by V. The implementation does not call Triton or external FlashAttention.

When PyTorch exposes `scaled_dot_product_attention`, the chapter compares values against it. Benchmarking measures the explicit PyTorch reference with the shared synchronized benchmark helper.

### Chapter 12: Online Softmax

The PyTorch implementation iterates over query blocks and key/value blocks. For each row it tracks running maximum `m`, running denominator `l`, and unnormalized output accumulator `acc`. Causal masking uses absolute query and key token indices; completely masked positions contribute zero after exponentiation.

The final output is `acc / l`. Correctness compares causal and non-causal results against the explicit PyTorch attention reference. The benchmark is educational and is expected to show Python-loop overhead; the notebook also compares full score-matrix element count with one block's score count.

### Chapter 13: FlashAttention v1 Forward

The initial supported contract is contiguous fp16 CUDA tensors with identical shape `[B, H, S, D]` and `D == 64`. Fixing D to 64 keeps the teaching kernel stable and avoids presenting architecture-dependent head-dimension tuning as a beginner feature.

The grid is `(ceil_div(S, BLOCK_M), B * H)`. Each program loads one query block and iterates over K/V blocks. It computes fp32 scaled scores with `tl.dot`, applies sequence-boundary and optional causal masks, and updates fp32 `m_i`, `l_i`, and `acc` using the numerically stable online-softmax recurrence. It stores fp16 output for valid query rows.

Correctness covers all four required shapes/modes with `rtol=1e-2` and `atol=1e-2`. Benchmarks cover sequence lengths 128, 256, and 512 for `B=2`, `H=4`, and `D=64`, comparing the explicit PyTorch reference and the teaching Triton implementation.

### Chapter 14: Tiny Self-Attention

Inputs have shape `[B, S, C]` with `C = H * D`. A small weights container holds `Wq`, `Wk`, `Wv`, and `Wo`, each `[C, C]`. Both paths compute projections with PyTorch matrix multiplication, reshape to contiguous `[B, H, S, D]`, run either explicit PyTorch attention or the copied Triton FlashAttention kernel, reshape to `[B, S, C]`, and apply the output projection.

The default shape is `B=2`, `S=128`, `C=256`, `H=4`, `D=64`. Since projections are shared PyTorch work, the benchmark measures complete tiny self-attention paths and explains that it is not an isolated kernel benchmark.

## Error Handling

Wrappers fail before launch with clear messages for CPU tensors, unsupported dimensionality, mismatched shapes or dtypes, non-contiguous inputs, invalid matrix dimensions, empty reduction dimensions, unsupported LayerNorm width, and unsupported FlashAttention head dimensions.

## Verification

Static tests will require all chapter 00-14 files, parse every Python file and notebook, check main guards and required symbols, reject `time.time()` CUDA timing, and ensure Chapter 13/14 do not import one another or `common.attention`.

The tests also compare the AST of `flash_attention_kernel` and the core FlashAttention wrapper validation/launch logic between chapters 13 and 14. Notebook code cells will be compiled independently.

On a CUDA server, every standalone script should be executed. The FlashAttention checks are the highest-risk runtime gate because Triton compilation and GPU behavior cannot be fully validated on the current non-CUDA machine. The project will state this limitation explicitly rather than claiming GPU execution success.

## README Changes

The chapter table and learning path extend through Chapter 14. Chapters 11-14 are marked as the attention/FlashAttention advanced section. The README lists the exact FlashAttention exclusions and explains the deliberate Chapter 13/14 kernel duplication. It includes direct execution examples for Chapters 07, 10, 13, and 14.
