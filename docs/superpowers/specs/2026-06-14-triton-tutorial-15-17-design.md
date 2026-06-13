# Triton Tutorial Chapters 15-17 Design

## Goal

Extend the tutorial with three CS336 and LLM-systems-aligned chapters focused on performance reasoning, trustworthy GPU measurement, profiling, and attention memory analysis rather than new complex Triton kernels.

## Boundaries

Chapters 15-17 are fully independent and do not import previous chapters. Existing Chapter 00-14 core code remains unchanged. Each new chapter has a real teaching notebook and an equivalent standalone script.

Chapter 15 and Chapter 17 always run their theoretical tables on CPU. If CUDA is unavailable, they print a clear skip message for hardware measurements and exit successfully. Chapter 16 requires CUDA for every practical demonstration; it prints a clear message and exits successfully when CUDA is unavailable.

## Chapter 15: Performance Mental Model

The chapter defines a small estimate record with FLOPs, bytes moved, and arithmetic intensity. Vector add counts one add per element and three tensor transfers. Matmul counts `2*M*N*K` FLOPs and one read of A/B plus one write of C as an idealized minimum. Softmax and LayerNorm use explicitly documented approximate operation and traffic counts.

The standalone script prints a dependency-free table for the requested default shapes. On CUDA, it benchmarks PyTorch vector add, fp16 matmul, softmax, and LayerNorm with `triton.testing.do_bench`, then reports estimated achieved GB/s and TFLOP/s. These achieved values are labeled approximate because cache, hidden intermediates, implementation details, launch overhead, and simplified FLOP counts are not modeled.

## Chapter 16: Profiling and Benchmarking

The chapter includes one intentionally incorrect timing function that uses `time.time()` without synchronization. Its name, output, and documentation explicitly say the result is untrustworthy. Static tests permit unsynchronized `time.time()` only inside this function.

Three correct timing helpers use synchronized `perf_counter`, CUDA events, and `triton.testing.do_bench`. A TinyMLP contains Linear, GELU, LayerNorm, and Linear layers. Separate closures measure forward-only, forward+backward, and complete training steps. Gradients are cleared between repetitions to prevent state contamination, and fp16 losses are reduced in fp32.

The profiler demo uses CPU and CUDA activities, shape recording, memory profiling, and no stack collection. It catches profiler failures and prints a clear message. Profiler output is separated from formal benchmark output because profiling overhead changes timings.

## Chapter 17: Attention Memory Analysis

The chapter estimates Q/K/V, scores, probabilities, output, total important activations, and score/probability-to-QKV ratios. A table extends to sequence length 4096 while the optional CUDA peak-memory demo stops at 1024.

The FlashAttention working-set estimate includes Q, K, V, score tile, fp32 accumulator, and fp32 m/l state. It is labeled a teaching approximation rather than exact register/shared-memory usage.

The CUDA demo uses an explicit materialized attention implementation so scores and probabilities exist as tensors. It resets peak memory statistics, executes and synchronizes the function, then reports peak allocated memory. Causal masking uses absolute token indices. The chapter does not import Chapter 13 and prints that optional Triton comparison is not enabled.

## Verification

Static tests require all Chapter 15-17 files, specified APIs, main guards, valid notebooks, and no Chapter 18+ directories. AST inspection permits `time.time()` only inside `wrong_timer_example`; all other timing functions must include synchronization, CUDA events, or `triton.testing.do_bench` as appropriate.

All Python files and notebook code cells are compiled. CPU-only execution verifies Chapter 15 and Chapter 17 theoretical paths and Chapter 16's graceful CUDA skip. GPU benchmark, profiler, and memory behavior remains a server-side validation step.

