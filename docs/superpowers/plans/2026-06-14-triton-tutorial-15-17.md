# Triton Tutorial Chapters 15-17 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add independent performance-analysis, profiling, and attention-memory chapters aligned with CS336 and LLM systems coursework.

**Architecture:** Keep estimates and practical demos inside each chapter. Theoretical paths work without CUDA; hardware measurement paths skip clearly or exit normally. Static AST checks isolate the intentionally wrong timer example from all production benchmark helpers.

**Tech Stack:** Python 3.10+, PyTorch, Triton, Jupyter Notebook, torch.profiler, standard-library `unittest`

---

### Task 1: Extend Acceptance Tests

- [ ] Require Chapter 15-17 scripts/notebooks and specified symbols.
- [ ] Change the later-chapter guard to Chapter 18+.
- [ ] Permit `time.time()` only inside `wrong_timer_example`.
- [ ] Require correct synchronization primitives in formal timing helpers.
- [ ] Run tests and confirm failure because new files are absent.

### Task 2: Implement Chapter 15

- [ ] Implement estimate functions, formatting, table, synchronized benchmark, and CPU-only graceful path.
- [ ] Compile and execute the theoretical path locally.

### Task 3: Implement Chapter 16

- [ ] Implement wrong and correct timers, TinyMLP, forward/backward/train benchmarks, and profiler demo.
- [ ] Ensure no-CUDA execution exits successfully with a clear message.

### Task 4: Implement Chapter 17

- [ ] Implement attention and FlashAttention working-set estimates.
- [ ] Implement materialized causal attention and synchronized peak-memory measurement.
- [ ] Execute the theoretical table locally and skip CUDA clearly.

### Task 5: Create Notebooks and Update README

- [ ] Scaffold and populate three complete teaching notebooks.
- [ ] Extend README chapter table, learning route, CS336/LLM systems explanation, and run commands.

### Task 6: Final Verification

- [ ] Run full compileall and static tests.
- [ ] Compile every code cell in all 18 notebooks.
- [ ] Execute Chapter 15-17 CPU/no-CUDA paths.
- [ ] Inspect final tree and confirm no Chapter 18+ code.
