import math

import torch


def format_bytes(num_bytes: float) -> str:
    units = ("B", "KiB", "MiB", "GiB", "TiB")
    value = float(num_bytes)
    for unit in units:
        if abs(value) < 1024 or unit == units[-1]:
            return f"{value:.2f} {unit}"
        value /= 1024
    raise AssertionError("unreachable")


def estimate_attention_tensors(B: int, H: int, S: int, D: int, dtype_bytes: int = 2) -> dict[str, float]:
    qkv = 3 * B * H * S * D * dtype_bytes
    scores = B * H * S * S * dtype_bytes
    probabilities = scores
    output = B * H * S * D * dtype_bytes
    total = qkv + scores + probabilities + output
    return {
        "qkv_memory": qkv,
        "scores_memory": scores,
        "probs_memory": probabilities,
        "output_memory": output,
        "total_important_activation_memory": total,
        "scores_probs_to_qkv_ratio": (scores + probabilities) / qkv,
    }


def estimate_flashattention_working_set(BLOCK_M: int, BLOCK_N: int, D: int, dtype_bytes: int = 2, acc_bytes: int = 4) -> dict[str, float]:
    q_block = BLOCK_M * D * dtype_bytes
    k_block = BLOCK_N * D * dtype_bytes
    v_block = BLOCK_N * D * dtype_bytes
    score_block = BLOCK_M * BLOCK_N * acc_bytes
    accumulator = BLOCK_M * D * acc_bytes
    ml_state = 2 * BLOCK_M * acc_bytes
    return {
        "q_block": q_block,
        "k_block": k_block,
        "v_block": v_block,
        "score_block": score_block,
        "accumulator": accumulator,
        "m_l_state": ml_state,
        "total_working_set": q_block + k_block + v_block + score_block + accumulator + ml_state,
    }


def print_attention_memory_table() -> None:
    B, H, D = 1, 16, 64
    print(f"Theory table: B={B}, H={H}, D={D}, fp16")
    print(f"{'S':>6} {'QKV':>12} {'scores':>12} {'probs':>12} {'total':>12} {'S/P : QKV':>12}")
    print("-" * 72)
    for S in (128, 256, 512, 1024, 2048, 4096):
        item = estimate_attention_tensors(B, H, S, D)
        print(f"{S:6d} {format_bytes(item['qkv_memory']):>12} {format_bytes(item['scores_memory']):>12} {format_bytes(item['probs_memory']):>12} {format_bytes(item['total_important_activation_memory']):>12} {item['scores_probs_to_qkv_ratio']:12.2f}")

    working_set = estimate_flashattention_working_set(32, 32, D)
    print("\nTeaching FlashAttention tile working-set estimate (BLOCK_M=32, BLOCK_N=32):")
    for name, value in working_set.items():
        print(f"  {name:<20} {format_bytes(value)}")
    print("This is an educational estimate, not exact register or shared-memory usage.")


def naive_attention_materialized(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, causal: bool = False) -> torch.Tensor:
    if q.ndim != 4 or q.shape != k.shape or q.shape != v.shape:
        raise ValueError("q, k, and v must share shape [B, H, S, D]")
    scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(q.shape[-1])
    if causal:
        positions = torch.arange(q.shape[-2], device=q.device)
        scores = scores.masked_fill(positions[None, :] > positions[:, None], -float("inf"))
    probabilities = torch.softmax(scores, dim=-1)
    return torch.matmul(probabilities, v)


def measure_peak_memory(fn, *args, **kwargs) -> tuple[torch.Tensor, int]:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for peak-memory measurement")
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    baseline = torch.cuda.memory_allocated()
    output = fn(*args, **kwargs)
    torch.cuda.synchronize()
    peak = torch.cuda.max_memory_allocated()
    return output, peak - baseline


def run_peak_memory_demo() -> None:
    if not torch.cuda.is_available():
        print("Skipping actual peak-memory demo because CUDA is not available.")
        print("Skipping Triton FlashAttention comparison because chapter_13 implementation is unavailable.")
        return
    print("\nMaterialized attention peak-memory demo (allocator/cache effects may differ from theory):")
    for S in (128, 256, 512, 1024):
        q = torch.randn(1, 8, S, 64, device="cuda", dtype=torch.float16)
        k = torch.randn_like(q)
        v = torch.randn_like(q)
        output, peak = measure_peak_memory(naive_attention_materialized, q, k, v)
        print(f"S={S:4d}: peak additional allocated memory = {format_bytes(peak)}")
        del output, q, k, v
    print("Skipping Triton FlashAttention comparison because chapter_13 implementation is unavailable.")


def main() -> None:
    print_attention_memory_table()
    print("\nScores and probabilities scale as S^2: doubling S makes each about four times larger.")
    print("FlashAttention keeps tiled scores and online-softmax m/l/acc state instead of writing full SxS intermediates to HBM.")
    print("Its compute complexity remains approximately O(S^2 * D); the main change is memory traffic and storage.")
    run_peak_memory_demo()


if __name__ == "__main__":
    main()
