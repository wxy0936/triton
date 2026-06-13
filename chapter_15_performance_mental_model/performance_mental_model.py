from dataclasses import dataclass

import torch
import torch.nn.functional as F

try:
    import triton.testing
except ModuleNotFoundError:
    triton = None


@dataclass(frozen=True)
class Estimate:
    estimated_flops: float
    estimated_bytes: float
    arithmetic_intensity: float


def get_dtype_bytes(dtype: torch.dtype) -> int:
    return torch.empty((), dtype=dtype).element_size()


def _estimate(flops: float, moved_bytes: float) -> Estimate:
    return Estimate(flops, moved_bytes, flops / moved_bytes if moved_bytes else 0.0)


def estimate_vector_add(n: int, dtype_bytes: int = 4) -> Estimate:
    return _estimate(n, 3 * n * dtype_bytes)


def estimate_matmul(m: int, n: int, k: int, dtype_bytes: int = 2) -> Estimate:
    flops = 2 * m * n * k
    ideal_bytes = (m * k + k * n + m * n) * dtype_bytes
    return _estimate(flops, ideal_bytes)


def estimate_softmax(m: int, n: int, dtype_bytes: int = 4) -> Estimate:
    # Approximation: max, subtract, exp, sum, and divide per element.
    return _estimate(5 * m * n, 2 * m * n * dtype_bytes)


def estimate_layernorm(m: int, n: int, dtype_bytes: int = 4) -> Estimate:
    # Approximation: mean, variance, normalize, scale, and bias.
    return _estimate(8 * m * n, (2 * m * n + 2 * n) * dtype_bytes)


def format_bytes(num_bytes: float) -> str:
    units = ("B", "KiB", "MiB", "GiB", "TiB")
    value = float(num_bytes)
    for unit in units:
        if abs(value) < 1024 or unit == units[-1]:
            return f"{value:.2f} {unit}"
        value /= 1024
    raise AssertionError("unreachable")


def format_flops(num_flops: float) -> str:
    units = ((1e12, "TFLOP"), (1e9, "GFLOP"), (1e6, "MFLOP"), (1e3, "KFLOP"))
    for scale, unit in units:
        if num_flops >= scale:
            return f"{num_flops / scale:.2f} {unit}"
    return f"{num_flops:.2f} FLOP"


def benchmark(fn, warmup: int = 20, rep: int = 100) -> float:
    if triton is None:
        raise RuntimeError("Triton is required for CUDA benchmarking")
    return float(triton.testing.do_bench(fn, warmup=warmup, rep=rep, return_mode="median"))


def run_estimation_table() -> None:
    rows = [
        ("vector add fp32", estimate_vector_add(1_000_000, 4)),
        ("matmul fp16", estimate_matmul(1024, 1024, 1024, 2)),
        ("softmax fp32", estimate_softmax(1024, 1024, 4)),
        ("layernorm fp32", estimate_layernorm(1024, 1024, 4)),
    ]
    print(f"{'operation':<20} {'FLOPs':>14} {'bytes moved':>16} {'FLOPs/byte':>12}")
    print("-" * 66)
    for name, estimate in rows:
        print(f"{name:<20} {format_flops(estimate.estimated_flops):>14} {format_bytes(estimate.estimated_bytes):>16} {estimate.arithmetic_intensity:12.2f}")
    print("\nLow intensity often suggests memory-bound behavior; high intensity makes compute limits more likely.")


def run_actual_benchmarks() -> None:
    if not torch.cuda.is_available():
        print("Skipping actual benchmarks because CUDA is not available.")
        return
    if triton is None:
        print("Skipping actual benchmarks because Triton is not installed.")
        return

    device = torch.device("cuda")
    x = torch.randn(1_000_000, device=device)
    y = torch.randn_like(x)
    a = torch.randn(1024, 1024, device=device, dtype=torch.float16)
    b = torch.randn_like(a)
    matrix = torch.randn(1024, 1024, device=device)
    weight = torch.ones(1024, device=device)
    bias = torch.zeros(1024, device=device)
    operations = [
        ("vector add", lambda: x + y, estimate_vector_add(x.numel(), get_dtype_bytes(x.dtype))),
        ("matmul", lambda: torch.matmul(a, b), estimate_matmul(1024, 1024, 1024, get_dtype_bytes(a.dtype))),
        ("softmax", lambda: torch.softmax(matrix, dim=1), estimate_softmax(1024, 1024, get_dtype_bytes(matrix.dtype))),
        ("layernorm", lambda: F.layer_norm(matrix, (1024,), weight, bias), estimate_layernorm(1024, 1024, get_dtype_bytes(matrix.dtype))),
    ]
    print(f"\n{'operation':<14} {'time ms':>10} {'approx GB/s':>14} {'approx TFLOP/s':>16}")
    print("-" * 58)
    for name, fn, estimate in operations:
        milliseconds = benchmark(fn)
        seconds = milliseconds / 1000
        gb_per_second = estimate.estimated_bytes / seconds / 1e9
        tflops = estimate.estimated_flops / seconds / 1e12
        print(f"{name:<14} {milliseconds:10.3f} {gb_per_second:14.2f} {tflops:16.3f}")
    print("These achieved numbers are rough estimates; cache, intermediates, implementation details, and launch overhead are simplified.")


def main() -> None:
    run_estimation_table()
    run_actual_benchmarks()


if __name__ == "__main__":
    main()

