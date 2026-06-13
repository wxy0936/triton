import time

import torch

try:
    import triton.testing
except ModuleNotFoundError:
    triton = None


def require_cuda_or_skip() -> bool:
    if not torch.cuda.is_available():
        print("CUDA is not available. Skipping Chapter 16 GPU benchmark and profiler demos.")
        return False
    return True


def wrong_timer_example(fn, repeat: int = 100) -> float:
    """Intentionally wrong: CUDA is asynchronous and this timer does not synchronize."""
    start = time.time()
    for _ in range(repeat):
        fn()
    elapsed_ms = (time.time() - start) * 1000 / repeat
    print(f"UNTRUSTWORTHY unsynchronized result: {elapsed_ms:.6f} ms")
    return elapsed_ms


def benchmark_sync(fn, warmup: int = 20, repeat: int = 100) -> float:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(repeat):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - start) * 1000 / repeat


def benchmark_cuda_event(fn, warmup: int = 20, repeat: int = 100) -> float:
    for _ in range(warmup):
        fn()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(repeat):
        fn()
    end.record()
    end.synchronize()
    return start.elapsed_time(end) / repeat


def benchmark_triton_do_bench(fn, warmup: int = 20, repeat: int = 100) -> float:
    if triton is None:
        raise RuntimeError("Triton is not installed")
    return float(triton.testing.do_bench(fn, warmup=warmup, rep=repeat, return_mode="median"))


def compare_benchmark_methods() -> None:
    a = torch.randn(1024, 1024, device="cuda", dtype=torch.float16)
    b = torch.randn_like(a)
    fn = lambda: torch.matmul(a, b)
    wrong_timer_example(fn)
    print(f"synchronize + perf_counter: {benchmark_sync(fn):.3f} ms")
    print(f"CUDA Event:                 {benchmark_cuda_event(fn):.3f} ms")
    if triton is None:
        print("Triton unavailable; skipping triton.testing.do_bench comparison.")
    else:
        print(f"triton.testing.do_bench:    {benchmark_triton_do_bench(fn):.3f} ms")


class TinyMLP(torch.nn.Module):
    def __init__(self, hidden_dim: int = 1024):
        super().__init__()
        self.linear_in = torch.nn.Linear(hidden_dim, hidden_dim * 2)
        self.activation = torch.nn.GELU()
        self.norm = torch.nn.LayerNorm(hidden_dim * 2)
        self.linear_out = torch.nn.Linear(hidden_dim * 2, hidden_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear_out(self.norm(self.activation(self.linear_in(x))))


def benchmark_forward_backward_train_step() -> None:
    dtype = torch.float16
    model = TinyMLP(1024).to(device="cuda", dtype=dtype)
    x = torch.randn(64, 1024, device="cuda", dtype=dtype)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

    def forward_only():
        with torch.no_grad():
            return model(x)

    def forward_backward():
        model.zero_grad(set_to_none=True)
        loss = model(x).float().square().mean()
        loss.backward()

    def train_step():
        optimizer.zero_grad(set_to_none=True)
        loss = model(x).float().square().mean()
        loss.backward()
        optimizer.step()

    print(f"forward-only:    {benchmark_cuda_event(forward_only, repeat=50):.3f} ms")
    print(f"forward+backward: {benchmark_cuda_event(forward_backward, repeat=30):.3f} ms")
    print(f"train step:      {benchmark_cuda_event(train_step, repeat=30):.3f} ms")
    optimizer.zero_grad(set_to_none=True)


def run_torch_profiler_demo() -> None:
    model = TinyMLP(1024).to(device="cuda", dtype=torch.float16)
    x = torch.randn(64, 1024, device="cuda", dtype=torch.float16)
    try:
        activities = [torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA]
        with torch.profiler.profile(
            activities=activities,
            record_shapes=True,
            profile_memory=True,
            with_stack=False,
        ) as profile:
            for _ in range(5):
                model(x)
        torch.cuda.synchronize()
        print(profile.key_averages().table(sort_by="self_cuda_time_total", row_limit=15))
    except Exception as error:
        print(f"torch.profiler CUDA demo is unavailable in this environment: {error}")
    print("Profiler results include measurement overhead; use clean benchmarks for final timing numbers.")


def main() -> None:
    if not require_cuda_or_skip():
        return
    compare_benchmark_methods()
    benchmark_forward_backward_train_step()
    print("\nProfiler demo (separate from formal benchmarks):")
    run_torch_profiler_demo()


if __name__ == "__main__":
    main()

