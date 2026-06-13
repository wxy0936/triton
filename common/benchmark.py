from collections.abc import Callable

import triton.testing


def bench(fn: Callable[[], object], warmup: int = 25, rep: int = 100) -> float:
    """Return median CUDA execution time in milliseconds."""
    if warmup < 0 or rep <= 0:
        raise ValueError("warmup must be non-negative and rep must be positive")
    return float(
        triton.testing.do_bench(
            fn,
            warmup=warmup,
            rep=rep,
            return_mode="median",
        )
    )
