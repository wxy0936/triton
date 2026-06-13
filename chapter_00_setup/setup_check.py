import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch

from common.utils import get_device, print_gpu_info, set_seed


def main() -> None:
    device = get_device()
    set_seed(0)
    print_gpu_info()

    x = torch.tensor([1.0, 2.0, 3.0], device=device)
    y = torch.tensor([4.0, 5.0, 6.0], device=device)
    result = x + y
    expected = torch.tensor([5.0, 7.0, 9.0], device=device)
    if not torch.equal(result, expected):
        raise RuntimeError("The CUDA tensor test produced an unexpected result")

    print(f"CUDA tensor test: {result}")
    print("Triton tutorial environment is ready")


if __name__ == "__main__":
    try:
        main()
    except RuntimeError as error:
        raise SystemExit(f"Environment check failed: {error}") from None
