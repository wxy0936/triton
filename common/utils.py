import torch

try:
    import triton
except ModuleNotFoundError:
    triton = None


def get_device() -> torch.device:
    """Require a CUDA-capable PyTorch runtime."""
    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA is not available. This tutorial requires an NVIDIA GPU and "
            "a CUDA-enabled PyTorch installation."
        )
    return torch.device("cuda")


def print_gpu_info() -> None:
    """Print the software versions and active CUDA device."""
    device = get_device()
    if triton is None:
        raise RuntimeError(
            "Triton is not installed. Install the project dependencies with "
            "'pip install -r requirements.txt'."
        )
    index = torch.cuda.current_device()
    print(f"PyTorch version: {torch.__version__}")
    print(f"Triton version: {triton.__version__}")
    print(f"CUDA runtime: {torch.version.cuda}")
    print(f"CUDA device: {device}")
    print(f"GPU: {torch.cuda.get_device_name(index)}")


def set_seed(seed: int = 0) -> None:
    """Seed CPU and CUDA random number generators."""
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
