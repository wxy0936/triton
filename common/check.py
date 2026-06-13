import torch


def assert_close(
    name: str,
    actual: torch.Tensor,
    expected: torch.Tensor,
    rtol: float = 1e-4,
    atol: float = 1e-4,
) -> None:
    """Print the maximum error and raise if two tensors are not close."""
    if actual.shape != expected.shape:
        raise AssertionError(
            f"{name}: shape mismatch: actual={tuple(actual.shape)}, "
            f"expected={tuple(expected.shape)}"
        )
    if actual.numel() == 0:
        max_error = 0.0
    else:
        max_error = (actual - expected).abs().max().item()
    print(f"{name}: max error = {max_error:.6e}")
    if not torch.allclose(actual, expected, rtol=rtol, atol=atol):
        raise AssertionError(
            f"{name}: tensors differ (rtol={rtol}, atol={atol}, "
            f"max error={max_error:.6e})"
        )
    print(f"{name}: correctness check passed")

