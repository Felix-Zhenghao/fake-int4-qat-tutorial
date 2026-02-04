import torch


def ceil_div(a, b):
    return (a + b - 1) // b


class _FakeInt4QuantizationSTE(torch.autograd.Function):
    """
    Straight-Through Estimator for INT4 quantization.

    Forward: Simulates INT4 quantization (quantize -> dequantize)
    Backward: Passes gradients unchanged (identity function)
    """

    @staticmethod
    def forward(ctx, x, group_size):
        m, n = x.shape
        block_size_m, block_size_n = 1, group_size

        # Step 1: Pad to align with group boundaries
        m_padded = ceil_div(m, block_size_m) * block_size_m
        n_padded = ceil_div(n, block_size_n) * block_size_n

        x_padded = torch.zeros(
            (m_padded, n_padded),
            dtype=x.dtype, device=x.device
        )
        x_padded[:m, :n] = x

        # Step 2: Reshape for group-wise operations
        x_view = x_padded.view(
            m_padded // block_size_m,
            block_size_m,
            n_padded // block_size_n,
            block_size_n
        )

        # Step 3: Compute per-group scale (symmetric quantization)
        x_max = x_view.abs().float().amax(dim=(1, 3), keepdim=True)
        q_max = 7  # INT4 symmetric range: [-7, 7]
        x_scale = x_max / q_max
        x_scale = x_scale.clamp(min=1e-5)

        # Step 4: Quantize (round to nearest integer)
        x_div = x_view / x_scale
        x_round = torch.round(x_div)
        x_q_clamped = x_round.clamp(-q_max, q_max)

        # Step 5: Dequantize (restore to original scale)
        x_dequant_view = x_q_clamped * x_scale

        # Step 6: Remove padding and return
        x_dequant_full = x_dequant_view.view_as(x_padded)
        x_out = x_dequant_full[:m, :n].contiguous().to(x.dtype)

        return x_out

    @staticmethod
    def backward(ctx, grad_output):
        # STE: Gradients pass through unchanged
        return grad_output, None


def compute_scales(x, group_size):
    """Compute per-group quantization scales."""
    m, n = x.shape
    block_size_m, block_size_n = 1, group_size

    m_padded = ceil_div(m, block_size_m) * block_size_m
    n_padded = ceil_div(n, block_size_n) * block_size_n

    x_padded = torch.zeros(
        (m_padded, n_padded),
        dtype=x.dtype, device=x.device
    )
    x_padded[:m, :n] = x

    x_view = x_padded.view(
        m_padded // block_size_m,
        block_size_m,
        n_padded // block_size_n,
        block_size_n
    )

    x_max = x_view.abs().float().amax(dim=(1, 3), keepdim=True)
    q_max = 7
    x_scale = x_max / q_max
    x_scale = x_scale.clamp(min=1e-5)

    return x_scale


def fake_int4_quantize(x, group_size):
    """Convenience wrapper."""
    return _FakeInt4QuantizationSTE.apply(x, group_size)
