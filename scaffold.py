import torch


def ceil_div(a, b):
    return (a + b - 1) // b


def _pad_and_reshape(x, group_size):
    """Pad x and reshape into groups. Returns (x_view, x_padded, m, n).

    - x_view: reshaped tensor of shape (m_padded, 1, n_padded/group_size, group_size)
    - x_padded: zero-padded tensor of shape (m_padded, n_padded)
    - m, n: original dimensions of x
    """
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
    return x_view, x_padded, m, n


class _FakeInt4QuantizationSTE(torch.autograd.Function):
    """
    Straight-Through Estimator for INT4 quantization.

    Forward: Simulates INT4 quantization (quantize -> dequantize)
    Backward: Passes gradients unchanged (identity function)

    Your task: Implement the forward and backward methods below.
    """

    @staticmethod
    def forward(ctx, x, group_size):
        """
        Simulate INT4 quantization with group-wise symmetric quantization.

        Args:
            x: Input tensor of shape (m, n)
            group_size: Number of elements per quantization group (block_size_n).
                        block_size_m is always 1.

        Returns:
            Dequantized tensor of the same shape and dtype as x

        Hint: Use _pad_and_reshape(x, group_size) and compute_scales(x, group_size).

        Steps:
            1. Call _pad_and_reshape(x, group_size) to get (x_view, x_padded, m, n)
            2. Call compute_scales(x, group_size) to get per-group scales
            3. Quantize: divide x_view by scale, round to nearest int, clamp to [-7, 7]
            4. Dequantize: multiply by scale
            5. Reshape back to x_padded shape, slice to (m, n), return with original dtype
        """
        # TODO: Implement the forward pass
        raise NotImplementedError("Implement the forward pass")

    @staticmethod
    def backward(ctx, grad_output):
        """
        Straight-Through Estimator backward pass.

        For STE, gradients pass through unchanged.

        Returns:
            Tuple of gradients for each forward() input: (grad_x, grad_group_size)
            Since group_size is not a tensor, its gradient should be None.
        """
        # TODO: Implement the backward pass
        raise NotImplementedError("Implement the backward pass")


def compute_scales(x, group_size):
    """
    Compute per-group quantization scales.

    Args:
        x: Input tensor of shape (m, n)
        group_size: Number of elements per quantization group

    Returns:
        Scale tensor of shape (m, 1, n_padded/group_size, 1)
        where n_padded is n rounded up to the nearest multiple of group_size

    Hint: Call _pad_and_reshape(x, group_size) to get x_view, then compute
          scales from x_view using amax over dims (1, 3).
    """
    # TODO: Implement scale computation
    raise NotImplementedError("Implement scale computation")


def fake_int4_quantize(x, group_size):
    """Convenience wrapper for the quantization function."""
    return _FakeInt4QuantizationSTE.apply(x, group_size)


if __name__ == "__main__":
    # Fixed input tensor (same as in README and test)
    torch.manual_seed(42)
    x = torch.randn(2, 16, requires_grad=True)
    group_size = 4  # You can change this to experiment

    print("Input tensor:")
    print(x.data)
    print(f"Shape: {x.shape}")
    print(f"Group size: {group_size}")
    print()

    # Test forward pass
    try:
        out = fake_int4_quantize(x, group_size)
        print("Forward pass output:")
        print(out)
        print()
    except NotImplementedError as e:
        print(f"Forward pass: {e}")
        print()

    # Test backward pass
    try:
        if x.grad is not None:
            x.grad.zero_()
        out = fake_int4_quantize(x, group_size)
        loss = out.sum()
        loss.backward()
        print("Backward pass gradients:")
        print(x.grad)
        print()
    except NotImplementedError as e:
        print(f"Backward pass: {e}")
        print()

    # Test scale computation
    try:
        scales = compute_scales(x.detach(), group_size)
        print("Quantization scales:")
        print(scales)
    except NotImplementedError as e:
        print(f"Scale computation: {e}")
