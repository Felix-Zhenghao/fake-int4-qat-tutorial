import sys
import torch

from scaffold import (
    fake_int4_quantize as student_quantize,
    compute_scales as student_scales,
)
from gt_implementation import (
    fake_int4_quantize as gt_quantize,
    compute_scales as gt_scales,
)


def test_scales():
    """Test that computed scales match the GT."""
    torch.manual_seed(42)
    x = torch.randn(2, 16)
    group_size = 4

    s = student_scales(x.clone(), group_size)
    g = gt_scales(x.clone(), group_size)

    if torch.allclose(s, g, atol=1e-6):
        print("[PASS] Quantization scales match GT")
        return True
    else:
        print("[FAIL] Quantization scales do not match GT")
        print(f"  Student: {s.squeeze()}")
        print(f"  GT:      {g.squeeze()}")
        return False


def test_forward():
    """Test that the forward pass output matches the GT."""
    torch.manual_seed(42)
    x = torch.randn(2, 16)
    group_size = 4

    student_out = student_quantize(x.clone(), group_size)
    gt_out = gt_quantize(x.clone(), group_size)

    if torch.allclose(student_out, gt_out, atol=1e-6):
        print("[PASS] Forward pass matches GT")
        return True
    else:
        print("[FAIL] Forward pass does not match GT")
        print(f"  Max diff: {(student_out - gt_out).abs().max().item():.8f}")
        print(f"  Student: {student_out}")
        print(f"  GT:      {gt_out}")
        return False


def test_backward():
    """Test that gradients pass through unchanged (STE)."""
    torch.manual_seed(42)
    x_student = torch.randn(2, 16, requires_grad=True)
    x_gt = x_student.detach().clone().requires_grad_(True)
    group_size = 4

    # Student backward
    out_s = student_quantize(x_student, group_size)
    out_s.sum().backward()

    # GT backward
    out_g = gt_quantize(x_gt, group_size)
    out_g.sum().backward()

    if torch.allclose(x_student.grad, x_gt.grad, atol=1e-6):
        print("[PASS] Backward pass matches GT")
        return True
    else:
        print("[FAIL] Backward pass does not match GT")
        print(f"  Student grad: {x_student.grad}")
        print(f"  GT grad:      {x_gt.grad}")
        return False


if __name__ == "__main__":
    print("=" * 50)
    print("INT4 QAT Implementation Tests")
    print("=" * 50)
    print()

    results = []
    results.append(test_scales())
    results.append(test_forward())
    results.append(test_backward())

    print()
    if all(results):
        print("All tests passed!")
    else:
        print("Some tests failed. Check output above.")
        sys.exit(1)
