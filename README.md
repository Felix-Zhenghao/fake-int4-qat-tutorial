# Fake INT4 Quantization-Aware Training (QAT) Tutorial

## Overview

In Quantization-Aware Training (QAT), we simulate the effects of low-precision quantization during the forward pass while keeping full-precision weights for gradient updates. This is done using a **Fake Quantization** operation: quantize then immediately dequantize, so the model "feels" quantization noise during training.

For INT4, each weight is mapped to one of 15 discrete levels in the range [-7, 7].

The key challenge: quantization (rounding) is non-differentiable. We solve this with the **Straight-Through Estimator (STE)**, which passes gradients through the rounding operation unchanged.

## Algorithm: Group-wise Symmetric INT4 Fake Quantization

Given:
- Input tensor `x` of shape `(m, n)` — here `(2, 16)`
- `block_size_m = 1`, `block_size_n = group_size = 4`
- `q_max = 7` (INT4 signed symmetric: values in [-7, 7])

### Input

```
torch.manual_seed(42)
x = torch.randn(2, 16)
```

```
Row 0: [ 1.9269,  1.4873,  0.9007, -2.1055,  0.6784, -1.2345, -0.0431, -1.6047,
        -0.7521,  1.6487, -0.3925, -1.4036, -0.7279, -0.5594, -0.7688,  0.7624]
Row 1: [ 1.6423, -0.1596, -0.4974,  0.4396, -0.7581,  1.0783,  0.8008,  1.6806,
         1.2791,  1.2964,  0.6105,  1.3347, -0.2316,  0.0418, -0.2516,  0.8599]
```

---

### Step 1: Pad to Align with Group Boundaries

Pad `m` to a multiple of `block_size_m` (1) and `n` to a multiple of `block_size_n` (4).

Here `m=2` is already a multiple of 1 and `n=16` is already a multiple of 4, so **no padding is needed**.

If `n` were 15, we would pad to 16 by appending a column of zeros.

---

### Step 2: Reshape into Groups

Reshape `x` from `(2, 16)` into `(m, block_size_m, n/group_size, group_size)` = `(2, 1, 4, 4)`:

```
Group [0][0]: [ 1.9269,  1.4873,  0.9007, -2.1055]
Group [0][1]: [ 0.6784, -1.2345, -0.0431, -1.6047]
Group [0][2]: [-0.7521,  1.6487, -0.3925, -1.4036]
Group [0][3]: [-0.7279, -0.5594, -0.7688,  0.7624]

Group [1][0]: [ 1.6423, -0.1596, -0.4974,  0.4396]
Group [1][1]: [-0.7581,  1.0783,  0.8008,  1.6806]
Group [1][2]: [ 1.2791,  1.2964,  0.6105,  1.3347]
Group [1][3]: [-0.2316,  0.0418, -0.2516,  0.8599]
```

Each group of 4 elements will share a single quantization scale.

---

### Step 3: Compute Per-Group Scale

For symmetric quantization, the scale for each group is:

```
scale = max(|group|) / q_max
scale = clamp(scale, min=1e-5)
```

The `amax` is taken over dimensions `(1, 3)` — i.e., over `block_size_m` and `block_size_n` — to get one scale per group.

```
Scale [0][0]: 0.300789    (max_abs = 2.1055, scale = 2.1055 / 7)
Scale [0][1]: 0.229238    (max_abs = 1.6047, scale = 1.6047 / 7)
Scale [0][2]: 0.235532    (max_abs = 1.6487, scale = 1.6487 / 7)
Scale [0][3]: 0.109834    (max_abs = 0.7688, scale = 0.7688 / 7)

Scale [1][0]: 0.234617    (max_abs = 1.6423, scale = 1.6423 / 7)
Scale [1][1]: 0.240089    (max_abs = 1.6806, scale = 1.6806 / 7)
Scale [1][2]: 0.190677    (max_abs = 1.3347, scale = 1.3347 / 7)
Scale [1][3]: 0.122837    (max_abs = 0.8599, scale = 0.8599 / 7)
```

---

### Step 4: Quantize

Divide each element by its group's scale, round to the nearest integer, and clamp to [-7, 7]:

```
x_int = clamp(round(x / scale), -7, 7)
```

**Detailed example for Group [0][0]** (scale = 0.300789):

| Element | Value   | / scale  | round | clamp  |
|---------|---------|----------|-------|--------|
| [0]     |  1.9269 |  6.4062  |   6   |   6    |
| [1]     |  1.4873 |  4.9446  |   5   |   5    |
| [2]     |  0.9007 |  2.9945  |   3   |   3    |
| [3]     | -2.1055 | -7.0000  |  -7   |  -7    |

---

### Step 5: Dequantize

Multiply the quantized integers back by the scale to get the fake-quantized values:

```
x_dequant = x_int * scale
```

**Continuing Group [0][0]** (scale = 0.300789):

| Element | x_int | * scale | Original | Error    |
|---------|-------|---------|----------|----------|
| [0]     |   6   |  1.8047 |  1.9269  | 0.1222   |
| [1]     |   5   |  1.5039 |  1.4873  | 0.0167   |
| [2]     |   3   |  0.9024 |  0.9007  | 0.0016   |
| [3]     |  -7   | -2.1055 | -2.1055  | 0.0000   |

The element with the largest magnitude (-2.1055) is quantized exactly, while others incur small quantization errors.

---

### Step 6: Remove Padding and Return

Reshape the dequantized view back to the padded shape, then slice to the original `(m, n)` dimensions. Convert back to the input dtype.

---

### Backward Pass: Straight-Through Estimator (STE)

The rounding in Step 4 has zero gradient almost everywhere (and undefined gradient at half-integers). The STE simply passes the gradient through unchanged:

```
grad_input = grad_output    (identity)
grad_group_size = None      (not a tensor parameter)
```

This allows the model to learn despite the non-differentiable quantization step.

---

## Files

| File | Description |
|------|-------------|
| `scaffold.py` | Skeleton code with TODOs for you to implement |
| `gt_implementation.py` | Ground truth reference implementation |
| `test_implementation.py` | Tests to verify your implementation against the GT |

## Usage

```bash
# Activate the environment
conda activate int4-qat-tutorial

# Run your implementation (shows NotImplementedError until you fill in the TODOs)
python scaffold.py

# After implementing, run tests to check correctness
python test_implementation.py
```
