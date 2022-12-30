"""
Implement quantization routines.
"""
import numpy as np


def quant_parameters(min_val: np.float32, max_val: np.float32, bit_width: int, asymmetric: bool):
    min_qval, max_qval = -2.0 ** (bit_width - 1), 2.0 ** (bit_width - 1) - 1.0

    if asymmetric:
        scale = (max_val - min_val) / (max_qval - min_qval)
        zero_point0 = min_qval - min_val / scale
        zero_point = np.rint(zero_point0).astype(np.int64)
    else:
        scale = (2 * max(max_val, min_val)) / (max_qval - min_qval)
        zero_point = None

    scale = np.array(scale, dtype=np.float32)
    zero_point = zero_point and np.array(zero_point, dtype=np.int64)

    return scale, zero_point


def quantize(data: np.ndarray, bit_width: int, scale: np.float64, zero_point: np.int64 | None):
    if zero_point is not None:
        q_data_float = zero_point + data / scale
    else:
        q_data_float = data / scale

    min_qval, max_qval = -2.0 ** (bit_width - 1), 2.0 ** (bit_width - 1) - 1.0
    q_data_clipped = np.clip(q_data_float, min_qval, max_qval)
    q_data = np.array(np.rint(q_data_clipped), dtype=np.int64)

    return q_data


def dequantize(arr: np.ndarray, scale: float, zero_point: np.int64 | np.ndarray):
    if zero_point is not None:
        return ((arr - zero_point) * scale).astype(np.float32)
    else:
        return (arr * scale).astype(np.float32)


def q_matmul(arr_a: np.ndarray, scale_a: float, zero_point_a: int,
             arr_b: np.ndarray, scale_b: float, zero_point_b: int):
    s1 = arr_a.shape
    matmul = np.matmul(arr_a.astype(np.int64), arr_b)
    scale = scale_a * scale_b
    if zero_point_a is None and zero_point_b is None:
        return matmul, scale, None
    elif zero_point_a is None:
        zero_point = arr_a.sum(axis=-1, keepdims=True) * zero_point_b
        return matmul, scale, zero_point
    elif zero_point_b is None:
        zero_point = arr_b.sum(axis=-2, keepdims=True) * zero_point_a
        return matmul, scale, zero_point
    else:
        zero_point = (arr_a.sum(axis=-1, keepdims=True) * zero_point_b
                      + arr_b.sum(axis=-2, keepdims=True) * zero_point_a
                      - zero_point_a * zero_point_b * s1[-1])
        return matmul, scale, zero_point


def requantize(arr: np.ndarray, arr_scale: float, arr_zero_points: np.ndarray,
               res_scale: float, res_zero_point: int, bit_width: int):
    min_qval, max_qval = -2.0 ** (bit_width - 1), 2.0 ** (bit_width - 1) - 1.0
    dequant = dequantize(arr, arr_scale, arr_zero_points)
    if res_zero_point is None:
        qdata = np.clip(np.rint(1 / res_scale * dequant), min_qval, max_qval).astype(np.int64)
    else:
        qdata = np.clip(np.rint(res_zero_point + 1 / res_scale * dequant), min_qval, max_qval).astype(np.int64)
    return qdata
