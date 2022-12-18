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

    return np.array(scale, dtype=np.float32), zero_point and np.array(zero_point, dtype=np.int64)


def quantize(data: np.ndarray, bit_width: int, scale: np.float64, zero_point: np.int64 | None):
    if zero_point is not None:
        q_data_float = zero_point + data / scale
    else:
        q_data_float = data / scale

    min_qval, max_qval = -2.0 ** (bit_width - 1), 2.0 ** (bit_width - 1) - 1.0
    q_data_clipped = np.clip(q_data_float, min_qval, max_qval)
    q_data = np.array(np.rint(q_data_clipped), dtype=np.int64)

    return q_data
