"""
Implement quantization routines.
"""
import numpy as np

from tinyquant import QTensor, Tensor


def quantize_tensor(tensor: Tensor, bit_width: int, asymmetric: bool):
    data = tensor.data

    min_val = np.minimum(data.min(), 0.0)
    max_val = np.maximum(data.max(), 0.0)
    min_qval, max_qval = -2.0 ** (bit_width-1), 2.0 ** (bit_width-1) - 1.0

    if asymmetric:
        scale = (max_val - min_val) / (max_qval - min_qval)
        zero_point0 = min_qval - min_val / scale
        zero_point = np.rint(zero_point0).astype(np.int64)
        q_data_float = zero_point + data / scale
    else:
        scale = (2 * max(max_val, min_val)) / (max_qval - min_qval)
        zero_point = None
        q_data_float = data / scale

    q_data_clipped = np.clip(q_data_float, min_qval, max_qval)
    q_data = np.array(np.rint(q_data_clipped), dtype=np.int64)

    return QTensor(q_data, bit_width, scale=scale, zero_point=zero_point)

