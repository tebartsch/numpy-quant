"""
Implement quantization routines.
"""
import numpy as np

from tinyquant import QTensor, Tensor


def quantize_tensor(tensor: Tensor, bit_width: int, asymmetric: bool):
    data = tensor.data

    min_val, max_val = data.min(), data.max()
    min_qval, max_qval = -2.0 ** (bit_width-1), 2.0 ** (bit_width-1) - 1.0

    if asymmetric:
        offset = 0.5 * (min_val + max_val)
        scale = (max_qval - min_qval) / (max_val - min_val)
    else:
        offset = 0.0
        scale = (max_qval - min_qval) / (2 * max(max_val, min_val))

    q_data_f32 = (data - offset) * scale - 0.5
    q_data = np.array(np.rint(q_data_f32), dtype=np.int32)

    if asymmetric:
        return QTensor(q_data, bit_width, scale=1/scale, zero_point=offset*scale - 0.5)
    else:
        return QTensor(q_data, bit_width, scale=1/scale)

