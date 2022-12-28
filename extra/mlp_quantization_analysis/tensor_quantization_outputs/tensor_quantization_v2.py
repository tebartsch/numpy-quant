import numpy as np

arr = np.array([[ 4.4037123 ], [-2.9683902 ], [-4.4077654 ], [ 2.3313837 ], [ 0.05330967]], dtype=np.float32)

def quantize(data: np.ndarray, scale: np.float64, zero_point: np.int64, bit_width: int):
    q_data = zero_point + data / scale

    min_qval, max_qval = -2.0 ** (bit_width - 1), 2.0 ** (bit_width - 1) - 1.0
    # Rescaled numbers which are smaller than the minimum quantized value `min_qval` are clipped to
    # this minimum value. Analogously for rescaled numbers greater than `max_qval`.
    q_data_clipped = np.clip(q_data, min_qval, max_qval)  
    q_data_boxed = np.array(np.rint(q_data_clipped), dtype=np.int64)

    return q_data_boxed

def dequantize(arr: np.ndarray, scale: np.float64, zero_point: np.int64):
    return ((arr - zero_point) * scale).astype(np.float32)

def quant_parameters(min_val: np.float32, max_val: np.float32, bit_width: int, asymmetric: bool):
    min_qval, max_qval = -2.0 ** (bit_width - 1), 2.0 ** (bit_width - 1) - 1.0

    if asymmetric:
        scale = (max_val - min_val) / (max_qval - min_qval)
        zero_point0 = min_qval - min_val / scale
        zero_point = np.rint(zero_point0).astype(np.int64)
    else:
        scale = (2 * max(max_val, min_val)) / (max_qval - min_qval)
        zero_point = np.array(0, np.int64)

    scale = np.array(scale, dtype=np.float32)
    zero_point = zero_point and np.array(zero_point, dtype=np.int64)

    return scale, zero_point

scale, zero_point = quant_parameters(arr.min(), arr.max(), bit_width=8, asymmetric=True)

arr_quantized = quantize(arr, scale, zero_point, bit_width=8)
arr_round_trip = dequantize(arr_quantized, scale, zero_point)

with np.printoptions(precision=4, suppress=True):
    print("arr:\n", np.array2string(arr))
    print("arr_quantized:\n", np.array2string(arr_quantized))
    print("arr_round_trip:\n", np.array2string(arr_round_trip))
