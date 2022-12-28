import numpy as np

arr = np.array([[ 4.4037123 ], [-2.9683902 ], [-4.4077654 ], [ 2.3313837 ], [ 0.05330967]], dtype=np.float32)

def quantize(data: np.ndarray, scale: float, zero_point: int, bit_width: int):
    q_data = zero_point + data / scale

    min_qval, max_qval = -2.0 ** (bit_width - 1), 2.0 ** (bit_width - 1) - 1.0
    # Rescaled numbers which are smaller than the minimum quantized value `min_qval` are clipped to
    # this minimum value. Analogously for rescaled numbers greater than `max_qval`.
    q_data_clipped = np.clip(q_data, min_qval, max_qval)  
    q_data_boxed = np.array(np.rint(q_data_clipped), dtype=np.int64)

    return q_data_boxed

def dequantize(arr: np.ndarray, scale: float, zero_point: int | np.ndarray):
    return ((arr - zero_point) * scale).astype(np.float32)

# We choose scaling parameters which rescale the data approximately to an interval of -128 to 127
scale = np.array(0.04, np.float32)
zero_point = np.array(0, np.int64)

arr_quantized = quantize(arr, scale, zero_point, bit_width=8)
arr_round_trip = dequantize(arr_quantized, scale, zero_point)

with np.printoptions(precision=4, suppress=True):
    print("arr:\n", np.array2string(arr))
    print("arr_quantized:\n", np.array2string(arr_quantized))
    print("arr_round_trip:\n", np.array2string(arr_round_trip))
    print("round-trip error:\n", np.abs(arr - arr_round_trip))
