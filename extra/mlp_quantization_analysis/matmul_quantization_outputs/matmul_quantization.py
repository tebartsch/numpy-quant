import numpy as np

x1 = np.array([[-0.68969274,  0.36898366], [ 0.48721004,  0.59565425], [ 0.9734074 , -0.08323386]], dtype=np.float32)
x2 = np.array([[ 4.4037123 , -2.9683902 , -4.4077654 ,  2.3313837 ,  0.05330967], [-1.0420023 ,  3.5323772 , -1.5059234 ,  4.3279686 , -4.243471  ]], dtype=np.float32)

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

def q_matmul(arr_a: np.ndarray, scale_a: float, zero_point_a: int,
           arr_b: np.ndarray, scale_b: float, zero_point_b: int):
    q_matmul_result = np.matmul(arr_a.astype(np.int64), arr_b)
    scale = scale_a * scale_b
    zero_points = (arr_a.sum(axis=-1, keepdims=True) * zero_point_b
                  + arr_b.sum(axis=-2, keepdims=True) * zero_point_a
                  - zero_point_a * zero_point_b * arr_a.shape[-1])
    return q_matmul_result, scale, zero_points

def requantize(arr: np.ndarray, arr_scale: float, arr_zero_points: np.ndarray,
               res_scale: float, res_zero_point: int, bit_width: int):
    min_qval, max_qval = -2.0 ** (bit_width - 1), 2.0 ** (bit_width - 1) - 1.0
    dequant = dequantize(arr, arr_scale, arr_zero_points)
    qdata = np.clip(np.rint(res_zero_point + 1 / res_scale * dequant), min_qval, max_qval).astype(np.int64)
    return qdata

# Float matrix multiplication
z_f32 = np.matmul(x1, x2)

# Quantize input arrays
x1_scale, x1_zero_point = quant_parameters(x1.min(), x1.max(), bit_width=8, asymmetric=True)
x2_scale, x2_zero_point = quant_parameters(x2.min(), x2.max(), bit_width=8, asymmetric=True)
x1_quant = quantize(x1, x1_scale, x1_zero_point, bit_width=8)
x2_quant = quantize(x2, x2_scale, x2_zero_point, bit_width=8)

# Perform matrix multiplication. Result is quantized with a higher bit width, i.e. for `bit_width == 8`
# the elements of result `q_mm` have a bit_width of 32.
y, y_scale, y_zero_points = q_matmul(x1_quant, x1_scale, x1_zero_point, x2_quant, x2_scale, x2_zero_point)

# Requantize to original bit_width, i.e. 8. For that use quantization parameters obtained from `f32_matmul`.
z_scale, z_zero_point = quant_parameters(z_f32.min(), z_f32.max(), bit_width=8, asymmetric=True)
z_quant = requantize(y, y_scale, y_zero_points,
                     z_scale, z_zero_point, bit_width=8)

# Dequantize result
z_round_trip = dequantize(z_quant, z_scale, z_zero_point)

with np.printoptions(precision=4, suppress=True):
    print("z_f32:\n", np.array2string(z_f32))
    print("z_round_trip:\n", np.array2string(z_round_trip))
    print("round-trip error:\n", np.abs(z_f32 - z_round_trip))
