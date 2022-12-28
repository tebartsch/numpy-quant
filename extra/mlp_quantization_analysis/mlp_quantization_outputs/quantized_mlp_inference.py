import numpy as np

# Quantization routines

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

# Trained MLP for circles dataset

# # Every row of `inp` contains the x and y position of a point
inp = np.array([[0., 0.], [1., 1.]], dtype=np.float32) 

fc1_weight = np.array([[ 4.4037123 , -2.9683902 , -4.4077654 ,  2.3313837 ,  0.05330967], [-1.0420023 ,  3.5323772 , -1.5059234 ,  4.3279686 , -4.243471  ]], dtype=np.float32)
fc1_bias = np.array([-2.0229015, -2.446563 , -2.7381809, -2.715235 , -1.9951458], dtype=np.float32)
fc2_weight = np.array([[ 2.7341564, -2.7900338], [ 3.049221 , -3.114146 ], [ 2.761332 , -2.8246257], [ 3.0681298, -3.1184993], [ 2.8039508, -2.8363247]], dtype=np.float32)
fc2_bias = np.array([-4.372014,  4.457383], dtype=np.float32)

fc1_out = np.matmul(inp, fc1_weight) + fc1_bias  # dense layer #1
fc1_act = fc1_out.copy()
fc1_act[fc1_out < 0] = 0.0  # first layer activation  # activation #1 (relu)
fc2_out = np.matmul(fc1_act, fc2_weight) + fc2_bias  # dense layer #2
fc2_act = 1.0 / (1.0 + np.exp(-fc2_out))  # activation #2 (sigmoid)

# Quantize MLP

# # Input
inp_scale, inp_zero_point = quant_parameters(inp.min(), inp.max(), bit_width=8, asymmetric=True)
inp_q = quantize(inp, inp_scale, inp_zero_point, bit_width=8)

# # FC layer 1
fc1_weight_scale, fc1_weight_zero_point = quant_parameters(fc1_weight.min(), fc1_weight.max(), bit_width=8, asymmetric=False)
fc1_weight_q = quantize(fc1_weight, fc1_weight_scale, fc1_weight_zero_point, bit_width=8)
fc1_out_scale, fc1_out_zero_point = quant_parameters(fc1_out.min(), fc1_out.max(), bit_width=8, asymmetric=True)
fc1_bias_q = quantize(fc1_bias, inp_scale * fc1_weight_scale, 0, bit_width=32)

# # FC layer 2
fc2_weight_scale, fc2_weight_zero_point = quant_parameters(fc2_weight.min(), fc2_weight.max(), bit_width=8, asymmetric=False)
fc2_weight_q = quantize(fc2_weight, fc2_weight_scale, fc2_weight_zero_point, bit_width=8)
fc2_out_scale, fc2_out_zero_point = quant_parameters(fc2_out.min(), fc2_out.max(), bit_width=8, asymmetric=True)
fc2_bias_q = quantize(fc2_bias, fc1_out_scale * fc2_weight_scale, 0, bit_width=32)

# Run inference using quantized MLP

# # FC layer 1
fc1_y, fc1_y_scale, fc1_y_zero_points = q_matmul(inp_q, inp_scale, inp_zero_point,
                                                 fc1_weight_q, fc1_weight_scale, fc1_weight_zero_point)
fc1_out_q = requantize(fc1_y + fc1_bias_q, fc1_y_scale, fc1_y_zero_points,
                       fc1_out_scale, fc1_out_zero_point, bit_width=8)

# # ReLU activation
fc1_act_q = fc1_out_q.copy()
fc1_act_q[fc1_out_q < fc1_out_zero_point] = fc1_out_zero_point

# # FC layer 2
fc2_y, fc2_y_scale, fc2_y_zero_points = q_matmul(fc1_act_q, fc1_out_scale, fc1_out_zero_point,
                                                 fc2_weight_q, fc2_weight_scale, fc2_weight_zero_point)
fc2_out_q = requantize(fc2_y + fc2_bias_q, fc2_y_scale, fc2_y_zero_points,
                       fc2_out_scale, fc2_out_zero_point, bit_width=8)

# Dequantize output of 2. FC layer         
fc2_out_deq = dequantize(fc2_out_q, fc2_out_scale, fc2_out_zero_point)

# # Sigmoid activation on dequantized output of 2. FC layer
fc2_act_deq = 1.0 / (1.0 + np.exp(-fc2_out_deq))

with np.printoptions(precision=4, suppress=True):
    print("fc2_act:\n", np.array2string(fc2_act))
    print("fc2_act_deq:\n", np.array2string(fc2_act_deq))
    print("quantized inference error:\n", np.abs(fc2_act_deq - fc2_act))
