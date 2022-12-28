"""Quantize multi layer perceptron classifying a circles dataset with different bit widths and visualize results."""
import io
import re
import sys
import textwrap

import numpy as np
import onnx
import requests
from fontTools.ttLib import woff2
from tempfile import NamedTemporaryFile
import matplotlib
from matplotlib import font_manager
from matplotlib.colors import ListedColormap
from sklearn.datasets import make_circles
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import TensorDataset, DataLoader
import matplotlib.pyplot as plt

from numpy_quant.model import Model
from numpy_quant.tensor import FTensor


def get_google_fonts_ttf(weight: int):
    # Chose user agent with https://stackoverflow.com/questions/25011533/google-font-api-uses-browser-detection-how-to-get-all-font-variations-for-font
    content = requests.get(
        url=f"https://fonts.googleapis.com/css2?family=Roboto+Slab:wght@{weight}&display=swap",
        headers={
            "User-Agent": "Mozilla/5.0 (X11; Linux x86_64; rv:107.0) Gecko/20100101 Firefox/107.0",
        }).text

    font_url_regex = r"/\* latin \*/[\s\S]*url\((.*/[a-zA-Z0-9-_]*.woff2)\)"
    # Download fonts
    font_links = re.findall(font_url_regex, content)
    font_url = font_links[0]
    font_response = requests.get(url=font_url).content
    with NamedTemporaryFile(suffix='.woff2') as infile:
        infile.write(font_response)
        with NamedTemporaryFile(suffix='.ttf') as outfile:
            woff2.decompress(infile, outfile)
            outfile.seek(0)
            font_ttf = outfile.read()

    return font_ttf


# Set a custom font
ttf_file = get_google_fonts_ttf(weight=300)
f = NamedTemporaryFile(delete=False, suffix='.woff2')
f.write(ttf_file)
f.close()
font_manager.fontManager.addfont(f.name)
prop = font_manager.FontProperties(fname=f.name)
font = {'size': 14, 'sans-serif': prop.get_name(), 'weight': 350}
matplotlib.rc('font', **font)


def hex_to_RGB(hex_str):
    """ #FFFFFF -> [255,255,255]"""
    # Pass 16 to the integer function for change of base
    return [int(hex_str[i:i + 2], 16) for i in range(1, 6, 2)]


def get_color_gradient(c1, c2, n):
    """
    Given two hex colors, returns a color gradient
    with n colors.

    Source: https://medium.com/@BrendanArtley/matplotlib-color-gradients-21374910584b
    """
    assert n > 1
    c1_rgb = np.array(hex_to_RGB(c1)) / 255
    c2_rgb = np.array(hex_to_RGB(c2)) / 255
    mix_pcts = [x / (n - 1) for x in range(n)]
    rgb_colors = [((1 - mix) * c1_rgb + (mix * c2_rgb)) for mix in mix_pcts]
    return ["#" + "".join([format(int(round(val * 255)), "02x") for val in item]) for item in rgb_colors]


main_colors = ['#519259', '#E5004C', '#0066bb', '#FFB100']
color_gradient_color_map = ListedColormap(get_color_gradient(main_colors[0], main_colors[1], n=10))


def create_np_array_markdown_table(arrays: dict[str, np.ndarray]):
    table_shape = (max(a.shape[0] for a in arrays.values()),
                   sum(a.shape[1] + 1 for a in arrays.values()) - 1)
    cell_content_strings = np.full(table_shape, fill_value="", dtype=object)
    cell_content_lengths = np.zeros(table_shape, dtype=np.int64)

    curr_col = 0
    for name, arr in arrays.items():
        for r in range(arr.shape[0]):
            for c in range(arr.shape[1]):
                if arr.dtype == np.float32:
                    string_val = f"{arr[r, c]:.4f}"
                elif arr.dtype == np.int64:
                    string_val = f"{arr[r, c]}"
                else:
                    raise ValueError(f"Unsupported dtype of np.ndarray: {arr.dtype}")
                cell_content_strings[r, curr_col + c] = string_val
                cell_content_lengths[r, curr_col + c] = len(string_val)
        curr_col += arr.shape[1] + 1

    col_widths = cell_content_lengths.max(axis=0, initial=2)
    curr_col = 0
    for name, arr in arrays.items():
        col_widths[curr_col] = max(col_widths[curr_col], len(name))
        curr_col += arr.shape[1] + 1

    res = ""

    # Header
    curr_col = 0
    res += "|"
    for name, arr in arrays.items():
        res += f"{name:>{col_widths[curr_col]}}" + "|"
        res += "|".join(" " * l for l in col_widths[curr_col + 1:curr_col + arr.shape[1] + 1])
        if curr_col + arr.shape[1] + 1 < len(col_widths):
            res += "|"
        curr_col += arr.shape[1] + 1
    res += "\n"
    # Horizontal Line
    res += "|:" + ":|:".join("-" * (l - 2) for l in col_widths) + ":|\n"
    # Content
    for row in cell_content_strings:
        res += "|" + "|".join(f"{e:>{l}}" for e, l in zip(row, col_widths)) + "|" + "\n"

    return res


def create_tensor_quantization_script_v1(arr: np.ndarray, scale: float, zero_point: int):
    nl = '\n'
    script = textwrap.dedent(f"""\
        import numpy as np

        arr = np.array({np.array2string(arr, separator=', ').replace(nl, '')}, dtype=np.float32)

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
        scale = np.array({scale}, np.float32)
        zero_point = np.array({zero_point}, np.int64)
        
        arr_quantized = quantize(arr, scale, zero_point, bit_width=8)
        arr_round_trip = dequantize(arr_quantized, scale, zero_point)
        
        with np.printoptions(precision=4, suppress=True):
            print("arr:\\n", np.array2string(arr))
            print("arr_quantized:\\n", np.array2string(arr_quantized))
            print("arr_round_trip:\\n", np.array2string(arr_round_trip))
            print("round-trip error:\\n", np.abs(arr - arr_round_trip))
    """)

    return script


def create_tensor_quantization_script_v2(arr: np.ndarray):
    nl = '\n'
    script = textwrap.dedent(f"""\
        import numpy as np

        arr = np.array({np.array2string(arr, separator=', ').replace(nl, '')}, dtype=np.float32)

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

        scale, zero_point = quant_parameters(arr.min(), arr.max(), bit_width=8, asymmetric=True)

        arr_quantized = quantize(arr, scale, zero_point, bit_width=8)
        arr_round_trip = dequantize(arr_quantized, scale, zero_point)

        with np.printoptions(precision=4, suppress=True):
            print(f"scale = {{scale}}, zero_point = {{zero_point}}")
            print("arr:\\n", np.array2string(arr))
            print("arr_quantized:\\n", np.array2string(arr_quantized))
            print("arr_round_trip:\\n", np.array2string(arr_round_trip))
            print("round-trip error:\\n", np.abs(arr - arr_round_trip))
    """)

    return script


def create_quantized_matmul_script(x1: np.ndarray, x2: np.ndarray):
    nl = '\n'
    script = textwrap.dedent(f"""\
        import numpy as np

        x1 = np.array({np.array2string(x1, separator=', ').replace(nl, '')}, dtype=np.float32)
        x2 = np.array({np.array2string(x2, separator=', ').replace(nl, '')}, dtype=np.float32)

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

        # Float matrix multiplication
        w_f32 = np.matmul(x1, x2)

        # Quantize input arrays
        x1_scale, x1_zero_point = quant_parameters(x1.min(), x1.max(), bit_width=8, asymmetric=True)
        x2_scale, x2_zero_point = quant_parameters(x2.min(), x2.max(), bit_width=8, asymmetric=True)
        x1_quant = quantize(x1, x1_scale, x1_zero_point, bit_width=8)
        x2_quant = quantize(x2, x2_scale, x2_zero_point, bit_width=8)

        # Perform matrix multiplication. Result is quantized with a higher bit width, i.e. for `bit_width == 8`
        # the elements of result `q_mm` have a bit_width of 32.
        y, y_scale, y_zero_points = q_matmul(x1_quant, x1_scale, x1_zero_point, x2_quant, x2_scale, x2_zero_point)

        # Requantize to original bit_width, i.e. 8. For that use quantization parameters obtained from `f32_matmul`.
        w_scale, w_zero_point = quant_parameters(w_f32.min(), w_f32.max(), bit_width=8, asymmetric=True)
        w_quant = requantize(y, y_scale, y_zero_points,
                             w_scale, w_zero_point, bit_width=8)

        # Dequantize result
        w_round_trip = dequantize(w_quant, w_scale, w_zero_point)

        with np.printoptions(precision=4, suppress=True):
            print("w_f32:\\n", np.array2string(w_f32))
            print("w_round_trip:\\n", np.array2string(w_round_trip))
            print("round-trip error:\\n", np.abs(w_f32 - w_round_trip))
    """)

    return script


def create_quantized_inference_script(inp: np.ndarray,
                                      fc1_weight: np.ndarray, fc1_bias: np.ndarray,
                                      fc2_weight: np.ndarray, fc2_bias: np.ndarray,
                                      bit_width: int,
                                      weights_asymmetric: bool = False,
                                      activations_asymmetric: bool = True):
    nl = '\n'
    script = textwrap.dedent(f"""\
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
        inp = np.array({np.array2string(inp, separator=', ', threshold=sys.maxsize).replace(nl, '')}, dtype=np.float32) 

        fc1_weight = np.array({np.array2string(fc1_weight, separator=', ').replace(nl, '')}, dtype=np.float32)
        fc1_bias = np.array({np.array2string(fc1_bias, separator=', ').replace(nl, '')}, dtype=np.float32)
        fc2_weight = np.array({np.array2string(fc2_weight, separator=', ').replace(nl, '')}, dtype=np.float32)
        fc2_bias = np.array({np.array2string(fc2_bias, separator=', ').replace(nl, '')}, dtype=np.float32)
                      
        fc1_out = np.matmul(inp, fc1_weight) + fc1_bias  # dense layer #1
        fc1_act = fc1_out.copy()
        fc1_act[fc1_out < 0] = 0.0  # first layer activation  # activation #1 (relu)
        fc2_out = np.matmul(fc1_act, fc2_weight) + fc2_bias  # dense layer #2
        fc2_act = 1.0 / (1.0 + np.exp(-fc2_out))  # activation #2 (sigmoid)
        
        # Quantize MLP
        
        # # Input
        inp_scale, inp_zero_point = quant_parameters(inp.min(), inp.max(), bit_width={bit_width}, asymmetric={activations_asymmetric})
        inp_q = quantize(inp, inp_scale, inp_zero_point, bit_width={bit_width})
        
        # # FC layer 1
        fc1_weight_scale, fc1_weight_zero_point = quant_parameters(fc1_weight.min(), fc1_weight.max(), bit_width={bit_width}, asymmetric={weights_asymmetric})
        fc1_weight_q = quantize(fc1_weight, fc1_weight_scale, fc1_weight_zero_point, bit_width={bit_width})
        fc1_out_scale, fc1_out_zero_point = quant_parameters(fc1_out.min(), fc1_out.max(), bit_width={bit_width}, asymmetric={activations_asymmetric})
        fc1_bias_q = quantize(fc1_bias, inp_scale * fc1_weight_scale, 0, bit_width=32)
        
        # # FC layer 2
        fc2_weight_scale, fc2_weight_zero_point = quant_parameters(fc2_weight.min(), fc2_weight.max(), bit_width={bit_width}, asymmetric={weights_asymmetric})
        fc2_weight_q = quantize(fc2_weight, fc2_weight_scale, fc2_weight_zero_point, bit_width={bit_width})
        fc2_out_scale, fc2_out_zero_point = quant_parameters(fc2_out.min(), fc2_out.max(), bit_width={bit_width}, asymmetric={activations_asymmetric})
        fc2_bias_q = quantize(fc2_bias, fc1_out_scale * fc2_weight_scale, 0, bit_width=32)
        
        # Run inference using quantized MLP
        
        # # FC layer 1
        fc1_y, fc1_y_scale, fc1_y_zero_points = q_matmul(inp_q, inp_scale, inp_zero_point,
                                                         fc1_weight_q, fc1_weight_scale, fc1_weight_zero_point)
        fc1_out_q = requantize(fc1_y + fc1_bias_q, fc1_y_scale, fc1_y_zero_points,
                               fc1_out_scale, fc1_out_zero_point, bit_width={bit_width})
        
        # # ReLU activation
        fc1_act_q = fc1_out_q.copy()
        fc1_act_q[fc1_out_q < fc1_out_zero_point] = fc1_out_zero_point
        
        # # FC layer 2
        fc2_y, fc2_y_scale, fc2_y_zero_points = q_matmul(fc1_act_q, fc1_out_scale, fc1_out_zero_point,
                                                         fc2_weight_q, fc2_weight_scale, fc2_weight_zero_point)
        fc2_out_q = requantize(fc2_y + fc2_bias_q, fc2_y_scale, fc2_y_zero_points,
                               fc2_out_scale, fc2_out_zero_point, bit_width={bit_width})
        
        # Dequantize output of 2. FC layer         
        fc2_out_deq = dequantize(fc2_out_q, fc2_out_scale, fc2_out_zero_point)
           
        # # Sigmoid activation on dequantized output of 2. FC layer
        fc2_act_deq = 1.0 / (1.0 + np.exp(-fc2_out_deq))
        
        with np.printoptions(precision=4, suppress=True):
            print("fc2_act:\\n", np.array2string(fc2_act))
            print("fc2_act_deq:\\n", np.array2string(fc2_act_deq))
            print("quantized inference error:\\n", np.abs(fc2_act_deq - fc2_act))
    """)

    return script


class MultiLayerPerceptron(torch.nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(MultiLayerPerceptron, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.fc1 = torch.nn.Linear(self.input_size, self.hidden_size, bias=True)
        self.relu = torch.nn.ReLU()
        self.fc2 = torch.nn.Linear(self.hidden_size, self.output_size, bias=True)
        self.sigmoid = torch.nn.Sigmoid()

    def forward(self, x):
        hidden = self.fc1(x)
        relu = self.relu(hidden)
        output = self.fc2(relu)
        output = self.sigmoid(output)
        return output


if __name__ == "__main__":
    np.random.seed(0)
    torch.manual_seed(1)

    print("MLP Dataset")
    n_samples = 1000
    X, Y = make_circles(n_samples=n_samples, noise=0.03)
    X = np.array(X, dtype=np.float32)
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, train_size=0.7, random_state=0)
    Y_train_one_hot = np.eye(2, dtype=np.float32)[Y_train]  # One-Hot encoding
    trainset = TensorDataset(torch.tensor(X_train), torch.tensor(Y_train_one_hot))
    trainloader = DataLoader(trainset, batch_size=1)

    print("MLP Model Creation")
    torch_model = MultiLayerPerceptron(input_size=X.shape[1], hidden_size=5, output_size=X.shape[1])

    print("MLP Training")
    optimizer = torch.optim.SGD(torch_model.parameters(), lr=0.2)
    criterion = torch.nn.CrossEntropyLoss()
    for epoch in range(10):
        average_loss = 0.0
        for i, (x, y) in enumerate(trainloader):
            optimizer.zero_grad()
            outputs = torch_model(x)
            loss = criterion(outputs, y)
            loss.backward()
            optimizer.step()

            average_loss += loss.item()
        average_loss /= len(trainset)
        test_outputs = torch_model(torch.tensor(X_test, requires_grad=False)).detach().numpy()
        acc = np.mean(test_outputs.argmax(axis=1) == Y_test)
        print(f" - Epoch: {epoch:2d}, Mean Accuracy: {acc:.2f}, Average Loss: {average_loss:.2f}")

    print("MLP ONNX export")
    args = torch.Tensor(X_test)
    onnx_model_bytes = io.BytesIO()
    torch.onnx.export(torch_model,
                      args,
                      onnx_model_bytes,
                      export_params=True,
                      opset_version=10,
                      do_constant_folding=True,
                      input_names=['input'],
                      output_names=['output'],
                      dynamic_axes={'input': {0: 'batch_size'},
                                    'output': {0: 'batch_size'}})
    onnx_model = onnx.load_from_string(onnx_model_bytes.getvalue())
    onnx.checker.check_model(onnx_model)

    model = Model.from_onnx(onnx_model)

    # Store all tensors of the model
    model_value_dict = {v.name: v for v in model.values}
    fc1_weight = model_value_dict["fc1.weight"].data.data
    fc1_bias = model_value_dict["fc1.bias"].data.data
    fc2_weight = model_value_dict["fc2.weight"].data.data
    fc2_bias = model_value_dict["fc2.bias"].data.data

    # Create a simple numpy script quantizing & dequantizing fc1_weight for given quantization parameters
    scale, zero_point = 0.04, 0
    arr = fc1_weight[:, 0:1]
    tensor_quantization_script_v1 = create_tensor_quantization_script_v1(arr, scale, zero_point)
    print("Tensor quantization Script")
    print("-" * 100)
    print(tensor_quantization_script_v1)
    print("-" * 100)
    print("Tensor quantization Script output:")
    print("-" * 100)
    loc = {}
    exec(tensor_quantization_script_v1, globals(), loc)
    print("-" * 100)
    with open("tensor_quantization_outputs/tensor_quantization_v1.py", "w") as f:
        f.write(tensor_quantization_script_v1)
    # # Create markdown table of the script's output
    quantize = loc["quantize"]
    dequantize = loc["dequantize"]
    arr_quantized = quantize(arr, scale, zero_point, bit_width=8)
    arr_round_trip = dequantize(arr_quantized, scale, zero_point)
    table = create_np_array_markdown_table({"f32 origin": arr,
                                            "int8 quantized": arr_quantized,
                                            "f32 round-tripped": arr_round_trip,
                                            "round-trip error": np.abs(arr - arr_round_trip)})
    with open("tensor_quantization_outputs/tensor_quantization_v1_output.md", "w") as f:
        f.write(table)

    # Create a simple numpy script quantizing & dequantizing fc1_weight
    arr = fc1_weight[:, 0:1]
    tensor_quantization_script_v2 = create_tensor_quantization_script_v2(arr)
    print("Tensor quantization Script")
    print("-" * 100)
    print(tensor_quantization_script_v2)
    print("-" * 100)
    print("Tensor quantization Script output:")
    print("-" * 100)
    loc = {}
    exec(tensor_quantization_script_v2, globals(), loc)
    print("-" * 100)
    with open("tensor_quantization_outputs/tensor_quantization_v2.py", "w") as f:
        f.write(tensor_quantization_script_v2)
    # # Create markdown table of the script's output
    quant_parameters = loc["quant_parameters"]
    quantize = loc["quantize"]
    dequantize = loc["dequantize"]
    scale, zero_point = quant_parameters(arr.min(), arr.max(), bit_width=8, asymmetric=True)
    arr_quantized = quantize(arr, scale, zero_point, bit_width=8)
    arr_round_trip = dequantize(arr_quantized, scale, zero_point)
    table = create_np_array_markdown_table({"f32 origin": arr,
                                            "int8 quantized": arr_quantized,
                                            "f32 round-tripped": arr_round_trip,
                                            "round-trip error": np.abs(arr - arr_round_trip)})
    with open("tensor_quantization_outputs/tensor_quantization_v2_output.md", "w") as f:
        f.write(table)

    # Test Tensor quantization for different bit widths
    arr = fc1_weight[:, 0:1]
    bit_widths = np.arange(1, 18 + 1)
    scales_symm = np.zeros(bit_widths.shape)
    scales_asymm = np.zeros(bit_widths.shape)
    zero_points_asymm = np.zeros(bit_widths.shape)
    mean_errors_symm = np.zeros(bit_widths.shape)
    mean_errors_asymm = np.zeros(bit_widths.shape)
    for i, bw in enumerate(bit_widths):
        scale, zero_point = quant_parameters(arr.min(), arr.max(), bit_width=bw, asymmetric=False)
        arr_quantized = quantize(arr, scale, zero_point, bit_width=bw)
        arr_round_trip = dequantize(arr_quantized, scale, zero_point)
        scales_symm[i] = scale
        mean_errors_symm[i] = np.mean(np.abs(arr - arr_round_trip))
        scale, zero_point = quant_parameters(arr.min(), arr.max(), bit_width=bw, asymmetric=True)
        arr_quantized = quantize(arr, scale, zero_point, bit_width=bw)
        arr_round_trip = dequantize(arr_quantized, scale, zero_point)
        scales_asymm[i] = scale
        zero_points_asymm[i] = zero_point
        mean_errors_asymm[i] = np.mean(np.abs(arr - arr_round_trip))
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.3333), tight_layout=True)
    axes[0].plot(bit_widths, mean_errors_symm, color=main_colors[0], label="symmetric quantization")
    axes[0].plot(bit_widths, mean_errors_asymm, color=main_colors[1], label="asymmetric quantization")
    axes[0].set_title("mean absolute quantization error")
    axes[0].set_yscale("log")
    axes[0].set_xticks(np.concatenate((np.arange(1, 10), np.arange(10, np.max(bit_widths) + 1, 2))))
    axes[0].set_xlabel("bit width")
    axes[0].legend()
    axes[1].plot(bit_widths, zero_points_asymm, color=main_colors[2], label="zero point\n(asymmetric quantization)")
    axes[1].set_title("quantization parameters")
    axes[1].set_xticks(np.concatenate((np.arange(1, 10), np.arange(10, np.max(bit_widths) + 1, 2))))
    axes[1].set_xlabel("bit width")
    axes[1].legend()
    fig.savefig("tensor_quantization_outputs/tensor_quantization_v2_varying_bit_width.png", transparent=True,
                dpi=500)

    # Create a simple numpy script performing a quantized matrix-matrix multiplication
    x1, x2 = X_test[:3, :], fc1_weight.T
    q_matmul_script = create_quantized_matmul_script(x1, x2)
    print("Quantization Matmul Script")
    print("-" * 100)
    print(q_matmul_script)
    print("-" * 100)
    print("Quantization Matmul Script output:")
    print("-" * 100)
    loc = {}
    exec(q_matmul_script, globals(), loc)
    print("-" * 100)
    with open("matmul_quantization_outputs/matmul_quantization.py", "w") as f:
        f.write(q_matmul_script)


    def perform_quantized_matmul(_x1: np.array, _x1_asymm: bool, _x2: np.array, _x2_asymm: bool,
                                 bit_width: int):
        quant_parameters = loc["quant_parameters"]
        quantize = loc["quantize"]
        dequantize = loc["dequantize"]
        q_matmul = loc["q_matmul"]
        requantize = loc["requantize"]

        z_f32 = np.matmul(_x1, _x2)

        _x1_scale, _x1_zero_point = quant_parameters(_x1.min(), _x1.max(), bit_width=bit_width, asymmetric=_x1_asymm)
        _x2_scale, _x2_zero_point = quant_parameters(_x2.min(), _x2.max(), bit_width=bit_width, asymmetric=_x2_asymm)
        _x1_quant = quantize(_x1, _x1_scale, _x1_zero_point, bit_width=bit_width)
        _x2_quant = quantize(_x2, _x2_scale, _x2_zero_point, bit_width=bit_width)
        y, y_scale, y_zero_points = q_matmul(_x1_quant, _x1_scale, _x1_zero_point, _x2_quant, _x2_scale, _x2_zero_point)

        z_scale, z_zero_point = quant_parameters(z_f32.min(), z_f32.max(), bit_width=bit_width, asymmetric=True)
        z_quant = requantize(y, y_scale, y_zero_points,
                             z_scale, z_zero_point, bit_width=bit_width)

        z_round_trip = dequantize(z_quant, z_scale, z_zero_point)

        return z_f32, z_round_trip


    # # Create markdown table of the script's output
    z_f32, z_round_trip = perform_quantized_matmul(x1, True, x2, True, bit_width=8)
    table = create_np_array_markdown_table({"f32 matmul": z_f32,
                                            "quantized mamtul": z_round_trip,
                                            "error": np.abs(z_f32 - z_round_trip)})
    with open("matmul_quantization_outputs/matmul_quantization_output.md", "w") as f:
        f.write(table)

    with np.printoptions(precision=4, suppress=True):
        print("z_f32:\\n", np.array2string(z_f32))
        print("z_round_trip:\\n", np.array2string(z_round_trip))
        print("round-trip error:\\n", np.abs(z_f32 - z_round_trip))

    # Test matmul quantization for different bit widths
    arr = fc1_weight[:, 0:1]
    bit_widths = np.arange(1, 18 + 1)
    mean_errors_symm_symm = np.zeros(bit_widths.shape)
    mean_errors_asymm_symm = np.zeros(bit_widths.shape)
    mean_errors_symm_asymm = np.zeros(bit_widths.shape)
    mean_errors_asymm_asymm = np.zeros(bit_widths.shape)
    for i, bw in enumerate(bit_widths):
        z_f32, z_round_trip = perform_quantized_matmul(x1, False, x2, False, bit_width=bw)
        mean_errors_symm_symm[i] = np.mean(np.abs(z_f32 - z_round_trip))
        z_f32, z_round_trip = perform_quantized_matmul(x1, True, x2, False, bit_width=bw)
        mean_errors_asymm_symm[i] = np.mean(np.abs(z_f32 - z_round_trip))
        z_f32, z_round_trip = perform_quantized_matmul(x1, False, x2, True, bit_width=bw)
        mean_errors_symm_asymm[i] = np.mean(np.abs(z_f32 - z_round_trip))
        z_f32, z_round_trip = perform_quantized_matmul(x1, True, x2, True, bit_width=bw)
        mean_errors_asymm_asymm[i] = np.mean(np.abs(z_f32 - z_round_trip))
    fig, ax = plt.subplots(1, 1, figsize=(10, 3.3333), tight_layout=True)
    ax.plot(bit_widths, mean_errors_asymm_asymm, color=main_colors[3], label="asymmetric x asymmetric")
    ax.plot(bit_widths, mean_errors_symm_asymm, color=main_colors[2], label="symmetric x asymmetric")
    ax.plot(bit_widths, mean_errors_asymm_symm, color=main_colors[1], label="asymmetric x symmetric")
    ax.plot(bit_widths, mean_errors_symm_symm, color=main_colors[0], label="symmetric x symmetric")
    ax.set_title("mean absolute quantization error")
    ax.set_yscale("log")
    ax.set_xticks(np.concatenate((np.arange(1, 10), np.arange(10, np.max(bit_widths) + 1, 2))))
    ax.set_xlabel("bit width")
    ax.legend()
    fig.savefig("matmul_quantization_outputs/matmul_quantization_varying_bit_width.png", transparent=True,
                dpi=500)

    # Create a simple numpy script running the model
    inp = np.array([[0.0, 0.0],
                    [1.0, 1.0]], dtype=np.float32)
    quantized_inference_script = create_quantized_inference_script(inp, fc1_weight.T, fc1_bias, fc2_weight.T, fc2_bias,
                                                                   bit_width=8)
    print("Float inference Script")
    print("-" * 100)
    print(quantized_inference_script)
    print("-" * 100)
    print("Float inference Script output:")
    print("-" * 100)
    exec(quantized_inference_script, {})
    print("-" * 100)
    with open("mlp_quantization_outputs/quantized_mlp_inference.py", "w") as f:
        f.write(quantized_inference_script)


    def run_quantized_mlp_inference(_inp: np.array, bit_width: int,
                                    weights_asymmetric: bool = False,
                                    activations_asymmetric: bool = True):
        curr_script = create_quantized_inference_script(_inp,
                                                        fc1_weight.T, fc1_bias,
                                                        fc2_weight.T, fc2_bias,
                                                        bit_width=bit_width,
                                                        weights_asymmetric=weights_asymmetric,
                                                        activations_asymmetric=activations_asymmetric)
        loc = {}
        exec(curr_script, globals(), loc)
        f32_result = loc["fc2_act"]
        q_result = loc["fc2_act_deq"]
        return f32_result, q_result


    # Plot Accuracy Drop
    def visualize_quantized_mlp_accuracy_drop():
        outputs = run_quantized_mlp_inference(X_test, bit_width=8)[0]
        acc_mean_baseline = np.mean(outputs.argmax(axis=1) == Y_test)

        bit_widths = np.arange(1, 12 + 1, dtype=int)
        q_acc_w_symm_a_symm = np.zeros(bit_widths.shape)
        q_acc_w_symm_a_asymm = np.zeros(bit_widths.shape)
        q_acc_w_asymm_a_symm = np.zeros(bit_widths.shape)
        q_acc_w_asymm_a_asymm = np.zeros(bit_widths.shape)
        for i, bw in enumerate(bit_widths):
            qoutputs = run_quantized_mlp_inference(X_test, bit_width=bw,
                                                   weights_asymmetric=False, activations_asymmetric=False)[1]
            q_acc_mean = np.mean(qoutputs.argmax(axis=1) == Y_test)
            q_acc_w_symm_a_symm[i] = q_acc_mean
            qoutputs = run_quantized_mlp_inference(X_test, bit_width=bw,
                                                   weights_asymmetric=False, activations_asymmetric=True)[1]
            q_acc_mean = np.mean(qoutputs.argmax(axis=1) == Y_test)
            q_acc_w_symm_a_asymm[i] = q_acc_mean
            qoutputs = run_quantized_mlp_inference(X_test, bit_width=bw,
                                                   weights_asymmetric=True, activations_asymmetric=False)[1]
            q_acc_mean = np.mean(qoutputs.argmax(axis=1) == Y_test)
            q_acc_w_asymm_a_symm[i] = q_acc_mean
            qoutputs = run_quantized_mlp_inference(X_test, bit_width=bw,
                                                   weights_asymmetric=True, activations_asymmetric=True)[1]
            q_acc_mean = np.mean(qoutputs.argmax(axis=1) == Y_test)
            q_acc_w_asymm_a_asymm[i] = q_acc_mean
        fig, ax = plt.subplots(1, 1, tight_layout=True, figsize=(10, 3.3333))
        ax.axhline(acc_mean_baseline, color="grey", label="baseline")
        ax.set_title("Quantization accuracy drop for varying bit widths")
        ax.set_xlabel("bit width")
        ax.set_ylabel("test accuracy")
        ax.plot(bit_widths, q_acc_w_asymm_a_asymm, color=main_colors[3], label="asymmetric weights, asymmetric activations")
        ax.plot(bit_widths, q_acc_w_asymm_a_symm, color=main_colors[2], label="asymmetric weights, symmetric activations")
        ax.plot(bit_widths, q_acc_w_symm_a_asymm, color=main_colors[1], label="symmetric weights, asymmetric activations")
        ax.plot(bit_widths, q_acc_w_symm_a_symm, color=main_colors[0], label="symmetric weights, symmetric activations")
        ax.legend(fontsize=10)
        ax.set_xticks(bit_widths)

        return fig, ax


    fig, ax = visualize_quantized_mlp_accuracy_drop()
    fig.savefig("mlp_quantization_outputs/mlp_quantization_accuracy_drop.png", transparent=True)


    # Visualize Model Prediction
    def visualize_model_prediction(bit_width: int | None = None):
        xmin, xmax = -1.25, 1.25
        x = np.linspace(xmin, xmax, 100, dtype=np.float32)
        ymin, ymax = -1.25, 1.25
        y = np.linspace(ymin, ymax, 100, dtype=np.float32)
        xv, yv = np.meshgrid(x, y)
        grid_input = np.concatenate((xv.reshape((-1, 1)), yv.reshape((-1, 1))), axis=1)
        # grid_output = _model([FTensor(grid_input)])[0].data
        if bit_width is None:
            grid_output = run_quantized_mlp_inference(grid_input, bit_width=8)[0]
        else:
            grid_output = run_quantized_mlp_inference(grid_input, bit_width=bit_width)[1]
        zv = grid_output[:, 1].reshape((x.shape[0], y.shape[0]))

        fig, ax = plt.subplots(1, 1, tight_layout=True, figsize=(4, 4))

        ax.contourf(x, y, zv, cmap=color_gradient_color_map, alpha=0.4)
        n_train = 150
        n_test = 70
        ax.scatter(X_train[:n_train, 0], X_train[:n_train, 1], c=Y_train[:n_train],
                   cmap=color_gradient_color_map, edgecolors='k', label="train")
        ax.scatter(X_test[:n_test, 0], X_test[:n_test, 1], c=Y_test[:n_test],
                   cmap=color_gradient_color_map, alpha=0.4, edgecolors='k', label="test")
        xticks = np.linspace(xmin, xmax, 3)
        ax.set_xticks(xticks, [f"{t:.2f}" for t in xticks])
        yticks = np.linspace(ymin, ymax, 3)
        ax.set_yticks(yticks, [f"{t:.2f}" for t in yticks])
        ax.legend(loc="upper right")
        return fig, ax


    fig, ax = visualize_model_prediction()
    fig.savefig("mlp_quantization_outputs/mlp_float_predictions.png", transparent=True)
    for bw in np.arange(1, 12 + 1, dtype=int):
        # TODO qmodel = model.quantize([FTensor(X_test)], bit_width=bw)
        fig, ax = visualize_model_prediction(bit_width=bw)
        fig.savefig(f"mlp_quantization_outputs/mlp_int{bw}_predictions.png", transparent=True)
