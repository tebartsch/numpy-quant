#!/usr/bin/env python
import io
import pathlib
import unittest
import textwrap
import plotext as plt

import numpy as np
import onnx
import onnxruntime as ort
from sklearn.datasets import make_circles
import torch

from numpy_quant.model import Model
from numpy_quant.tensor import QTensor
from extra.model_summary import summarize


def dataset_plot(X, Y, title=None):
    plt.plot_size(50, 15)
    plt.axes_color('default')
    plt.canvas_color('default')
    plt.ticks_color('default')
    plt.scatter(X[Y == 0][:20, 0], X[Y == 0][:20, 1], label="class 0")
    plt.scatter(X[Y == 1][:20, 0], X[Y == 1][:20, 1], label="class 1")
    if title:
        plt.title(title)
    plt.show()
    plt.clear_figure()


def accuracy_plot(x, y, title=None, xlabel=None):
    plt.plot_size(100, 10)
    plt.axes_color('default')
    plt.canvas_color('default')
    plt.ticks_color('default')
    plt.xticks(x)
    plt.scatter(x, y)
    if title:
        plt.title(title)
    if xlabel:
        plt.xlabel(xlabel)
    plt.show()
    plt.clear_figure()


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


class TestMlp(unittest.TestCase):

    def __init__(self, *args, **kwargs):
        super(TestMlp, self).__init__(*args, **kwargs)

        n_samples = 1000
        X_test, Y_test = make_circles(n_samples=n_samples, noise=0.03)
        X_test = X_test.astype(np.float32)

        onnx_model = onnx.load(pathlib.Path(__file__).parent / ".." / "models" / "mlp.onnx")
        onnx.checker.check_model(onnx_model)

        self.X_test = X_test
        self.Y_test = Y_test
        self.onnx_model = onnx_model

    def test_mlp_onnx_import(self):
        model = Model.from_onnx(self.onnx_model)
        self.assertEqual(
            summarize(model), textwrap.dedent("""\
            =================+=====================+====================
            Node             | Inputs              | Outputs            
            =================+=====================+====================
            /fc1/Gemm        | input               | /fc1/Gemm_output_0 
                             | fc1.weight          |                    
                             | fc1.bias            |                    
            -----------------+---------------------+--------------------
            /relu/Relu       | /fc1/Gemm_output_0  | /relu/Relu_output_0
            -----------------+---------------------+--------------------
            /fc2/Gemm        | /relu/Relu_output_0 | /fc2/Gemm_output_0 
                             | fc2.weight          |                    
                             | fc2.bias            |                    
            -----------------+---------------------+--------------------
            /sigmoid/Sigmoid | /fc2/Gemm_output_0  | output             
            -----------------+---------------------+--------------------
            """)
        )
        print(summarize(model))

    def test_mlp_float_inference(self):
        model = Model.from_onnx(self.onnx_model)
        tinyq_outputs = model([self.X_test])[0]

        onnx_bytes = io.BytesIO()
        onnx.save_model(self.onnx_model, onnx_bytes)
        ort_sess = ort.InferenceSession(onnx_bytes.getvalue())

        print("Tinyquant Float Inference")
        acc = np.mean(tinyq_outputs.argmax(axis=1) == self.Y_test)
        print(f"  Mean Accuracy: {acc:.2f}")
        print()

        actual = tinyq_outputs
        desired = ort_sess.run(None, {'input': self.X_test})[0]
        print(f"Summed difference pytorch vs. numpy-quant: {np.sum(np.abs(actual - desired))}")
        np.testing.assert_allclose(
            actual=actual,
            desired=desired,
            rtol=1e-03,
        )

    def test_mlp_quantization(self):
        model = Model.from_onnx(self.onnx_model)
        qmodel = model.quantize([self.X_test])
        self.assertEqual(
            summarize(qmodel), textwrap.dedent("""\
                    =================+=====================+====================
                    Node             | Inputs              | Outputs            
                    =================+=====================+====================
                    /fc1/Gemm        | input               | /fc1/Gemm_output_0 
                                     | fc1.weight          |                    
                                     | fc1.bias            |                    
                    -----------------+---------------------+--------------------
                    /relu/Relu       | /fc1/Gemm_output_0  | /relu/Relu_output_0
                    -----------------+---------------------+--------------------
                    /fc2/Gemm        | /relu/Relu_output_0 | /fc2/Gemm_output_0 
                                     | fc2.weight          |                    
                                     | fc2.bias            |                    
                    -----------------+---------------------+--------------------
                    /sigmoid/Sigmoid | /fc2/Gemm_output_0  | output             
                    -----------------+---------------------+--------------------
                """)
        )
        print(summarize(qmodel))

    def test_mlp_quantized_inference(self):
        model = Model.from_onnx(self.onnx_model)
        qmodel = model.quantize([self.X_test], bit_width=8)

        outputs = model([self.X_test])[0]
        qoutputs = qmodel([self.X_test])[0]

        print("Mean difference of float and dequantized int tensors")
        qmodel_value_dict = {v.name: v for v in qmodel.values}
        max_name_len = max((len(name) for name in qmodel_value_dict), default=0)
        for value in model.values:
            x = value.data
            qx = qmodel_value_dict[value.name].data
            if isinstance(qx, QTensor):
                mean_diff = np.mean(np.abs(qx.dequantize().data - x.data)) / (x.data.max() - x.data.min())
            else:
                mean_diff = np.mean(np.abs(qx.data - x.data)) / (x.data.max() - x.data.min())
            print(f" - {value.name + ': ':<{max_name_len}} {mean_diff:.4E}")

        print("Quantized Inference")
        acc = np.mean(outputs.argmax(axis=1) == self.Y_test)
        qacc = np.mean(qoutputs.argmax(axis=1) == self.Y_test)
        print(f" - float Mean Accuracy:     {acc:.2f}")
        print(f" - quantized Mean Accuracy: {qacc:.2f}")
        print()

    def test_differing_bit_widths(self):
        model = Model.from_onnx(self.onnx_model)
        bit_width_list = list(range(1, 17))
        q_acc_list = []
        for bit_width in bit_width_list:
            qmodel = model.quantize([self.X_test], bit_width=bit_width)
            qoutputs = qmodel([self.X_test])[0]
            q_acc_mean = np.mean(qoutputs.argmax(axis=1) == self.Y_test)
            q_acc_list.append(q_acc_mean)
        accuracy_plot(bit_width_list, q_acc_list, title="accuracy", xlabel="bit width")
