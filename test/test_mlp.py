#!/usr/bin/env python
import io
import unittest
import pathlib
import textwrap
import plotext as plt

import numpy as np
import onnx
from sklearn.datasets import make_circles
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import TensorDataset, DataLoader

from tinyquant.model import Model
from tinyquant.tensor import FTensor, QTensor
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

        print("MLP Dataset")
        n_samples = 1000
        X, Y = make_circles(n_samples=n_samples, noise=0.03)
        X = np.array(X, dtype=np.float32)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, train_size=0.7, random_state=0)
        Y_train_one_hot = np.eye(2, dtype=np.float32)[Y_train]  # One-Hot encoding
        trainset = TensorDataset(torch.tensor(X_train), torch.tensor(Y_train_one_hot))
        trainloader = DataLoader(trainset, batch_size=1)

        dataset_plot(X[:100, :], Y[:100], title="Dataset")

        print("MLP Model Creation")
        torch_model = MultiLayerPerceptron(input_size=X.shape[1], hidden_size=10, output_size=X.shape[1])

        print("MLP Training")
        optimizer = torch.optim.SGD(torch_model.parameters(), lr=0.2)
        criterion = torch.nn.CrossEntropyLoss()
        for epoch in range(5):
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

        self.X_test = X_test
        self.Y_test = Y_test
        self.torch_model = torch_model
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
        tinyq_outputs = model([FTensor(self.X_test)])[0].data

        print("Tinyquant Float Inference")
        acc = np.mean(tinyq_outputs.argmax(axis=1) == self.Y_test)
        print(f"  Mean Accuracy: {acc:.2f}")
        print()

        actual = tinyq_outputs
        desired = self.torch_model(torch.tensor(self.X_test, requires_grad=False)).detach().numpy()
        print(f"Summed difference pytorch vs. tinyquant: {np.sum(np.abs(actual - desired))}")
        np.testing.assert_allclose(
            actual=actual,
            desired=desired,
            rtol=1e-05,
        )

    def test_mlp_quantization(self):
        model = Model.from_onnx(self.onnx_model)
        qmodel = model.quantize([FTensor(self.X_test)])
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
        qmodel = model.quantize([FTensor(self.X_test)], bit_width=8)

        outputs = model([FTensor(self.X_test)])[0].data
        qoutputs = qmodel([FTensor(self.X_test)])[0].data

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
            qmodel = model.quantize([FTensor(self.X_test)], bit_width=bit_width)
            qoutputs = qmodel([FTensor(self.X_test)])[0].data
            q_acc_sum = np.mean(qoutputs.argmax(axis=1) == self.Y_test)
            q_acc_list.append(q_acc_sum)
        accuracy_plot(bit_width_list, q_acc_list, title="accuracy", xlabel="bit width")
