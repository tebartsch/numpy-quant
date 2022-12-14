#!/usr/bin/env python
import unittest
import pathlib
import textwrap
import numpy as np
import onnx
from sklearn.datasets import make_circles
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import TensorDataset, DataLoader

from tinyquant.Model import Model
from tinyquant.Tensor import FTensor
from extra.model_summary import summarize


class MultiLayerPerceptron(torch.nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(MultiLayerPerceptron, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.fc1 = torch.nn.Linear(self.input_size, self.hidden_size)
        self.relu = torch.nn.ReLU()
        self.fc2 = torch.nn.Linear(self.hidden_size, self.output_size)
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
        X, Y = make_circles(n_samples=n_samples, noise=0.03, random_state=42)
        X = np.array(X, dtype=np.float32)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, train_size=0.7, random_state=0)
        Y_train_one_hot = np.eye(2, dtype=np.float32)[Y_train]  # One-Hot encoding
        trainset = TensorDataset(torch.tensor(X_train), torch.tensor(Y_train_one_hot))
        trainloader = DataLoader(trainset, batch_size=1)

        print("MLP Model Creation")
        torch_model = MultiLayerPerceptron(input_size=X.shape[1], hidden_size=10, output_size=X.shape[1])

        print("MLP Training")
        optimizer = torch.optim.SGD(torch_model.parameters(), lr=0.2)
        criterion = torch.nn.CrossEntropyLoss()
        for epoch in range(4):
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
        file_path = (pathlib.Path(__file__).parent / 'mlp.onnx').resolve()
        torch.onnx.export(torch_model,  # model being run
                          args,  # model input (or a tuple for multiple inputs)
                          file_path,  # where to save the model (can be a file or file-like object)
                          export_params=True,  # store the trained parameter weights inside the model file
                          opset_version=10,  # the ONNX version to export the model to
                          do_constant_folding=True,  # whether to execute constant folding for optimization
                          input_names=['input'],  # the model's input names
                          output_names=['output'],  # the model's output names
                          dynamic_axes={'input': {0: 'batch_size'},  # variable length axes
                                        'output': {0: 'batch_size'}})
        onnx_model = onnx.load(file_path)
        onnx.checker.check_model(onnx_model)

        self.X_test = X_test
        self.Y_test = Y_test
        self.torch_model = torch_model
        self.onnx_model = onnx_model

    def test_mlp_onnx_import(self):
        tinyq_model = Model(self.onnx_model)
        self.assertEqual(
            summarize(tinyq_model), textwrap.dedent("""\
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
                       d      | fc2.bias            |                    
            -----------------+---------------------+--------------------
            /sigmoid/Sigmoid | /fc2/Gemm_output_0  | output             
            -----------------+---------------------+--------------------
            """)
        )
        print(summarize(tinyq_model))

    def test_mlp_float_inference(self):
        tinyq_model = Model(self.onnx_model)
        tinyq_outputs = tinyq_model([FTensor(self.X_test)])[0].data

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
