"""Quantize multi layer perceptron classifying a circles dataset with different bit widths and visualize results."""
import io
import textwrap

import numpy as np
import onnx
from matplotlib.colors import ListedColormap
from sklearn.datasets import make_circles
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import TensorDataset, DataLoader
import matplotlib.pyplot as plt

from numpy_quant.model import Model
from numpy_quant.tensor import FTensor


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


def create_model_script(fc1_weight: np.ndarray, fc1_bias: np.ndarray,
                        fc2_weight: np.ndarray, fc2_bias: np.ndarray):
    nl = '\n'
    script = textwrap.dedent(f"""\
        import numpy as np

        fc1_weight = np.array({np.array2string(fc1_weight, separator=', ').replace(nl, '')}, dtype=np.float32)
        fc1_bias = np.array({np.array2string(fc1_bias, separator=', ').replace(nl, '')}, dtype=np.float32)
        fc2_weight = np.array({np.array2string(fc2_weight, separator=', ').replace(nl, '')}, dtype=np.float32)
        fc2_bias = np.array({np.array2string(fc2_bias, separator=', ').replace(nl, '')}, dtype=np.float32)

        inp = np.array([[0.0, 0.0],   # x and y position in the middle of the area
                        [1.0, 1.0]])  # x and y position at the outer part of the area
                      
        x = np.matmul(inp, fc1_weight.T) + fc1_bias  # dense layer #1
        x[x < 0] = 0.0  # first layer activation  # activation #1 (relu)
        x = np.matmul(x, fc2_weight.T) + fc2_bias  # dense layer #2
        out = 1.0 / (1.0 + np.exp(-x))  # activation #2 (sigmoid)
        
        with np.printoptions(precision=1, suppress=True):
            print(np.array2string(out))
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
    qmodel = model.quantize([FTensor(X_test)])

    # Store all tensors of the model
    model_value_dict = {v.name: v for v in model.values}
    fc1_weight = model_value_dict["fc1.weight"].data.data
    fc1_bias = model_value_dict["fc1.bias"].data.data
    fc2_weight = model_value_dict["fc2.weight"].data.data
    fc2_bias = model_value_dict["fc2.bias"].data.data

    # Create a simple numpy script running the model
    script = create_model_script(fc1_weight, fc1_bias, fc2_weight, fc2_bias)
    print("Inference Script")
    print("-"*100)
    print(script)
    print("-"*100)
    print("Inference Script output:")
    print("-"*100)
    exec(script)
    print("-"*100)

    # Visualize Model Prediction
    xmin, xmax = -1.25, 1.25
    x = np.linspace(xmin, xmax, 100, dtype=np.float32)
    ymin, ymax = -1.25, 1.25
    y = np.linspace(ymin, ymax, 100, dtype=np.float32)
    xv, yv = np.meshgrid(x, y)
    grid_input = np.concatenate((xv.reshape((-1, 1)), yv.reshape((-1, 1))), axis=1)
    grid_output = model([FTensor(grid_input)])[0].data
    zv = grid_output[:, 1].reshape((x.shape[0], y.shape[0]))

    color_map = ListedColormap(get_color_gradient('#519259', '#E5004C', n=10))

    fig, ax = plt.subplots(1, 1, tight_layout=True)

    ax.contourf(x, y, zv, cmap=color_map, alpha=0.4)
    n_train = 150
    n_test = 70
    ax.scatter(X_train[:n_train, 0], X_train[:n_train, 1], c=Y_train[:n_train],
               cmap=color_map, edgecolors='k', label="train")
    ax.scatter(X_test[:n_test, 0], X_test[:n_test, 1], c=Y_test[:n_test],
               cmap=color_map, alpha=0.4, edgecolors='k', label="test")
    xticks = np.linspace(xmin, xmax, 3)
    ax.set_xticks(xticks, [f"{t:.2f}" for t in xticks])
    yticks = np.linspace(ymin, ymax, 3)
    ax.set_yticks(yticks, [f"{t:.2f}" for t in yticks])
    ax.legend()
    fig.savefig("trained_model_predictions.png", transparent=True)

    # Visualize Model tensors
