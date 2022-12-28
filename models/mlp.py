import io
import pathlib

import numpy as np
import onnx
from sklearn.datasets import make_circles
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import TensorDataset, DataLoader


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


def get_torch_model():
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

    return torch_model, X_train, X_test, Y_train, Y_test


if __name__ == "__main__":
    """Store model as ONNX file"""

    base_path = pathlib.Path(__file__).parent

    torch_model, _, X_test, _, _ = get_torch_model()
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

    onnx.save(onnx_model, base_path / "mlp.onnx")
