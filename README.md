# numpy-quant

Quantize ONNX-models with arbitrary bit-width.

See [this](https://tebartsch.ai/blog/Quantizing-Transformers.html) blog post about this project.

**Work In Progress**

`numpy-quant` aims to
 
 - allow evaluating neural network quantization methods for arbitrary bit-widths
 - support importing ONNX models
 - be able to quantize common
   - CNNs
   - transformers
   - graph neural networks
 - consist of less than ~~500~~ 1000 lines of code

## Installation

```bash
pip install . --extra-index-url https://download.pytorch.org/whl/cpu
```

## Getting started

Clone numpy-quant, install and switch to folder

```bash
git clone https://github.com/tebartsch/numpy-quant
cd numpy-quant
pip install .
pip install scikit-learn==1.2.0
```

then run

```python
import numpy as np
from numpy_quant.model import Model
import onnx
from sklearn.datasets import make_circles

onnx_model = onnx.load("models/mlp.onnx")

X_calibration, _ = make_circles(n_samples=100, noise=0.03)
X_calibration = X_calibration.astype(np.float32)
X_test, Y_test = make_circles(n_samples=5, noise=0.03)
X_test = X_test.astype(np.float32)

model = Model.from_onnx(onnx_model)
qmodel = model.quantize([X_calibration], bit_width=4)

print("labels")
print(Y_test)
print("predictions (float32 model)")
print(model([X_test])[0].argmax(axis=1))
print("predictions (int4 quantized model)")
```

## Tests

Fast tests
```bash
python models/mlp.py   # Create mlp onnx models
python -m unittest discover -s test -p 'test_*.py' 
```
Slow tests
```
python models/vit_image_classifier.py   # Create vit onnx models
python -m unittest discover -s test/long_running -p 'test_*.py'  # Run long-running tests (> 1 minute in total)
```

## Notes

 - tested with ONNX Opset Version 17
 - ONNX models are expected to not use dynamic axis