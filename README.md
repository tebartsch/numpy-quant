# tinyquant

Evaluate ONNX-model quantization with arbitrary bit-width.

**Work In Progress**

Tinyquant aims to
 
 - consist of <500 lines of code
 - evaluate neural network quantization methods
 - support importing ONNX models
 - support arbitrary bit-width quantization
 - be able to quantize common CNNs, transformers and graph neural networks

## Installation

```bash
pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cpu
``