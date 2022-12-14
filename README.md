# tinyquant

Evaluate ONNX-model quantization with arbitrary bit-width.

**Work In Progress**

`tinyquant` aims to
 
 - evaluate neural network quantization methods for arbitrary bit-widths
 - support importing ONNX models
 - be able to quantize common
   - CNNs
   - transformers
   - graph neural networks
 - consist of <500 lines of code

## Installation

```bash
pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cpu
```