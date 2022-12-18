# tinyquant

Quantize ONNX-models with arbitrary bit-width.

**Work In Progress**

`tinyquant` aims to
 
 - allow evaluating neural network quantization methods for arbitrary bit-widths
 - support importing ONNX models
 - be able to quantize common
   - CNNs
   - transformers
   - graph neural networks
 - consist of less than ~~500~~ 1000 lines of code

## Installation

```bash
pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cpu
```

## Tests

```bash
python models/onnx_models.py  # Create onnx models 
python -m unittest discover -s test -p 'test_*.py'  # Run all test
```

## Notes

 - tested with ONNX Opset Version 17
 - ONNX models are expected to not use dynamic axis