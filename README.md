# numpy-quant

Quantize ONNX-models with arbitrary bit-width.

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

## Tests

```bash
python models/onnx_models.py  # Create onnx models
python -m unittest discover -s test -p 'test_*.py'  # Run fast test
python -m unittest discover -s test/long_running -p 'test_*.py'  # Run long-running tests (> 1 minute in total)
```

## Notes

 - tested with ONNX Opset Version 17
 - ONNX models are expected to not use dynamic axis