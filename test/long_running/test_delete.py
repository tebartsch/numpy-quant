#!/usr/bin/env python
import unittest

import numpy as np

from models.vit import vit_image_classifier
from numpy_quant.model import Model


class TestInference(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        super(TestInference, self).__init__(*args, **kwargs)

    def test_repeated_quantization(self):
        rng = np.random.default_rng(0)
        input_data = rng.normal(size=(1, 3, 228, 228)).astype(np.float32)
        onnx_model = vit_image_classifier(batch_axis_dynamic=False)

        model = Model.from_onnx(onnx_model)
        # This will lead to OOM error if the quantized models are not properly deleted
        for i in range(100):
            qmodel = model.quantize([input_data])
