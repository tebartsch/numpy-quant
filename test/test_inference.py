#!/usr/bin/env python
import textwrap
import unittest
import numpy as np
import onnx.numpy_helper

import onnx_models
from extra.model_summary import summarize
from tinyquant.model import Model
from tinyquant.numpy_helper import conv2d
from tinyquant.tensor import FTensor


class TestInference(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        super(TestInference, self).__init__(*args, **kwargs)

    def test_gemm(self):
        k, m, n = 3, 4, 2

        onnx_model = onnx_models.gemm(k, m, n, random_seed=0)
        initializers = {i.name: i for i in onnx_model.graph.initializer}
        weight_data = onnx.numpy_helper.to_array(initializers["weight"])
        bias_data = onnx.numpy_helper.to_array(initializers["bias"])

        model = Model.from_onnx(onnx_model)

        self.assertEqual(
            summarize(model), textwrap.dedent("""\
            =====+============+============
            Node | Inputs     | Outputs    
            =====+============+============
            Gemm | input_name | output_name
                 | weight     |            
                 | bias       |            
            -----+------------+------------
            """)
        )

        rng = np.random.default_rng(0)
        input_data = rng.normal(size=(k, m))
        output = model([FTensor(input_data)])[0]

        actual = output.data
        desired = input_data.dot(weight_data) + bias_data
        mean_diff = np.mean(np.abs(actual - desired)) / (desired.max() - desired.min())
        self.assertLessEqual(mean_diff, 0.2)

    def test_conv(self):
        b, c, inp_shape = 2, 3, (9, 10)
        pads = (0, 2, 2, 1)
        strides = (2, 1)
        onnx_model = onnx_models.conv(
            b=b,
            c=c,
            inp_shape=inp_shape,
            out_c=2,
            kernel_shape=(3, 2),
            pads=pads,
            strides=strides,
            random_seed=0
        )
        initializers = {i.name: i for i in onnx_model.graph.initializer}
        weight_data = onnx.numpy_helper.to_array(initializers["weight"])
        bias_data = onnx.numpy_helper.to_array(initializers["bias"])

        model = Model.from_onnx(onnx_model)

        rng = np.random.default_rng(0)
        input_data = rng.normal(size=(b, c, *inp_shape))

        actual = model([FTensor(input_data)])[0].data

        input_data_t = input_data.transpose((0, 2, 3, 1))
        weight_data_t = weight_data.transpose((2, 3, 1, 0))
        desired_t = conv2d(input_data_t, weight_data_t, pads, strides) + bias_data
        desired = desired_t.transpose((0, 3, 1, 2))

        np.testing.assert_allclose(actual, desired)
