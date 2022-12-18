#!/usr/bin/env python
import io
import unittest
import numpy as np
import onnx.numpy_helper
import onnxruntime as ort

import onnx_models
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
        rng = np.random.default_rng(0)
        input_data = rng.normal(size=(k, m))
        output = model([FTensor(input_data)])[0]

        actual = output.data
        desired = input_data.dot(weight_data) + bias_data
        mean_diff = np.mean(np.abs(actual - desired)) / (desired.max() - desired.min())
        self.assertLessEqual(mean_diff, 0.2)

    def test_matmul(self):
        a_shape = (2, 1, 4, 3)
        b_shape = (1, 3, 3, 5)

        onnx_model = onnx_models.matmul(a_shape, b_shape)

        model = Model.from_onnx(onnx_model)
        rng = np.random.default_rng(0)
        input_a_data = rng.normal(size=a_shape)
        input_b_data = rng.normal(size=b_shape)
        output = model([FTensor(input_a_data), FTensor(input_b_data)])[0]

        actual = output.data
        desired = np.matmul(input_a_data, input_b_data)
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
        input_data = rng.normal(size=(b, c, *inp_shape)).astype(np.float32)

        actual = model([FTensor(input_data)])[0].data

        input_data_t = input_data.transpose((0, 2, 3, 1))
        weight_data_t = weight_data.transpose((2, 3, 1, 0))
        desired_t = conv2d(input_data_t, weight_data_t, pads, strides) + bias_data
        desired = desired_t.transpose((0, 3, 1, 2))

        print(np.mean(np.abs(actual.data - desired)))
        np.testing.assert_equal(actual, desired)

    def test_vit_self_attention(self):
        batch_size = 1
        embeddings_size = 10
        hidden_size = 16
        num_attention_heads = 4
        onnx_model = onnx_models.vit_self_attention(batch_size, embeddings_size, hidden_size, num_attention_heads)

        rng = np.random.default_rng()

        model = Model.from_onnx(onnx_model)
        input_data = rng.normal(size=(batch_size, embeddings_size, hidden_size)).astype(np.float32)
        actual = model([FTensor(input_data)])[0].data

        onnx_bytes = io.BytesIO()
        onnx.save_model(onnx_model, onnx_bytes)
        ort_sess = ort.InferenceSession(onnx_bytes.getvalue())
        desired = ort_sess.run(None, {'inputs': input_data})[0]
        print(np.mean(np.abs(actual.data - desired)))

