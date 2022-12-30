#!/usr/bin/env python
import io
import unittest
import numpy as np
import onnx.numpy_helper
import onnxruntime as ort

from models import onnx_models
from numpy_quant.model import Model
from numpy_quant.numpy_helper import conv2d


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
        input_data = rng.normal(size=(k, m)).astype(np.float32)
        output = model([input_data])[0]

        actual = output
        desired = input_data.dot(weight_data) + bias_data
        mean_diff = np.mean(np.abs(actual - desired)) / (desired.max() - desired.min())
        self.assertLessEqual(mean_diff, 0.2)

    def test_matmul(self):
        a_shape = (2, 1, 4, 3)
        b_shape = (1, 3, 3, 5)

        onnx_model = onnx_models.matmul(a_shape, b_shape)

        model = Model.from_onnx(onnx_model)
        rng = np.random.default_rng(0)
        input_a_data = rng.normal(size=a_shape).astype(np.float32)
        input_b_data = rng.normal(size=b_shape).astype(np.float32)
        output = model([input_a_data, input_b_data])[0]

        actual = output
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

        actual = model([input_data])[0]

        input_data_t = input_data.transpose((0, 2, 3, 1))
        weight_data_t = weight_data.transpose((2, 3, 1, 0))
        desired_t = conv2d(input_data_t, weight_data_t, pads, strides) + bias_data
        desired = desired_t.transpose((0, 3, 1, 2))

        # print(np.mean(np.abs(actual - desired)))
        np.testing.assert_equal(actual, desired)

    def test_expand(self):
        onnx_model = onnx_models.expand()
        onnx_bytes = io.BytesIO()
        onnx.save_model(onnx_model, onnx_bytes)
        ort_sess = ort.InferenceSession(onnx_bytes.getvalue())

        model = Model.from_onnx(onnx_model)

        input_data = np.ones((1, 1, 8), dtype=np.float32)
        shape_data = np.array([1, 1, 1], dtype=np.int64)

        desired = ort_sess.run(None, {'input': input_data,
                                      'shape': shape_data})[0]
        actual = model([input_data, shape_data])[0]

        # print(np.mean(np.abs(actual - desired)))
        np.testing.assert_equal(actual, desired)

    def test_vit_self_attention(self):
        batch_size = 1
        embeddings_size = 10
        hidden_size = 16
        num_attention_heads = 4

        onnx_model = onnx_models.vit_self_attention(batch_size, embeddings_size, hidden_size, num_attention_heads)
        onnx_bytes = io.BytesIO()
        onnx.save_model(onnx_model, onnx_bytes)
        ort_sess = ort.InferenceSession(onnx_bytes.getvalue())

        rng = np.random.default_rng()

        model = Model.from_onnx(onnx_model)
        input_data = rng.normal(size=(batch_size, embeddings_size, hidden_size)).astype(np.float32)
        actual = model([input_data])[0]

        desired = ort_sess.run(None, {'inputs': input_data})[0]

        # print(np.mean(np.abs(actual - desired)))
        np.testing.assert_allclose(actual, desired, atol=1e-6)  # TODO Why is it not exactly equal?

    def test_vit_embedding(self):
        batch_size = 1
        image_size = 16
        patch_size = 4
        hidden_size = 8
        onnx_model = onnx_models.vit_embedding(batch_size, image_size, patch_size, hidden_size)
        onnx_bytes = io.BytesIO()
        onnx.save_model(onnx_model, onnx_bytes)
        ort_sess = ort.InferenceSession(onnx_bytes.getvalue())

        rng = np.random.default_rng()

        model = Model.from_onnx(onnx_model)
        input_data = rng.normal(size=(batch_size, 3, image_size, image_size)).astype(np.float32)
        actual = model([input_data])[0]

        desired = ort_sess.run(None, {'inputs': input_data})[0]

        # print(np.mean(np.abs(actual - desired)))
        np.testing.assert_allclose(actual, desired, atol=1e-6)  # TODO Why is it not exactly equal?

    def test_vit_layer(self):
        batch_size = 1
        image_size = 16
        patch_size = 4
        intermediate_size = 22
        hidden_size = 8
        num_attention_heads = 2
        onnx_model = onnx_models.vit_layer(batch_size, image_size, patch_size, intermediate_size,
                                           hidden_size, num_attention_heads)
        onnx_bytes = io.BytesIO()
        onnx.save_model(onnx_model, onnx_bytes)
        ort_sess = ort.InferenceSession(onnx_bytes.getvalue())

        rng = np.random.default_rng()

        model = Model.from_onnx(onnx_model)
        input_data = rng.normal(size=(batch_size, (image_size // patch_size)**2 + 1, hidden_size)).astype(np.float32)
        actual = model([input_data])[0]

        desired = ort_sess.run(None, {'inputs': input_data})[0]

        # print(np.mean(np.abs(actual - desired)))
        np.testing.assert_allclose(actual, desired, atol=1e-6)  # TODO Why is it not exactly equal?

    def test_vit_pooler(self):
        batch_size = 1
        image_size = 16
        patch_size = 4
        hidden_size = 8
        onnx_model = onnx_models.vit_pooler(batch_size, image_size, patch_size, hidden_size)
        onnx_bytes = io.BytesIO()
        onnx.save_model(onnx_model, onnx_bytes)
        ort_sess = ort.InferenceSession(onnx_bytes.getvalue())

        rng = np.random.default_rng()

        model = Model.from_onnx(onnx_model)
        input_data = rng.normal(size=(batch_size, (image_size // patch_size)**2 + 1, hidden_size)).astype(np.float32)
        actual = model([input_data])[0]

        desired = ort_sess.run(None, {'inputs': input_data})[0]

        # print(np.mean(np.abs(actual - desired)))
        np.testing.assert_allclose(actual, desired, atol=1e-6)  # TODO Why is it not exactly equal?

    def test_vit(self):
        batch_size = 2
        image_size = 16
        patch_size = 4
        intermediate_size = 22
        hidden_size = 8
        num_attention_heads = 2

        onnx_model = onnx_models.vit(batch_size, image_size, patch_size,
                                     intermediate_size, hidden_size, num_attention_heads)
        onnx_bytes = io.BytesIO()
        onnx.save_model(onnx_model, onnx_bytes)
        ort_sess = ort.InferenceSession(onnx_bytes.getvalue())

        rng = np.random.default_rng()

        model = Model.from_onnx(onnx_model)
        input_data = rng.normal(size=(batch_size, 3, image_size, image_size)).astype(np.float32)
        actual = model([input_data])[0]

        desired = ort_sess.run(None, {'inputs': input_data})[0]

        # print(np.mean(np.abs(actual - desired)))
        np.testing.assert_allclose(actual, desired, atol=1e-4)  # TODO It seems like the deviation propagates
