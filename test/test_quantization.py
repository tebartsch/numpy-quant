#!/usr/bin/env python
import itertools
import unittest
from time import time

import numpy as np
import onnx.numpy_helper

from extra.model_summary import summarize
from models import onnx_models
from tinyquant.model import Model
from tinyquant.quantize import quant_parameters
from tinyquant.tensor import FTensor, quantize_tensor_min_max, tensor_min_max, quantize_tensor


def shapes_broadcastable(shape_a: tuple[int, ...], shape_b: tuple[int, ...]):
    return all((m == n) or (m == 1) or (n == 1) for m, n in zip(shape_a[::-1], shape_b[::-1]))


class TestQuantization(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        super(TestQuantization, self).__init__(*args, **kwargs)

    def test_quantize_tensor(self):
        x_data = np.array([4.2, 2.1, 4.0], dtype=np.float32).T
        x = FTensor(x_data)
        qx_symmetric = quantize_tensor_min_max(x, bit_width=8, asymmetric=False)
        qx_asymmetric = quantize_tensor_min_max(x, bit_width=8, asymmetric=True)
        np.testing.assert_allclose(
            actual=qx_symmetric.dequantize().data,
            desired=x_data,
            rtol=1e-01,
        )
        np.testing.assert_allclose(
            actual=qx_asymmetric.dequantize().data,
            desired=x_data,
            rtol=1e-01,
        )

    def test_quantized_matmul(self):
        rng = np.random.default_rng(0)

        # Simple example
        w_data = np.array([
            [+1.3, +5.0, -0.3],
            [+2.1, -3.4, -0.1],
            [-0.4, +4.0, +1.7]
        ], dtype=np.float32)
        w = FTensor(w_data)

        x_data = np.array([[2.2], [2.1], [-2.0]], dtype=np.float32)
        x = FTensor(x_data)

        for w_asym, x_asym in [(False, False), (False, True), (True, False), (True, True)]:
            qw = quantize_tensor_min_max(w, bit_width=8, asymmetric=w_asym)
            qx = quantize_tensor_min_max(x, bit_width=8, asymmetric=x_asym)
            y = qw.matmul(qx)

            print(f"calculate (w^T)x with 'w asymmetric quant: {w_asym}', 'x asymmetric quant: {x_asym}'")
            print(" - Actual: ", np.matmul(w_data, x_data).flatten())
            print(" - Expected: ", y.dequantize().data.flatten())
            print(" - Mean Error: ", np.mean(np.abs(y.dequantize().data - np.matmul(w_data, x_data))))
            print()
            np.testing.assert_allclose(
                actual=y.dequantize().data,
                desired=np.matmul(w_data, x_data),
                rtol=0.5,
            )

        # Test on random data
        w_data = rng.random((2, 1, 4, 3)).astype(np.float32)
        w = FTensor(w_data)
        x_data = rng.random((1, 2, 3, 4)).astype(np.float32)
        x = FTensor(x_data)

        for w_asym, x_asym in [(False, False), (False, True), (True, False), (True, True)]:
            qw = quantize_tensor_min_max(w, bit_width=8, asymmetric=w_asym)
            qx = quantize_tensor_min_max(x, bit_width=8, asymmetric=x_asym)
            y = qw.matmul(qx)
            print(f"calculate (w^T)x with 'w asymmetric quant: {w_asym}', 'x asymmetric quant: {x_asym}'")
            print(" - Mean Error: ", np.mean(np.abs(y.dequantize().data - np.matmul(w_data, x_data))))
            np.testing.assert_allclose(
                actual=y.dequantize().data,
                desired=np.matmul(w_data, x_data),
                rtol=0.5,
            )

    def test_quantized_matmul_with_requantize(self):
        rng = np.random.default_rng(0)

        # Simple example
        w_data = np.array([
            [+1.3, +5.0, -0.3],
            [+2.1, -3.4, -0.1],
            [-0.4, +4.0, +1.7]
        ], dtype=np.float32)
        w = FTensor(w_data)

        x_data = np.array([[2.2], [2.1], [-2.0]], dtype=np.float32)
        x = FTensor(x_data)

        y = w.matmul(x)

        for w_asym, x_asym, y_asym in itertools.product([False, True], repeat=3):
            qw = quantize_tensor_min_max(w, bit_width=8, asymmetric=w_asym)
            qx = quantize_tensor_min_max(x, bit_width=8, asymmetric=x_asym)
            y_scale, y_zero_point = quant_parameters(*tensor_min_max(y), bit_width=8, asymmetric=True)
            qy = quantize_tensor(y, bit_width=8, scale=y_scale, zero_point=y_zero_point)

            out = qw.matmul(qx).requantize(8, y_scale, y_zero_point)

            w_asym_str = "asymm" if w_asym else "symm"
            x_asym_str = "asymm" if x_asym else "symm"
            y_asym_str = "asymm" if y_asym else "symm"
            print(f"calculate y=(w^T)x with w {w_asym_str} quant, x {x_asym_str} quant, y {y_asym_str} quant")
            print(" - Actual: ", out.data.flatten())
            print(" - Expected: ", qy.data.flatten())
            print(" - Mean Relative Error: ", np.mean(np.abs((out.data - qy.data) / out.data)))
            print()
            np.testing.assert_allclose(
                actual=out.data,
                desired=qy.data,
                rtol=0.5,
            )

        # Test on random data
        w_data = rng.random((2, 1, 4, 3)).astype(np.float32)
        w = FTensor(w_data)
        x_data = rng.random((1, 2, 3, 4)).astype(np.float32)
        x = FTensor(x_data)
        y = w.matmul(x)

        for w_asym, x_asym, y_asym in itertools.product([False, True], repeat=3):
            qw = quantize_tensor_min_max(w, bit_width=8, asymmetric=w_asym)
            qx = quantize_tensor_min_max(x, bit_width=8, asymmetric=x_asym)
            y_scale, y_zero_point = quant_parameters(*tensor_min_max(y), bit_width=8, asymmetric=True)
            qy = quantize_tensor(y, bit_width=8, scale=y_scale, zero_point=y_zero_point)

            out = qw.matmul(qx).requantize(8, y_scale, y_zero_point)
            w_asym_str = "asymm" if w_asym else "symm"
            x_asym_str = "asymm" if x_asym else "symm"
            y_asym_str = "asymm" if y_asym else "symm"
            print(f"calculate y=(w^T)x with w {w_asym_str} quant, x {x_asym_str} quant, y {y_asym_str} quant")
            print(" - Mean Error: ", np.mean(np.abs(out.data - qy.data)) / (qy.data.max() - qy.data.min()))
            np.testing.assert_allclose(
                actual=out.data,
                desired=qy.data,
                rtol=2,
            )

    def test_matmul(self):
        a_shape = (1, 2, 3, 4)
        b_shape = (2, 1, 4, 3)

        rng = np.random.default_rng(0)

        input_a = rng.normal(size=a_shape).astype(np.float32)
        input_b = rng.normal(size=b_shape).astype(np.float32)

        onnx_model = onnx_models.matmul(a_shape, b_shape)
        model = Model.from_onnx(onnx_model)
        qmodel = model.quantize([FTensor(input_a), FTensor(input_b)], bit_width=8)

        actual = qmodel([FTensor(input_a), FTensor(input_b)])[0].data
        desired = model([FTensor(input_a), FTensor(input_b)])[0].data

        mean_elem_l2 = np.mean(np.abs(actual - desired))
        self.assertLessEqual(mean_elem_l2, 0.2)

    def test_gemm(self):
        rng = np.random.default_rng(0)

        k, m, n = 3, 4, 2

        onnx_model = onnx_models.gemm(k, m, n, random_seed=0)
        initializers = {i.name: i for i in onnx_model.graph.initializer}
        weight_data = onnx.numpy_helper.to_array(initializers["weight"])
        bias_data = onnx.numpy_helper.to_array(initializers["bias"])

        model = Model.from_onnx(onnx_model)
        input_data = rng.normal(size=(k, m)).astype(np.float32)
        qmodel = model.quantize([FTensor(input_data)], bit_width=8)

        qoutput = qmodel([FTensor(input_data)])[0]
        actual = qoutput.data
        desired = input_data.dot(weight_data) + bias_data
        mean_elem_l2 = np.mean(np.abs(actual - desired))
        self.assertLessEqual(mean_elem_l2, 0.2)

    def test_vit_self_attention(self):
        rng = np.random.default_rng()

        batch_size = 1
        embeddings_size = 10
        hidden_size = 16
        num_attention_heads = 4

        input_data = rng.normal(size=(batch_size, embeddings_size, hidden_size)).astype(np.float32)

        onnx_model = onnx_models.vit_self_attention(batch_size, embeddings_size, hidden_size, num_attention_heads)
        model = Model.from_onnx(onnx_model)
        qmodel = model.quantize([FTensor(input_data)], bit_width=8)

        actual = qmodel([FTensor(input_data)])[0].data
        desired = model([FTensor(input_data)])[0].data

        mean_elem_l2 = np.mean(np.abs(actual - desired))
        # print(mean_elem_l2)
        self.assertLessEqual(mean_elem_l2, 0.01)

    def test_vit(self):
        rng = np.random.default_rng()

        batch_size = 8
        image_size = 96
        patch_size = 16
        intermediate_size = 156
        hidden_size = 120
        num_attention_heads = 4

        input_data = rng.normal(size=(batch_size, 3, image_size, image_size)).astype(np.float32)

        onnx_model = onnx_models.vit(batch_size, image_size, patch_size,
                                     intermediate_size, hidden_size, num_attention_heads)
        model = Model.from_onnx(onnx_model)
        qmodel = model.quantize([FTensor(input_data)], bit_width=8)

        startime = time()
        desired = model([FTensor(input_data)])[0].data
        float32_time = time() - startime
        startime = time()
        actual = qmodel([FTensor(input_data)])[0].data
        int8_time = time() - startime

        mean_elem_l2 = np.mean(np.abs(actual - desired))
        print(mean_elem_l2)
        self.assertLessEqual(mean_elem_l2, 0.1)

        print(f"ONNX Inference Time: {float32_time:.2f}s")
        print(f"Tinyquant Inference Time: {int8_time:.2f}s")

