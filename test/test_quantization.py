#!/usr/bin/env python
import itertools
import unittest
import numpy as np
import onnx
import onnx.shape_inference

from tinyquant.model import Model
from tinyquant.quantize import quant_parameters
from tinyquant.tensor import FTensor, quantize_tensor_min_max, tensor_min_max, quantize_tensor


class TestQuantization(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        super(TestQuantization, self).__init__(*args, **kwargs)

    def test_quantize_tensor(self):
        x_data = np.array([4.2, 2.1, 4.0]).T
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

    def test_quantized_dot(self):
        # Simple example
        w_data = np.array([
            [+1.3, +5.0, -0.3],
            [+2.1, -3.4, -0.1],
            [-0.4, +4.0, +1.7]
        ])
        w = FTensor(w_data)

        x_data = np.array([2.2, 2.1, -2.0]).T
        x = FTensor(x_data)

        for w_asym, x_asym in [(False, False), (False, True), (True, False), (True, True)]:
            qw = quantize_tensor_min_max(w, bit_width=8, asymmetric=w_asym)
            qx = quantize_tensor_min_max(x, bit_width=8, asymmetric=x_asym)
            y = qw.dot(qx)

            print(f"calculate (w^T)x with 'w asymmetric quant: {w_asym}', 'x asymmetric quant: {x_asym}'")
            print(" - Actual: ", w_data.dot(x_data))
            print(" - Expected: ", y.dequantize().data)
            print(" - Mean Error: ", np.mean(np.abs(y.dequantize().data - w_data.dot(x_data))))
            print()
            np.testing.assert_allclose(
                actual=y.dequantize().data,
                desired=w_data.dot(x_data),
                rtol=0.5,
            )

        # Test on random data
        w_data = np.random.random((2, 1, 4, 3))
        w = FTensor(w_data)
        x_data = np.random.random((1, 2, 3, 4))
        x = FTensor(x_data)

        for w_asym, x_asym in [(False, False), (False, True), (True, False), (True, True)]:
            qw = quantize_tensor_min_max(w, bit_width=8, asymmetric=w_asym)
            qx = quantize_tensor_min_max(x, bit_width=8, asymmetric=x_asym)
            y = qw.dot(qx)
            print(f"calculate (w^T)x with 'w asymmetric quant: {w_asym}', 'x asymmetric quant: {x_asym}'")
            print(" - Mean Error: ", np.mean(np.abs(y.dequantize().data - w_data.dot(x_data))))
            np.testing.assert_allclose(
                actual=y.dequantize().data,
                desired=w_data.dot(x_data),
                rtol=0.5,
            )

    def test_quantized_dot_with_requantize(self):
        # Simple example
        w_data = np.array([
            [+1.3, +5.0, -0.3],
            [+2.1, -3.4, -0.1],
            [-0.4, +4.0, +1.7]
        ])
        w = FTensor(w_data)

        x_data = np.array([2.2, 2.1, -2.0]).T
        x = FTensor(x_data)

        y = w.dot(x)

        for w_asym, x_asym, y_asym in itertools.product([False, True], repeat=3):
            qw = quantize_tensor_min_max(w, bit_width=8, asymmetric=w_asym)
            qx = quantize_tensor_min_max(x, bit_width=8, asymmetric=x_asym)
            y_scale, y_zero_point = quant_parameters(*tensor_min_max(y), bit_width=8, asymmetric=True)
            qy = quantize_tensor(y, bit_width=8, scale=y_scale, zero_point=y_zero_point)

            out = qw.dot(qx).requantize(8, y_scale, y_zero_point)

            w_asym_str = "asymm" if w_asym else "symm"
            x_asym_str = "asymm" if x_asym else "symm"
            y_asym_str = "asymm" if y_asym else "symm"
            print(f"calculate y=(w^T)x with w {w_asym_str} quant, x {x_asym_str} quant, y {y_asym_str} quant")
            print(" - Actual: ", out.data)
            print(" - Expected: ", qy.data)
            print(" - Mean Relative Error: ", np.mean(np.abs((out.data - qy.data) / out.data)))
            print()
            np.testing.assert_allclose(
                actual=out.data,
                desired=qy.data,
                rtol=0.5,
            )

        # Test on random data
        w_data = np.random.random((2, 1, 4, 3))
        w = FTensor(w_data)
        x_data = np.random.random((1, 2, 3, 4))
        x = FTensor(x_data)
        y = w.dot(x)

        for w_asym, x_asym, y_asym in itertools.product([False, True], repeat=3):
            qw = quantize_tensor_min_max(w, bit_width=8, asymmetric=w_asym)
            qx = quantize_tensor_min_max(x, bit_width=8, asymmetric=x_asym)
            y_scale, y_zero_point = quant_parameters(*tensor_min_max(y), bit_width=8, asymmetric=True)
            qy = quantize_tensor(y, bit_width=8, scale=y_scale, zero_point=y_zero_point)

            out = qw.dot(qx).requantize(8, y_scale, y_zero_point)
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

    def test_gemm(self):
        k, m, n = 3, 4, 2

        input_name, output_name = "input_name", "output_name"
        input = onnx.helper.make_tensor_value_info(input_name, onnx.TensorProto.FLOAT, [k, m])
        output = onnx.helper.make_tensor_value_info(output_name, onnx.TensorProto.FLOAT, [k, n])

        weight_data = np.random.normal(size=(m, n)).astype(np.float32)
        bias_data = np.random.normal(size=n).astype(np.float32)

        weight_name = "weight"
        weight = onnx.helper.make_tensor(name=weight_name,
                                         data_type=onnx.TensorProto.FLOAT,
                                         dims=weight_data.shape,
                                         vals=weight_data.flatten().tolist())

        bias_name = "bias"
        bias = onnx.helper.make_tensor(name=bias_name,
                                       data_type=onnx.TensorProto.FLOAT,
                                       dims=bias_data.shape,
                                       vals=bias_data.flatten().tolist())

        node = onnx.helper.make_node(
            name="Gemm",
            op_type="Gemm",
            inputs=[input_name, weight_name, bias_name],
            outputs=[output_name],
        )

        graph_def = onnx.helper.make_graph(
            nodes=[node],
            name="Gemm",
            inputs=[input],
            outputs=[output],
            initializer=[weight, bias],
        )

        onnx_model = onnx.helper.make_model(graph_def, producer_name="tinyquant-test")
        onnx_model.opset_import[0].version = 13

        onnx_model = onnx.shape_inference.infer_shapes(onnx_model)

        onnx.checker.check_model(onnx_model)
        # onnx.save(model_def, "Gemm.onnx")

        model = Model.from_onnx(onnx_model)
        input_data = np.random.normal(size=(k, m))
        qmodel = model.quantize_model([FTensor(input_data)], bit_width=8)
        qoutput = qmodel([FTensor(input_data)])[0]

        actual = qoutput.data
        desired = input_data.dot(weight_data) + bias_data
        mean_diff = np.mean(np.abs(actual - desired)) / (desired.max() - desired.min())
        self.assertLessEqual(mean_diff, 0.2)
