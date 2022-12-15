#!/usr/bin/env python
import unittest
import numpy as np

from tinyquant import FTensor
from tinyquant.quantize import quantize_tensor


class TestQuantization(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        super(TestQuantization, self).__init__(*args, **kwargs)

    def test_quantize_tensor(self):
        x_data = np.array([4.2, 2.1, 4.0]).T
        x = FTensor(x_data)
        qx_symmetric = quantize_tensor(x, bit_width=8, asymmetric=False)
        qx_asymmetric = quantize_tensor(x, bit_width=8, asymmetric=True)
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
            qw = quantize_tensor(w, bit_width=8, asymmetric=w_asym)
            qx = quantize_tensor(x, bit_width=8, asymmetric=x_asym)
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
            qw = quantize_tensor(w, bit_width=8, asymmetric=w_asym)
            qx = quantize_tensor(x, bit_width=8, asymmetric=x_asym)
            y = qw.dot(qx)
            print(f"calculate (w^T)x with 'w asymmetric quant: {w_asym}', 'x asymmetric quant: {x_asym}'")
            print(" - Mean Error: ", np.mean(np.abs(y.dequantize().data - w_data.dot(x_data))))
            np.testing.assert_allclose(
                actual=y.dequantize().data,
                desired=w_data.dot(x_data),
                rtol=0.5,
            )
