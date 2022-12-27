#!/usr/bin/env python
import unittest

import numpy as np
from torch import zeros as torch_zeros
from torch import Tensor as TorchTensor
from torch import concat as torch_concat
from torch.nn.functional import conv2d as torch_conv2d

from numpy_quant.numpy_helper import conv2d


class TestInference(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        super(TestInference, self).__init__(*args, **kwargs)

    def test_conv2d(self):
        rng = np.random.default_rng(0)

        b, c, inp_shape = 2, 3, (9, 10)
        out_c = 2
        kernel_shape = (3, 2)
        pads = (0, 2, 2, 1)
        strides = (2, 1)

        input = rng.normal(size=(b, c, *inp_shape))
        weight = rng.normal(size=(out_c, c, *kernel_shape)).astype(np.float32)
        bias = rng.normal(size=out_c).astype(np.float32)

        pad_left_shape = input.shape[0:2] + pads[0:1] + input.shape[3:4]
        pad_right_shape = input.shape[0:2] + pads[2:3] + input.shape[3:4]
        torch_input = torch_concat([torch_zeros(pad_left_shape), TorchTensor(input), torch_zeros(pad_right_shape)],
                                   dim=-2)
        pad_upper_shape = input.shape[0:2] + (input.shape[2] + pads[0] + pads[2],) + pads[1:2]
        pad_lower_shape = input.shape[0:2] + (input.shape[2] + pads[0] + pads[2],) + pads[3:4]
        torch_input = torch_concat([torch_zeros(pad_upper_shape), torch_input, torch_zeros(pad_lower_shape)], dim=-1)
        desired = torch_conv2d(torch_input, TorchTensor(weight), TorchTensor(bias),
                               stride=strides).numpy()

        input_t = input.transpose((0, 2, 3, 1))
        weight_t = weight.transpose((2, 3, 1, 0))

        print("x_t", input_t.shape, "w_t", weight_t.shape, "pads", pads, "strides", strides)
        actual0_t = conv2d(input_t,
                           weight_t,
                           pads, strides)
        actual0 = actual0_t.transpose((0, 3, 1, 2))
        actual = actual0 + np.expand_dims(bias, (0, 2, 3))
        np.testing.assert_allclose(actual, desired, atol=1e-5)
