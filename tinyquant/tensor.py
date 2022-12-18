"""
Represent both floating point and fixed point tensors.
"""
from typing import Any, Optional, Union

import numpy as np

from tinyquant import numpy_helper
from tinyquant.quantize import quant_parameters, quantize


class FTensor:
    def __init__(self, data: np.ndarray):
        self._data = data

    @property
    def data(self):
        return self._data

    @property
    def shape(self):
        return self.data.shape

    @property
    def T(self):
        return FTensor(self.data.T)

    def reshape(self, shape: tuple[int, ...]):
        return FTensor(self.data.reshape(shape))

    def transpose(self, *axes):
        return FTensor(self.data.transpose(*axes))

    def __neg__(self):
        return FTensor(-self.data)

    def __add__(self, other: 'FTensor'):
        if isinstance(other, FTensor):
            return FTensor(self.data + other.data)
        if isinstance(other, float):
            return FTensor(self.data + other)
        else:
            raise ValueError(f"Value of type {type(other)} cannot be added")

    def __radd__(self, other):
        return self.__add__(other)

    def matmul(self, other: 'FTensor'):
        return FTensor(np.matmul(self.data, other.data))

    def exp(self):
        return FTensor(np.exp(self.data))

    def inv(self):
        return FTensor(1 / self.data)

    def relu(self):
        return FTensor((self.data > 0) * self.data)

    def sigmoid(self):
        return (1.0 + (-self).exp()).inv()


class QTensor:
    def __init__(self, data: np.ndarray[Any, np.int64], bit_width: int,
                 scale: np.float32,
                 zero_point: Optional[Union[np.ndarray[Any, np.int64]]] = None):
        self.bit_width = bit_width
        self.scale = scale
        self.zero_point = zero_point
        self._data = data.astype(np.int64)

    @property
    def shape(self):
        return self.data.shape

    @property
    def T(self):
        zero_point_T = None if self.zero_point is None else self.zero_point.T
        return QTensor(self.data.T, self.bit_width, self.scale, zero_point_T)

    def reshape(self, shape: tuple[int, ...]):
        return QTensor(self.data.reshape(shape), self.bit_width, self.scale, self.zero_point)

    def transpose(self, *axes):
        return QTensor(self.data.transpose(*axes))

    def __add__(self, other: 'QTensor'):
        if isinstance(other, QTensor):
            return QTensor(self.data + other.data, self.bit_width, self.scale, self.zero_point)

    def dequantize(self):
        if self.zero_point is None:
            return FTensor(self._data * self.scale)
        else:
            return FTensor((self._data - self.zero_point) * self.scale)

    def requantize(self, bit_width: int, scale: np.float32, zero_point: np.int64):
        min_qval, max_qval = -2.0 ** (bit_width - 1), 2.0 ** (bit_width - 1) - 1.0
        dequant = self.dequantize().data
        qdata = np.clip(np.rint(zero_point + 1 / scale * dequant), min_qval, max_qval).astype(np.int64)
        return QTensor(qdata, bit_width, scale, zero_point)

    @property
    def data(self):
        return self._data

    def matmul(self, other: 'QTensor'):
        s1 = self._data.shape
        assert self.bit_width == other.bit_width, f"{self.bit_width} != {other.bit_width}"
        matmul = np.matmul(self._data.astype(np.int64), other._data)
        scale = self.scale * other.scale
        if self.zero_point is None and other.zero_point is None:
            return QTensor(matmul, 4 * self.bit_width, scale=scale)
        elif self.zero_point is None:
            return QTensor(matmul, 4 * self.bit_width, scale=scale,
                           zero_point=self._data.sum(axis=-1, keepdims=True) * other.zero_point)
        elif other.zero_point is None:
            return QTensor(matmul, 4 * self.bit_width, scale=scale,
                           zero_point=other._data.sum(axis=-2, keepdims=True) * self.zero_point)
        else:
            zero_point = (self._data.sum(axis=-1, keepdims=True) * other.zero_point
                          + other._data.sum(axis=-2, keepdims=True) * self.zero_point
                          - self.zero_point * other.zero_point * s1[-1])
            return QTensor(matmul, 4 * self.bit_width, scale=scale, zero_point=zero_point)

    def relu(self):
        relu_data = self.data.copy()
        relu_data[relu_data < self.zero_point] = self.zero_point
        return QTensor(relu_data, self.bit_width, self.scale, self.zero_point)

    def sigmoid(self):
        dequant_tensor = self.dequantize()
        activations = (1.0 + (-dequant_tensor).exp()).inv()
        qactivations = quantize(activations.data, self.bit_width, self.scale, self.zero_point)
        return QTensor(qactivations, self.bit_width, self.scale, self.zero_point)


Tensor = Union[FTensor, QTensor]


def quantize_tensor(tensor: Tensor, bit_width: int, scale: np.float64, zero_point: np.int64 | None):
    qdata = quantize(tensor.data, bit_width, scale, zero_point)
    return QTensor(qdata, bit_width, scale=scale, zero_point=zero_point)


def tensor_min_max(tensor: Tensor):
    min_val = np.minimum(tensor.data.min(), 0.0)
    max_val = np.maximum(tensor.data.max(), 0.0)
    return min_val, max_val


def quantize_tensor_min_max(tensor: Tensor, bit_width: int, asymmetric: bool):
    min_val, max_val = tensor_min_max(tensor)
    scale, zero_point = quant_parameters(min_val, max_val, bit_width, asymmetric)
    return quantize_tensor(tensor, bit_width, scale, zero_point)


def fconv2d(x: FTensor, w: FTensor, b: FTensor,
            pads: (int, int, int, int), strides: (int, int)):
    x_data_t = x.data.transpose((0, 2, 3, 1))
    w_data_t = w.data.transpose((2, 3, 1, 0))
    y0_data_t = numpy_helper.conv2d(x_data_t, w_data_t, pads, strides)
    y0_data = y0_data_t.transpose((0, 3, 1, 2))
    b_data = b.data
    y_data = y0_data + np.expand_dims(b_data, (0, 2, 3))
    return FTensor(y_data)
