"""
Represent both floating point and fixed point tensors.
"""
from typing import Any, Optional, Union

import numpy as np


class Tensor:
    @property
    def data(self):
        return None


class FTensor(Tensor):
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

    def dot(self, other: 'FTensor'):
        return FTensor(self.data.dot(other.data))

    def exp(self):
        return FTensor(np.exp(self.data))

    def inv(self):
        return FTensor(1 / self.data)

    def relu(self):
        return FTensor((self.data > 0) * self.data)


class QTensor(Tensor):
    def __init__(self, data: np.ndarray[Any, np.int64], bit_width: int,
                 scale: np.float32,
                 zero_point: Optional[Union[np.ndarray[Any, np.int64]]] = None):
        self.bit_width = bit_width
        self.scale = scale
        self.zero_point = zero_point

        self._data = data.astype(np.int64)

    def dequantize(self):
        if self.zero_point is None:
            return FTensor(self._data * self.scale)
        else:
            return FTensor((self._data - self.zero_point) * self.scale)

    @property
    def data(self):
        return self._data

    def dot(self, other: 'QTensor'):
        s1 = self._data.shape
        l1 = len(s1)
        s2 = other._data.shape
        l2 = len(s2)

        assert self.bit_width == other.bit_width
        dot_product = self._data.astype(np.int32).dot(other._data)
        scale = self.scale * other.scale
        if self.zero_point is None and other.zero_point is None:
            return QTensor(dot_product, 4 * self.bit_width, scale=scale)
        elif self.zero_point is None:
            return QTensor(dot_product, 4 * self.bit_width, scale=scale,
                           zero_point=np.expand_dims(self._data.sum(axis=-1), tuple(range(l1-1, l1-1 + l2-1)))
                                      * other.zero_point)
        elif other.zero_point is None:
            return QTensor(dot_product, 4 * self.bit_width, scale=scale,
                           zero_point=np.expand_dims(other._data.sum(axis=max(-2, -l2)), tuple(range(l2-1)))
                                      * self.zero_point)
        else:
            zero_point = (np.expand_dims(self._data.sum(axis=-1), tuple(range(l1-1, l1-1 + l2-1)))
                            * other.zero_point
                          + np.expand_dims(other._data.sum(axis=max(-2, -l2)), tuple(range(l2-1)))
                            * self.zero_point
                          - self.zero_point * other.zero_point * s1[-1])
            return QTensor(dot_product, 4 * self.bit_width, scale=scale, zero_point=zero_point)
