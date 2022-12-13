import numpy as np


class Tensor:
    pass


class FTensor(Tensor):
    def __init__(self, data: np.ndarray):
        self.data = data

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
    def __init__(self, data: np.ndarray, bit_width=8):
        self.data = np.ndarray(data, dtype=np.int8)
