"""Test numpy and torch matrix multiplication """

from time import time

import numpy as np
import torch
import plotext as plt
from tqdm import tqdm

if __name__ == '__main__':
    rng = np.random.default_rng()

    max_size = 175
    step = 25
    N = np.arange(25, max_size + 1, step)
    float32_time = np.zeros(N.shape)
    torch_float32_time = np.zeros(N.shape)
    int8_time = np.zeros(N.shape)
    torch_int8_time = np.zeros(N.shape)

    k, l = 16, 12

    for i, n in tqdm(list(enumerate(N))):
        float32_matA = rng.normal(size=(k, l, n, n)).astype(np.float32)
        float32_matB = rng.normal(size=(k, l, n, n)).astype(np.float32)
        torch_float32_matA = torch.tensor(float32_matA)
        torch_float32_matB = torch.tensor(float32_matB)
        int8_matA = rng.integers(-2**7, 2**7-1, size=(k, l, n, n), dtype=np.int8)
        int8_matB = rng.integers(-2**7, 2**7-1, size=(k, l, n, n), dtype=np.int8)
        torch_int8_matA = torch.tensor(int8_matA, dtype=torch.int64)
        torch_int8_matB = torch.tensor(int8_matB, dtype=torch.int64)

        repetitions = max_size - n + 1

        stime = time()
        np.matmul(float32_matA, float32_matB)
        float32_time[i] = time() - stime

        stime = time()
        np.matmul(int8_matA, int8_matB)
        int8_time[i] = time() - stime

        stime = time()
        torch.matmul(torch_float32_matA, torch_float32_matB)
        torch_float32_time[i] = time() - stime

        stime = time()
        out = torch.zeros((k, l, n, n), dtype=torch.int64)
        torch.matmul(torch_int8_matA, torch_int8_matB, out=out)
        torch_int8_time[i] = time() - stime

    plt.plot_size(100, 10)
    plt.axes_color('default')
    plt.canvas_color('default')
    plt.ticks_color('default')
    plt.title("Numpy")
    plt.plot(N, float32_time, label='float32')
    plt.plot(N, int8_time, label='int8')
    plt.show()
    plt.clear_figure()

    plt.plot_size(100, 10)
    plt.axes_color('default')
    plt.canvas_color('default')
    plt.ticks_color('default')
    plt.title("Torch")
    plt.plot(N, torch_float32_time, label='float32')
    plt.plot(N, torch_int8_time, label='int8')
    plt.show()
    plt.clear_figure()

