"""Generate data for runtime tests."""

from time import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from preprocess import preprocessing_layer
from tqdm import tqdm

A = np.random.rand(200, 2000)
W = np.random.rand(2000, 10)

A_32bit = np.zeros(A.shape, dtype=np.float32)
W_32bit = np.zeros(W.shape, dtype=np.float32)

A_32bit = A
W_32bit = W
runtime_list = []


def test_runtime(A, W, lower, upper, stepsize):
    for i in tqdm(range(lower, upper, stepsize)):
        A_0 = A[0:i, :]
        start_p = time()
        _, kernel_calculations_p = preprocessing_layer(W, A_0)
        finish_p = time()
        start_m = time()
        _, kernel_calculations_m = preprocessing_layer(W, A_0, False)
        finish_m = time()
        runtime_list.append(
            {
                "m": i,
                "runtime_p": finish_p - start_p,
                "runtime_m": finish_m - start_m,
                "kernel_p": kernel_calculations_p,
                "kernel_m": kernel_calculations_m,
            }
        )


test_runtime(A_32bit, W_32bit, 1, 10, 1)
test_runtime(A_32bit, W_32bit, 10, 100, 5)
test_runtime(A_32bit, W_32bit, 100, 201, 10)
print(runtime_list)
fig1, axs1 = plt.subplots(1, 1)
fig2, axs2 = plt.subplots(1, 1)
df = pd.DataFrame(runtime_list)
df.plot(x="m", y="runtime_p", title="Runtime comparsion", ax=axs1, color="blue")
df.plot(x="m", y="runtime_m", ax=axs1, color="red")
df.plot(
    x="m", y="kernel_p", title="Number of Kernel computations", ax=axs2, color="blue"
)
df.plot(x="m", y="kernel_m", ax=axs2, color="red")


fig1.savefig(
    "/home/patric/Masterthesis/Numerics/data/runtime_tests/runtime_plot_32.jpg"
)
fig2.savefig("/home/patric/Masterthesis/Numerics/data/runtime_tests/kernel_calc_32.jpg")
df.to_csv("/home/patric/Masterthesis/Numerics/data/runtime_tests/runtime_table_32.csv")
np.save("/home/patric/Masterthesis/Numerics/data/runtime_tests/A.npy", A)
np.save("/home/patric/Masterthesis/Numerics/data/runtime_tests/W.npy", W)
