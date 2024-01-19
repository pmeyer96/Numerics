"""Script to generate data for the comparsion of GPFQ, OAQ and MSQ on normal distributed data"""

import os
from time import time

import numpy as np
import pandas as pd
from normal_distr_data_class import NormalDistributedData

# Layer and data generation
A = np.random.normal(0, 1, (200, 2000))
W = np.random.normal(0, 1, (2000, 30))

# change to float32
A_32bit = np.zeros(A.shape, dtype=np.float32)
W_32bit = np.zeros(W.shape, dtype=np.float32)
A_32bit = A
W_32bit = W
# Batchsizes to be tested
batch_sizes = (
    list(range(1, 10, 1)) + list(range(10, 100, 5)) + list(range(100, 201, 10))
)
data = NormalDistributedData(A_32bit, W_32bit, batch_sizes, 5.333333333333333, 5)


# Quantization
data.quantize_layer_gpfq()
start = time()
data.quantize_layer_maly()
end = time()
print("Maly done, took", end - start, "seconds")
start = time()
data.quantize_layer_msq()
end = time()
print("MSQ done, took", end - start, "seconds")

df_maly_n = pd.DataFrame(data.eval_neuron_maly)
df_maly_l = pd.DataFrame(data.eval_layer_maly)

df_gpfq_n = pd.DataFrame(data.eval_neuron_gpfq)
df_gpfq_l = pd.DataFrame(data.eval_layer_gpfq)

df_msq_l = pd.DataFrame(data.eval_layer_msq)
df_msq_n = pd.DataFrame(data.eval_neuron_msq)

# Save data

path_maly = "/home/patric/Masterthesis/Numerics/data/normal_distr_test/maly"
df_maly_n.to_csv(os.path.join(path_maly, "maly_neuron_thesis.csv"))

df_maly_l.to_csv(os.path.join(path_maly, "maly_layer_thesis.csv"))

path_gpfq = "/home/patric/Masterthesis/Numerics/data/normal_distr_test/gpfq"
df_gpfq_n.to_csv(os.path.join(path_gpfq, "gpfq_neuron_thesis.csv"))
df_gpfq_l.to_csv(os.path.join(path_gpfq, "gpfq_layer_thesis.csv"))

path_msq = "/home/patric/Masterthesis/Numerics/data/normal_distr_test/msq"

df_msq_l.to_csv(os.path.join(path_msq, "msq_layer_thesis.csv"))
df_msq_n.to_csv(os.path.join(path_msq, "msq_neuron_thesis.csv"))
