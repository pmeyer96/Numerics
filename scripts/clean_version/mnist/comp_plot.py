"""Script to generate plots comparing the relative quantization error of the OAQ, MSQ and GPFQ quantization schemes"""

import os

import matplotlib.pyplot as plt
import pandas as pd

plt.rcParams.update(
    {
        "font.size": 22,
        "axes.labelsize": 22,
        "axes.titlesize": 22,
        "axes.grid": True,
    }
)

plt.subplots_adjust(hspace=2)


"""Put path of generated data into path"""
path = "/home/patric/Masterthesis/Numerics/data/mnist_comparsion/06_01_2024_20_13_20_bits_5_c_alpha_8"
layer_1_df = pd.read_csv(os.path.join(path, "layer_1_batch_normalization.csv"))

layer_2_df = pd.read_csv(os.path.join(path, "layer_2_batch_normalization.csv"))
layer_3_df = pd.read_csv(os.path.join(path, "layer_3_batch_normalization.csv"))

fig, axs = plt.subplots(3, 2, figsize=(30, 20))

layer_1_df.plot(x="batch_size", y="Maly", ax=axs[0][0], label="OAQ", color="orange")
layer_1_df.plot(x="batch_size", y="MSQ", ax=axs[0][0], label="MSQ", color="green")
layer_1_df.plot(x="batch_size", y="GPFQ", ax=axs[0][0], label="GPFQ", color="blue")

layer_2_df.plot(x="batch_size", y="Maly", ax=axs[0][1], label="OAQ", color="orange")
layer_2_df.plot(x="batch_size", y="MSQ", ax=axs[0][1], label="MSQ", color="green")
layer_2_df.plot(x="batch_size", y="GPFQ", ax=axs[0][1], label="GPFQ", color="blue")

layer_3_df.plot(x="batch_size", y="Maly", ax=axs[1][0], label="OAQ", color="orange")
layer_3_df.plot(x="batch_size", y="MSQ", ax=axs[1][0], label="MSQ", color="green")
layer_3_df.plot(x="batch_size", y="GPFQ", ax=axs[1][0], label="GPFQ", color="blue")

layer_1_df.plot(x="batch_size", y="GPFQ", ax=axs[1][1], label="GPFQ", color="blue")
layer_1_df.plot(x="batch_size", y="Maly", ax=axs[1][1], label="OAQ", color="orange")

layer_2_df.plot(x="batch_size", y="Maly", ax=axs[2][0], label="OAQ", color="orange")
layer_2_df.plot(x="batch_size", y="GPFQ", ax=axs[2][0], label="GPFQ", color="blue")

layer_3_df.plot(x="batch_size", y="Maly", ax=axs[2][1], label="OAQ", color="orange")
layer_3_df.plot(x="batch_size", y="GPFQ", ax=axs[2][1], label="GPFQ", color="blue")

axs[0][0].set_xlabel("Batch Size")
axs[0][1].set_xlabel("Batch Size")
axs[1][0].set_xlabel("Batch Size")
axs[1][1].set_xlabel("Batch Size")
axs[2][0].set_xlabel("Batch Size")
axs[2][1].set_xlabel("Batch Size")
axs[0][0].set_ylabel("Relative Quantization Error")
axs[1][0].set_ylabel("Relative Quantization Error")
axs[2][0].set_ylabel("Relative Quantization Error")
axs[0][0].set_title("Layer 1")
axs[0][1].set_title("Layer 2")
axs[1][0].set_title("Layer 3")
axs[1][1].set_title("Layer 1")
axs[2][0].set_title("Layer 2")
axs[2][1].set_title("Layer 3")
plt.subplots_adjust(hspace=0.25)

"""Saves figure in same folder as data"""
fig.savefig(os.path.join(path, "layer_comp_pres.png"), bbox_inches="tight")
