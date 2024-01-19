""" Plot the quantization error for the CIFAR-10 dataset. """ ""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

plt.rcParams.update(
    {
        "font.size": 22,
        "axes.labelsize": 22,
        "axes.titlesize": 22,
        "axes.grid": True,
    }
)

fig, axs = plt.subplots(1, 2, figsize=(20, 10))

# Read data from csv files, need to copy location of csv files
layer_1_df = pd.read_csv(
    "/home/patric/Masterthesis/Numerics/data/CIFAR_10/05_01_2024_17_18_17/1.584962500721156layer_1.csv"
)
layer_2_df = pd.read_csv(
    "/home/patric/Masterthesis/Numerics/data/CIFAR_10/05_01_2024_17_18_17/1.584962500721156layer_2.csv"
)

layer_1_df.plot(x="Bits", y="Maly", ax=axs[0], label="OAQ", color="orange")
layer_1_df.plot(x="Bits", y="MSQ", ax=axs[0], label="MSQ", color="green")
layer_1_df.plot(x="Bits", y="GPFQ", ax=axs[0], label="GPFQ", color="blue")

axs[0].xaxis.set_ticks(np.arange(1, 5, 1))
axs[1].xaxis.set_ticks(np.arange(1, 5, 1))

layer_2_df.plot(x="Bits", y="Maly", ax=axs[1], label="OAQ", color="orange")
layer_2_df.plot(x="Bits", y="MSQ", ax=axs[1], label="MSQ", color="green")
layer_2_df.plot(x="Bits", y="GPFQ", ax=axs[1], label="GPFQ", color="blue")

axs[0].set_xlabel("Number of bits")
axs[1].set_xlabel("Number of bits")

axs[0].set_ylabel("Relative Quantization Error")
axs[1].set_ylabel("Relative Quantization Error")

axs[0].set_title("Dense Layer 128")
axs[1].set_title("Dense Layer Output")
fig.savefig(
    "/home/patric/Masterthesis/Numerics/data/CIFAR_10/05_01_2024_17_18_17/plot.png",
    bbox_inches="tight",
)
