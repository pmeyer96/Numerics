"""Script to generate plot for the comparsion of the different deeper layer quantization approaches in the OAQ scheme"""

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

# read data from csv files
df_quantized = pd.read_csv(
    "/home/patric/Masterthesis/Numerics/data/deeper_layer_mnist/07_01_2024_18_59_51_bits_5/quantized_input.csv"
)
df_analog = pd.read_csv(
    "/home/patric/Masterthesis/Numerics/data/deeper_layer_mnist/07_01_2024_18_59_51_bits_5/analog_input.csv"
)


fig, axs = plt.subplots(2, 2, figsize=(20, 20))


# Each batch size in different ax
df_quantized[df_quantized["m"] == 25].plot(
    x="layer", y="error", ax=axs[0, 0], label="Quantized"
)
df_analog[df_analog["m"] == 25].plot(x="layer", y="error", ax=axs[0, 0], label="Analog")

axs[0, 0].xaxis.set_ticks(np.arange(1, 6, 1))
axs[0, 0].set_title("Batchsize of 25")
axs[0, 0].set_ylabel("Relative Quantization Error")
axs[0, 0].set_xlabel("Layer")
df_quantized[df_quantized["m"] == 50].plot(
    x="layer", y="error", ax=axs[0, 1], label="Quantized"
)
df_analog[df_analog["m"] == 50].plot(x="layer", y="error", ax=axs[0, 1], label="Analog")
axs[0, 1].xaxis.set_ticks(np.arange(1, 6, 1))
axs[0, 1].set_title("Batchsize of 50")
axs[0, 1].set_xlabel("Layer")

df_quantized[df_quantized["m"] == 75].plot(
    x="layer", y="error", ax=axs[1, 0], label="Quantized"
)
df_analog[df_analog["m"] == 75].plot(x="layer", y="error", ax=axs[1, 0], label="Analog")
axs[1, 1].xaxis.set_ticks(np.arange(1, 6, 1))
axs[1, 1].set_title("Batchsize of 75")
axs[1, 0].set_xlabel("Layer")
axs[1, 0].set_ylabel("Relative Quantization Error")


df_quantized[df_quantized["m"] == 100].plot(
    x="layer", y="error", ax=axs[1, 1], label="Quantized"
)
df_analog[df_analog["m"] == 100].plot(
    x="layer", y="error", ax=axs[1, 1], label="Analog"
)
axs[1, 1].xaxis.set_ticks(np.arange(1, 6, 1))
axs[1, 1].set_title("Batchsize of 100")

fig.suptitle("Deeper Layer Approach Comparsion")

fig.savefig(
    "/home/patric/Masterthesis/Numerics/plots/MNIST/Deeper_Layer_Approach.png",
    bbox_inches="tight",
)
