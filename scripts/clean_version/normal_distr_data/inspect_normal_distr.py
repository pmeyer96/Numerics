"""Script to generate plots for the comparsion of GPFQ, OAQ and MSQ on normal distributed data"""

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

# Read data from csv files
df_layer_gpfq = pd.read_csv(
    "/home/patric/Masterthesis/Numerics/data/normal_distr_test/gpfq/gpfq_layer_thesis.csv"
)

df_layer_maly_s = pd.read_csv(
    "/home/patric/Masterthesis/Numerics/data/normal_distr_test/maly/maly_layer_s.csv"
)

df_layer_maly = pd.read_csv(
    "/home/patric/Masterthesis/Numerics/data/normal_distr_test/maly/maly_layer_thesis.csv"
)
df_layer_msq = pd.read_csv(
    "/home/patric/Masterthesis/Numerics/data/normal_distr_test/msq/msq_layer_thesis.csv"
)

# Plot data
fig_thesis, axs_thesis = plt.subplots(1, 1, figsize=(20, 10))
df_layer_gpfq.plot(x="m", y="error", ax=axs_thesis, label="GPFQ")
df_layer_maly.plot(x="m", y="error", ax=axs_thesis, label="OAQ")
df_layer_msq.plot(x="m", y="error", ax=axs_thesis, label="MSQ")

axs_thesis.set_title("Normal distributed data")
axs_thesis.set_xlabel("Batch size")
axs_thesis.set_ylabel("Relative Quantization Error")

# Save plot
fig_thesis.savefig(
    "/home/patric/Masterthesis/Numerics/plots/normal_distr_data/normal_distr_comp_thesis_2.jpg",
    bbox_inches="tight",
)
