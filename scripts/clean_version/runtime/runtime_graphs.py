"""Create plots on the basis of the generated data from runtime_tests.py."""

import matplotlib.pyplot as plt
import pandas as pd

df = pd.read_csv(
    "/home/patric/Masterthesis/Numerics/data/runtime_tests/runtime_table_32.csv"
)
plt.rcParams.update(
    {
        "font.size": 22,
        "axes.labelsize": 22,
        "axes.titlesize": 22,
        "axes.grid": True,
    }
)
runtime, r_axs = plt.subplots(1, 3, figsize=(30, 10))
# runtime.tight_layout()
runtime.suptitle("Runtime comparsion Algorithm 1 and Algorithm 2")

df[df["m"] <= 10].plot(x="m", y="runtime_m", ax=r_axs[0], label="Original")
df[(df["m"] <= 100) & (df["m"] >= 10)].plot(
    x="m", y="runtime_m", ax=r_axs[1], label="Original"
)
df[df["m"] >= 100].plot(x="m", y="runtime_m", ax=r_axs[2], label="Original")

df[df["m"] <= 10].plot(x="m", y="runtime_p", ax=r_axs[0], label="Variation")
df[(df["m"] <= 100) & (df["m"] >= 10)].plot(
    x="m", y="runtime_p", ax=r_axs[1], label="Variation"
)
df[df["m"] >= 100].plot(x="m", y="runtime_p", ax=r_axs[2], label="Variation")

r_axs[0].set_ylabel("Seconds")
r_axs[1].set_ylabel("Seconds")
r_axs[2].set_ylabel("Seconds")
kernel, k_axs = plt.subplots(1, 1, figsize=(10, 10))
df.plot(x="m", y="kernel_m", ax=k_axs, label="Original")
df.plot(
    x="m",
    y="kernel_p",
    title="Number of Kernel computations",
    ax=k_axs,
    label="Variation",
)
k_axs.set_ylabel("# of Kernel Computations")


runtime.savefig(
    "/home/patric/Masterthesis/Numerics/data/runtime_tests/runtime_presi.jpg",
    bbox_inches="tight",
)
kernel.savefig(
    "/home/patric/Masterthesis/Numerics/data/runtime_tests/kernel_presi.jpg",
    bbox_inches="tight",
)

runtime_2, r_axs_2 = plt.subplots(1, 1, figsize=(10, 10))
df.plot(x="m", y="runtime_m", ax=r_axs_2, label="Original")
df.plot(x="m", y="runtime_p", ax=r_axs_2, label="Variation")

r_axs_2.set_ylabel("Seconds")
r_axs_2.set_xlabel("m")
r_axs_2.set_title("Runtime comparsion Original and Variation")
runtime_2.savefig(
    "/home/patric/Masterthesis/Numerics/data/runtime_tests/runtime_presi_2.jpg",
    bbox_inches="tight",
)

kernel_and_runtime, k_r_axs = plt.subplots(1, 2, figsize=(20, 10))
df.plot(x="m", y="runtime_m", ax=k_r_axs[0], label="Original")
df.plot(x="m", y="runtime_p", ax=k_r_axs[0], label="Variation")
k_r_axs[0].set_ylabel("Seconds")
k_r_axs[0].set_xlabel("m")
k_r_axs[0].set_title("Runtime comparsion Original and Variation")
df.plot(x="m", y="kernel_m", ax=k_r_axs[1], label="Original")
df.plot(
    x="m",
    y="kernel_p",
    title="Number of Kernel computations",
    ax=k_r_axs[1],
    label="Variation",
)
k_axs.set_ylabel("# of Kernel Computations")

kernel_and_runtime.savefig(
    "/home/patric/Masterthesis/Numerics/data/runtime_tests/kernel_and_runtime_thesis.jpg",
    bbox_inches="tight",
)
