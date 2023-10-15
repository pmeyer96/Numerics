import pandas as pd

df = pd.read_csv(
    "/home/patric/Masterthesis/Numerics/data/comparsion_mnist/layer_1_msq.csv"
)
df_2 = pd.read_csv(
    "/home/patric/Masterthesis/Numerics/data/comparsion_mnist/layer_2_msq.csv"
)
df_3 = pd.read_csv(
    "/home/patric/Masterthesis/Numerics/data/comparsion_mnist/layer_3_msq.csv"
)


print(df)
print(df_2)
print(df_3)
