""" Script to crosstest the C_alpha parameter for the GPFQ quantization of an MNIST trained neural network"""

import os

import pandas as pd
from keras.datasets import mnist
from keras.models import load_model
from quantized_net_gpfq import QuantizedNeuralNetworkGPFQ

parentpath = "/home/patric/Masterthesis/Numerics/data/gpfq_cross_validate"

os.makedirs(parentpath, exist_ok=True)

bitsize = 5
c_alpha = 5

batch_size = 50


c_alpha = [7, 8, 9, 10]  # which c_alpha shall be tested

model = load_model(
    "/home/patric/Masterthesis/Numerics/data/mnist_comparsion/model_no_batch.keras"
)
(X_train, y_train), (X_test, y_test) = mnist.load_data()

results_layer_1 = []

results_layer_2 = []

results_layer_3 = []

for i in c_alpha:
    quantized_net_gpfq = QuantizedNeuralNetworkGPFQ(
        model, batch_size, X_train, bitsize, True, i
    )
    quantized_net_gpfq.quantize_network()
    results_layer_1.append(
        {
            "c_alpha": i,
            "GPFQ": quantized_net_gpfq.eval_layer["error"][0],
        }
    )
    results_layer_2.append(
        {
            "c_alpha": i,
            "GPFQ": quantized_net_gpfq.eval_layer["error"][1],
        }
    )
    results_layer_3.append(
        {
            "c_alpha": i,
            "GPFQ": quantized_net_gpfq.eval_layer["error"][2],
        }
    )

layer_1_df = pd.DataFrame(results_layer_1)
layer_2_df = pd.DataFrame(results_layer_2)
layer_3_df = pd.DataFrame(results_layer_3)

# save generated data
layer_1_df.to_csv(os.path.join(parentpath, "layer_1_batch_normalization_2.csv"))
layer_2_df.to_csv(os.path.join(parentpath, "layer_2_batch_normalization_2.csv"))
layer_3_df.to_csv(os.path.join(parentpath, "layer_3_batch_normalization_2.csv"))
