"""Generate data to compare the quantization error of the different approaches."""

import os
from datetime import datetime

import pandas as pd
from keras.datasets import mnist
from keras.models import load_model
from quantized_net import QuantizeNeuralNet
from quantized_net_gpfq import QuantizedNeuralNetworkGPFQ
from quantized_net_msq import QuantizeNeuralNetMSQ

# where to save the data
parentpath = "/home/patric/Masterthesis/Numerics/data/mnist_comparsion"

# set parameters
bitsize = 5
c_alpha = 8

now = datetime.now()
current_time = (
    now.strftime("%d_%m_%Y_%H_%M_%S")
    + "_bits_"
    + str(bitsize)
    + "_c_alpha_"
    + str(c_alpha)
)
# create folder for every run
folderpath = os.path.join(parentpath, current_time)
os.makedirs(folderpath)

# load mnist data
(X_train, y_train), (X_test, y_test) = mnist.load_data()

# load pretrained model
model_batch = load_model(
    "/home/patric/Masterthesis/Numerics/data/mnist_comparsion/model.keras"
)

results_layer_1 = []

results_layer_2 = []

results_layer_3 = []


# Set batch sizes
batch_sizes = list(range(10, 50, 20)) + list(range(50, 201, 25))

for batch_size in batch_sizes:
    # quantize the network using different quantization approaches
    quantized_net_gpfq = QuantizedNeuralNetworkGPFQ(
        model_batch, batch_size, X_train, bitsize, True, c_alpha
    )
    quantized_net_gpfq.quantize_network()

    quantized_net_maly = QuantizeNeuralNet(
        model_batch, batch_size, X_train, bitsize, True
    )
    quantized_net_msq = QuantizeNeuralNetMSQ(
        model_batch, batch_size, X_train, 1, bitsize
    )
    quantized_net_msq.quantize_network()
    quantized_net_maly.quantize_network()
    # save the results
    results_layer_1.append(
        {
            "batch_size": batch_size,
            "Maly": quantized_net_maly.eval_layer["error"][0],
            "MSQ": quantized_net_msq.eval_layer["error"][0],
            "GPFQ": quantized_net_gpfq.eval_layer["error"][0],
        }
    )
    results_layer_2.append(
        {
            "batch_size": batch_size,
            "Maly": quantized_net_maly.eval_layer["error"][1],
            "MSQ": quantized_net_msq.eval_layer["error"][1],
            "GPFQ": quantized_net_gpfq.eval_layer["error"][1],
        }
    )
    results_layer_3.append(
        {
            "batch_size": batch_size,
            "Maly": quantized_net_maly.eval_layer["error"][2],
            "MSQ": quantized_net_msq.eval_layer["error"][2],
            "GPFQ": quantized_net_gpfq.eval_layer["error"][2],
        }
    )

layer_1_df = pd.DataFrame(results_layer_1)
layer_2_df = pd.DataFrame(results_layer_2)
layer_3_df = pd.DataFrame(results_layer_3)

# save results as csv file
layer_1_df.to_csv(os.path.join(folderpath, "layer_1_batch_normalization.csv"))
layer_2_df.to_csv(os.path.join(folderpath, "layer_2_batch_normalization.csv"))
layer_3_df.to_csv(os.path.join(folderpath, "layer_3_batch_normalization.csv"))
