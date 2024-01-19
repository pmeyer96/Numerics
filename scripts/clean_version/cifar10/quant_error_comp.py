"""Script to generate data for the comparsion of the OAQ, GPFQ and MSQ scheme on a CIFAR10 trained neural network"""

import os
from datetime import datetime

import numpy as np
import pandas as pd
from keras.datasets import cifar10
from keras.models import load_model
from quantized_net import QuantizeNeuralNet
from quantized_net_gpfq import QuantizedNeuralNetworkGPFQ
from quantized_net_msq import QuantizeNeuralNetMSQ

# folder where you want to save the data
parentpath = "/home/patric/Masterthesis/Numerics/data/CIFAR_10"

# tuples of bitsize and c_alpha to be tested
b_c_a = [(np.log2(3), 2), (2, 2), (3, 2), (4, 3)]

now = datetime.now()
current_time = now.strftime("%d_%m_%Y_%H_%M_%S")

folderpath = os.path.join(parentpath, current_time)
os.makedirs(folderpath)

(X_train, y_train), (X_test, y_test) = cifar10.load_data()

model = load_model("/home/patric/Masterthesis/Numerics/data/CIFAR_10/model.keras")

# batchsizes to be tested
batch_sizes = [100, 200]
# iterate over all combinations of b_c_a and batchsizes
for b_i, c_i in b_c_a:
    for batch_size in batch_sizes:
        results_layer_1 = []

        results_layer_2 = []
        print(batch_size)
        quantized_net_gpfq = QuantizedNeuralNetworkGPFQ(
            model, batch_size, X_train, b_i, True, c_i
        )
        quantized_net_gpfq.quantize_network()

        quantized_net_maly = QuantizeNeuralNet(model, batch_size, X_train, b_i, True)
        quantized_net_msq = QuantizeNeuralNetMSQ(model, batch_size, X_train, 1, c_i)
        quantized_net_msq.quantize_network()
        quantized_net_maly.quantize_network()
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

        layer_1_df = pd.DataFrame(results_layer_1)
        layer_2_df = pd.DataFrame(results_layer_2)

        layer_1_df.to_csv(os.path.join(folderpath, str(b_i) + "layer_1.csv"))
        layer_2_df.to_csv(os.path.join(folderpath, str(b_i) + "layer_2.csv"))
