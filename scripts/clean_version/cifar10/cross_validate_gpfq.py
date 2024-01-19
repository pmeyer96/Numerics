""" Script to crosstest the C_alpha parameter for the GPFQ quantization of an CIFAR10 trained neural network"""


import os

import pandas as pd
from keras.datasets import cifar10
from keras.models import load_model
from quantized_net_gpfq import QuantizedNeuralNetworkGPFQ

# Folder where model is saved
parentpath = "/home/patric/Masterthesis/Numerics/data/CIFAR_10/cross_validation"
os.makedirs(parentpath, exist_ok=True)
(X_train, y_train), (X_test, y_test) = cifar10.load_data()

# C_alpha values to be tested
c_alpha = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
# Bitsizes to be tested
bitsize = [3, 4]
# Batchsize for quantization
batchsize = 200
# Load pretrained model
model = load_model("/home/patric/Masterthesis/Numerics/data/CIFAR_10/model.keras")

comparision = []
for b_i in bitsize:
    print(b_i)
    for c_i in c_alpha:
        print(c_i)
        quantized_net_gpfq = QuantizedNeuralNetworkGPFQ(
            model, batchsize, X_train, b_i, True, c_i
        )
        quantized_net_gpfq.quantize_network()
        comparision.append(
            {
                "bits": b_i,
                "c_alpha": c_i,
                "layer_1": quantized_net_gpfq.eval_layer["error"][0],
                "layer_2": quantized_net_gpfq.eval_layer["error"][1],
            }
        )

comparsion_df = pd.DataFrame(comparision)

# save as csv file
comparsion_df.to_csv(
    "/home/patric/Masterthesis/Numerics/data/CIFAR_10/gpfq_cross_validation/table_2.csv"
)
