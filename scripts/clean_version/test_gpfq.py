import time

import numpy as np
import pandas as pd
from keras.datasets import cifar10, mnist
from keras.layers import Dense, Flatten
from keras.models import Sequential, save_model
from keras.utils import to_categorical
from quantized_net import QuantizeNeuralNet
from quantized_net_gpfq import QuantizedNeuralNetworkGPFQ
from quantized_net_msq import QuantizeNeuralNetMSQ
from tqdm import tqdm

np.seterr("raise")

# Load data
(X_train, y_train), (X_test, y_test) = mnist.load_data()

# Use one-hot encoding for the labels
num_classes = np.unique(y_train).shape[0]
y_train = to_categorical(y_train, num_classes)
y_test = to_categorical(y_test, num_classes)

# Build the model
model = Sequential()
model.add(Flatten(input_shape=X_train[0].shape))
model.add(Dense(500, activation="relu"))
model.add(Dense(300, activation="relu"))
model.add(Dense(num_classes, activation="softmax"))
model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
history = model.fit(X_train, y_train, batch_size=128, epochs=2)

batch_size = 2

results_layer_1 = []

results_layer_2 = []

results_layer_3 = []

start = time.time()
for batch_size in tqdm(range(1, 10)):
    quantized_net_gpfq = QuantizedNeuralNetworkGPFQ(model, batch_size, X_train, 5, True)
    quantized_net_gpfq.quantize_network()

    quantized_net_maly = QuantizeNeuralNet(model, batch_size, X_train, 5, True)
    quantized_net_msq = QuantizeNeuralNetMSQ(model, batch_size, X_train, 5)
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
    results_layer_3.append(
        {
            "batch_size": batch_size,
            "Maly": quantized_net_maly.eval_layer["error"][2],
            "MSQ": quantized_net_msq.eval_layer["error"][2],
            "GPFQ": quantized_net_gpfq.eval_layer["error"][2],
        }
    )
ende = time.time()
dauer = ende - start
print(dauer)
layer_1_df = pd.DataFrame(results_layer_1)
layer_2_df = pd.DataFrame(results_layer_2)
layer_3_df = pd.DataFrame(results_layer_3)


layer_1_df.to_csv(
    "/home/patric/Masterthesis/Numerics/data/comparsion_mnist/layer_1_msq.csv"
)
layer_2_df.to_csv(
    "/home/patric/Masterthesis/Numerics/data/comparsion_mnist/layer_2_msq.csv"
)
layer_3_df.to_csv(
    "/home/patric/Masterthesis/Numerics/data/comparsion_mnist/layer_3_msq.csv"
)
