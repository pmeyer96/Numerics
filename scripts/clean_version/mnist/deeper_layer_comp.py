"""Train model with multiple hidden layers on MNIST data. Afetwards generate data comparing the relative quantization 
error of an MNIST neural network using different inputs
 for deeper layer quantization"""

import os
from datetime import datetime

import numpy as np
import pandas as pd
from keras.datasets import cifar10, mnist
from keras.layers import Dense, Flatten
from keras.models import Sequential, load_model, save_model
from keras.utils import to_categorical
from quantized_net import QuantizeNeuralNet

bitsize = 5

parentpath = "/home/patric/Masterthesis/Numerics/data/deeper_layer_mnist"
now = datetime.now()
current_time = now.strftime("%d_%m_%Y_%H_%M_%S") + "_bits_" + str(bitsize)

folderpath = os.path.join(parentpath, current_time)
os.makedirs(folderpath)

np.seterr("raise")

# Load data
(X_train, y_train), (X_test, y_test) = mnist.load_data()

# Use one-hot encoding for the labels
num_classes = np.unique(y_train).shape[0]
y_train = to_categorical(y_train, num_classes)
y_test = to_categorical(y_test, num_classes)

# Training the model if it is not trained yet
# # model = Sequential()
# # model.add(Flatten(input_shape=X_train[0].shape))
# # model.add(Dense(500, activation="relu"))
# # model.add(Dense(500, activation="relu"))
# # model.add(Dense(500, activation="relu"))
# # model.add(Dense(500, activation="relu"))
# # model.add(Dense(num_classes, activation="softmax"))
# # model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
# # history = model.fit(X_train, y_train, batch_size=128, epochs=100, validation_split=0.2)

# model.save("/home/patric/Masterthesis/Numerics/data/layer_data_mnist/model.keras")


# Loading trained model
model = load_model(
    "/home/patric/Masterthesis/Numerics/data/deeper_layer_mnist/model.keras"
)

results_q = []
results_a = []

for i in [25, 50, 75, 100]:  # batchsizes of 25,50,75,100
    quantized_net = QuantizeNeuralNet(model, i, X_train, 5, True)
    quantized_net_2 = QuantizeNeuralNet(model, i, X_train, 5, False)

    quantized_net.quantize_network()
    quantized_net_2.quantize_network()
    for j in quantized_net.eval_layer["layer"]:
        list_element_q = {
            "m": i,
            "layer": j,
            "error": quantized_net.eval_layer["error"][j - 1],
        }
        results_q.append(list_element_q)
    for j in quantized_net_2.eval_layer["layer"]:
        list_element_a = {
            "m": i,
            "layer": j,
            "error": quantized_net_2.eval_layer["error"][j - 1],
        }
        results_a.append(list_element_a)


df_quantized_input = pd.DataFrame(results_q)
df_analog_input = pd.DataFrame(results_a)


# save results
df_quantized_input.to_csv(os.path.join(folderpath, "quantized_input.csv"))
df_analog_input.to_csv(os.path.join(folderpath, "analog_input.csv"))
