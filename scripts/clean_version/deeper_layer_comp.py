import numpy as np
import pandas as pd
from keras.datasets import cifar10, mnist
from keras.layers import Dense, Flatten
from keras.models import Sequential, save_model
from keras.utils import to_categorical
from quantized_net import QuantizeNeuralNet

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

results_q = []
results_a = []

# print(model.layers[1].get_weights())

# print(np.asarray(model.layers[0].get_weights()).shape)

quantized_net = QuantizeNeuralNet(model, batch_size, X_train, 8, True)

quantized_net_2 = QuantizeNeuralNet(model, batch_size, X_train, 8, False)

# quantized_net.eval_layer_quant_error()
# quantized_net_2.eval_layer_quant_error()
for i in range(1, 30, 3):
    quantized_net = QuantizeNeuralNet(model, i, X_train, 8, True)
    quantized_net_2 = QuantizeNeuralNet(model, i, X_train, 8, False)

    quantized_net.quantize_network()
    quantized_net_2.quantize_network()

    results_q.append(quantized_net.eval_layer)
    results_a.append(quantized_net_2.eval_layer)
    print(results_a)
    print(results_q)

df_q = pd.DataFrame(results_q)
df_nq = pd.DataFrame(results_a)

df_q.to_csv("/home/patric/Masterthesis/Numerics/data/deeper_layers_q.csv")
df_nq.to_csv("/home/patric/Masterthesis/Numerics/data/deeper_layers_nq.csv")

print(quantized_net.eval_layer)
print(quantized_net_2.eval_layer)
