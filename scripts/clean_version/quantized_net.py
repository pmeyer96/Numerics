from time import time

import numpy as np
from keras.models import Model, clone_model
from preprocess import preprocessing_layer


def quantizing_weight(w, alphabet):
    return alphabet[np.argmin(abs(alphabet - w))]


def quantizing_layer(layer, alphabet):
    quantized_layer = layer.copy()
    for i in range(layer.shape[1]):
        quantized_layer[:, i] = quantizing_neuron(quantized_layer[:, i], alphabet)
    return quantized_layer


def quantizing_neuron(neuron, alphabet):
    quantized_neuron = neuron.copy()
    for i in range(neuron.shape[0]):
        quantized_neuron[i] = quantizing_weight(quantized_neuron[i], alphabet)
    return quantized_neuron


class QuantizeNeuralNet:
    def __init__(self, network: Model, batch_size, training_data, bits, deeper_layer):
        """Wrapper class for tensorflow.keras.models.Model for
        quantizing the weights for Dense layers"""

        # pretrained network
        self.trained_net = network

        # Copies pretrained network structure and the weights
        self.quantized_net = clone_model(network)
        self.quantized_net.set_weights(network.get_weights())

        # Training data, used for preprocessing the layers
        self.training_data = training_data

        # How many training data points should be used for preprocessing the layers
        self.batch_size = batch_size

        # Size of alphabet and alphabet
        self.bits = bits
        self.alphabet = np.linspace(-1, 1, num=int(round(2 ** (bits))))

        # If deeper_layer is true, the quantization input is the quantized layer
        self.deeper_layer = deeper_layer

        # Error evaluation per layer
        self.eval_layer = {"m": [], "error": [], "layer": []}

    def _update_weights(self, layer_idx: int, Q: np.array):
        if self.trained_net.layers[layer_idx].use_bias:
            bias = self.trained_net.layers[layer_idx].get_weights()[1]
            self.quantized_net.layers[layer_idx].set_weights([Q, bias])
        else:
            self.quantized_net.layers[layer_idx].set_weights([Q])

    def _quantize_layer(self, layer_idx: int, input_data):
        if layer_idx == 0:
            return
        W = self.trained_net.layers[layer_idx].get_weights()[0]
        Q = np.zeros(W.shape)
        if self.deeper_layer:
            data = np.asarray(self._get_intermediate_output_q(layer_idx, input_data))
            preprocessed_W = preprocessing_layer(W, data)

        else:
            data = np.asarray(self._get_intermediate_output_a(layer_idx, input_data))
            preprocessed_W = preprocessing_layer(W, data)
        c = max(W.flatten(), key=abs)
        # # print(c)
        # # print(preprocessed_W)
        # full = np.full(W.shape, np.abs(c))
        # print(np.abs(preprocessed_W) - full)
        # print("C entries", np.count_nonzero(np.isclose(np.abs(preprocessed_W), full)))
        alphabet = c * self.alphabet
        Q = quantizing_layer(preprocessed_W, alphabet)
        self._update_weights(layer_idx, Q)
        self._eval_layer_quant_error(Q, W, data, layer_idx)

    def _eval_layer_quant_error(self, quantized, unquantized, data, layer_idx):
        """Evaluates the quantization error for every layer of a NN on a given data set"""

        numerator = np.linalg.norm(
            np.matmul(data, unquantized)
            - np.matmul(
                data,
                quantized,
            ),
            "fro",
        )
        denominator = np.linalg.norm(
            np.matmul(data, quantized),
            "fro",
        )
        self.eval_layer["error"].append(numerator / denominator)
        self.eval_layer["m"].append(self.batch_size)
        self.eval_layer["layer"].append(layer_idx)
        return

    def _get_intermediate_output_q(self, layer_idx: int, data):
        if layer_idx == 0:
            return data

        inputs = self.quantized_net.input
        outputs = self.quantized_net.layers[layer_idx - 1].output
        intermediate_layer_model = Model(inputs=inputs, outputs=outputs)
        intermediate_output = intermediate_layer_model(data)

        return intermediate_output

    def _get_intermediate_output_a(self, layer_idx: int, data):
        if layer_idx == 0:
            return data

        inputs = self.trained_net.input
        outputs = self.trained_net.layers[layer_idx - 1].output
        intermediate_layer_model = Model(inputs=inputs, outputs=outputs)
        intermediate_output = intermediate_layer_model(data)

        return intermediate_output

    # def _compare_quant_deeper_layers(self):
    #     data = self.training_data[: self.batch_size, :]
    #     for layer_idx, layer in enumerate(self.trained_net.layers):
    #         self._quantize_layer(layer_idx, data, quantized=False)
    #         self._eval_layer_quant_error(data, layer_idx)
    #     return

    # def _get_layer_data_generator(self, layer_idx: int):
    #     """Gets input data for the layer at a given index"""
    #     mini_batch =self.training_data[:self.batch_size,:]

    #     if layer_idx == 0:
    #         return np.reshape(mini_batch

    #     layer = self.trained_net.layers[layer_idx]

    #     if layer_idx > 0:
    #         prev_quant_model = Model(inputs= self.quantized_net.input, outputs =  self.quantized_net.layers[layer_idx].output)
    #         intermediate_output = prev_quant_model()

    def quantize_network(self):
        data = self.training_data[: self.batch_size, :]

        for layer_idx, layer in enumerate(self.trained_net.layers):
            self._quantize_layer(layer_idx, data)
