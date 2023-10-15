import numpy as np
from keras.models import Model, clone_model


def quantizing_weight(w, alphabet):
    return alphabet[np.argmin(abs(alphabet - w))]


class QuantizeNeuralNetMSQ:
    def __init__(self, network: Model, batch_size, training_data, bits):
        self.trained_net = network
        self.quantized_net = clone_model(network)
        self.quantized_net.set_weights(network.get_weights())
        self.bits = bits
        self.alphabet = np.linspace(-1, 1, num=int(round(2 ** (bits))))
        self.eval_layer = {"m": [], "error": [], "layer": []}
        self.batch_size = batch_size
        self.training_data = training_data

    def _quantize_layer(self, layer_idx):
        if layer_idx == 0:
            return
        W = self.trained_net.layers[layer_idx].get_weights()[0]
        Q = np.zeros(W.shape)
        c = max(W.flatten(), key=abs)
        alphabet = c * self.alphabet
        for i in range(W.shape[0]):
            for j in range(W.shape[1]):
                Q[i, j] = quantizing_weight(W[i, j], alphabet)
        data = self._get_intermediate_output_a(layer_idx)
        self._eval_layer_quant_error(Q, W, data, layer_idx)

    def _get_intermediate_output_a(self, layer_idx: int):
        data = self.training_data[: self.batch_size, :]

        if layer_idx == 0:
            return data

        inputs = self.trained_net.input
        outputs = self.trained_net.layers[layer_idx - 1].output
        intermediate_layer_model = Model(inputs=inputs, outputs=outputs)
        intermediate_output = intermediate_layer_model(data)

        return intermediate_output

    def _eval_layer_quant_error(self, quantized, unquantized, data, layer_idx):
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

    def quantize_network(self):
        for layer_idx, layer in enumerate(self.trained_net.layers):
            self._quantize_layer(layer_idx)
