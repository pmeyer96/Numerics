import numpy as np
from keras.models import Model, clone_model
from tqdm import tqdm

# def _bit_rounding(w, alphabet):
#     return alphabet[np.argmin(abs(alphabet - w))]


def _quantize_neuron(w, analog_layer_input, quantized_layer_input, alphabet):
    u = np.zeros(analog_layer_input.shape[0])
    q = np.zeros_like(w)
    for t in range(w.shape[0]):
        u += w[t] * analog_layer_input[:, t]

        norm = np.linalg.norm(quantized_layer_input[:, t], 2) ** 2
        if norm > 0:
            q_arg = np.dot(quantized_layer_input[:, t], u) / norm
        else:
            q_arg = 0
        q[t] = alphabet[np.argmin(abs(alphabet - q_arg))]
        u -= q[t] * quantized_layer_input[:, t]
    return q


class QuantizedNeuralNetworkGPFQ:
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

    def _get_intermediate_output_q(self, layer_idx: int):
        data = self.training_data[: self.batch_size, :]
        if layer_idx == 0:
            return data

        inputs = self.quantized_net.input
        outputs = self.quantized_net.layers[layer_idx - 1].output
        intermediate_layer_model = Model(inputs=inputs, outputs=outputs)
        intermediate_output = intermediate_layer_model(data)

        return intermediate_output

    def _get_intermediate_output_a(self, layer_idx: int):
        data = self.training_data[: self.batch_size, :]

        if layer_idx == 0:
            return data

        inputs = self.trained_net.input
        outputs = self.trained_net.layers[layer_idx - 1].output
        intermediate_layer_model = Model(inputs=inputs, outputs=outputs)
        intermediate_output = intermediate_layer_model(data)

        return intermediate_output

    def _quantize_layer(self, layer_idx: int):
        if layer_idx == 0:
            return

        W = self.trained_net.layers[layer_idx].get_weights()[0]
        Q = np.zeros(W.shape)
        data = self.training_data[: self.batch_size, :]

        # radius of alphabet
        rad = np.median(abs(W.flatten()))
        layer_alphabet = rad * self.alphabet
        X_tilde = self._get_intermediate_output_q(layer_idx)
        X = self._get_intermediate_output_a(layer_idx)
        for i in tqdm(range(W.shape[1])):
            Q[:, i] = _quantize_neuron(W[:, i], X, X_tilde, layer_alphabet)
        self._update_weights(layer_idx, Q)
        self._eval_layer_quant_error(Q, W, X, layer_idx)

    def quantize_network(self):
        for layer_idx, layer in enumerate(self.trained_net.layers):
            self._quantize_layer(layer_idx)

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
