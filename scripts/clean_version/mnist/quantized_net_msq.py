"""Quantize a neural network using MSQ"""

import numpy as np
from keras.models import Model, clone_model


def quantizing_weight(w, alphabet):
    """Quantize a single weight.
    Parameters
    ----------
    w : np.array
        The weight of the neuron.
    alphabet : np.array
        The alphabet used for quantization.
    Returns
    -------
    q : np.array
        The quantized weight.
    """
    return alphabet[np.argmin(abs(alphabet - w))]


class QuantizeNeuralNetMSQ:
    def __init__(self, network: Model, batch_size, training_data, c_alpha, bits):
        """Wrapper class for tensorflow.keras.models.Model for
        quantizing the weights for Dense layers using MSQ.
        Parameters
        ----------
        network : tensorflow.keras.models.Model
            The pretrained neural network
        batch_size : int
            The batch size used for quantizing the layers
        training_data : np.array
            The training data used for quantizing the layers
        bits : int
            The number of bits used for quantization
        """
        self.trained_net = network
        self.quantized_net = clone_model(network)
        self.quantized_net.set_weights(network.get_weights())
        self.bits = bits
        self.alphabet = np.linspace(-c_alpha, c_alpha, num=int(round(2 ** (bits))))
        self.eval_layer = {"m": [], "error": [], "layer": []}
        self.batch_size = batch_size
        self.training_data = training_data
        # self.c_alpha = c_alpha # dont scale with c_alpha in msq case in these experiments

    def _update_weights(self, layer_idx: int, Q: np.array):
        """Updates the weights of the quantized neural network given a layer index and quantized weights.

        Parameters
        ----------
        layer_idx : int
            The layer index
        Q : np.array
            The quantized weights"""
        if self.trained_net.layers[layer_idx].use_bias:
            bias = self.trained_net.layers[layer_idx].get_weights()[1]
            self.quantized_net.layers[layer_idx].set_weights([Q, bias])
        else:
            self.quantized_net.layers[layer_idx].set_weights([Q])

    def _quantize_layer(self, layer_idx):
        """Quantizes a layer of the neural network.

        Parameters
        ----------
        layer_idx : int
            The layer index
        """
        if layer_idx == 0:
            return
        W = self.trained_net.layers[layer_idx].get_weights()[0]
        Q = np.zeros(W.shape)
        c = max(W.flatten(), key=abs)
        alphabet = c * self.alphabet
        for i in range(W.shape[0]):
            for j in range(W.shape[1]):
                Q[i, j] = quantizing_weight(W[i, j], alphabet)
        self._update_weights(layer_idx, Q)
        data_a = self._get_intermediate_output_a(layer_idx)
        data_q = self._get_intermediate_output_q(layer_idx)
        self._eval_layer_quant_error(Q, W, data_a, data_q, layer_idx)

    def _get_intermediate_output_a(self, layer_idx: int):
        """Generate the activation of the unquantized layer.

        Parameters
        ----------
        layer_idx : int
            The layer index
        Returns
        -------
        intermediate_output : np.array
            The activation of the unquantized layer"""
        data = self.training_data[: self.batch_size, :]

        if layer_idx == 0:
            return data

        inputs = self.trained_net.input
        outputs = self.trained_net.layers[layer_idx - 1].output
        intermediate_layer_model = Model(inputs=inputs, outputs=outputs)
        intermediate_output = intermediate_layer_model(data)

        return intermediate_output

    def _get_intermediate_output_q(self, layer_idx: int):
        """Generate the activation of the quantized layer.

        Parameters
        ----------
        layer_idx : int
            The layer index
        Returns
        -------
        intermediate_output : np.array
            The activation of the quantized layer"""
        data = self.training_data[: self.batch_size, :]
        if layer_idx == 0:
            return data

        inputs = self.quantized_net.input
        outputs = self.quantized_net.layers[layer_idx - 1].output
        intermediate_layer_model = Model(inputs=inputs, outputs=outputs)
        intermediate_output = intermediate_layer_model(data)

        return intermediate_output

    def _eval_layer_quant_error(
        self, quantized, unquantized, data_a, data_q, layer_idx
    ):
        """Evaluate the relative quantization error of a single layer.

        Parameters
        ----------
        quantized : np.array
            The quantized weights
        unquantized : np.array
            The unquantized weights
        data_a : np.array
            The activation of the unquantized layer
        data_q : np.array
            The activation of the quantized layer
        layer_idx : int
            The layer index"""
        numerator = np.linalg.norm(
            np.matmul(data_a, unquantized)
            - np.matmul(
                data_q,
                quantized,
            ),
            "fro",
        )
        denominator = np.linalg.norm(
            np.matmul(data_a, unquantized),
            "fro",
        )
        self.eval_layer["error"].append(numerator / denominator)
        self.eval_layer["m"].append(self.batch_size)
        self.eval_layer["layer"].append(layer_idx)
        return

    def quantize_network(self):
        """Quantize the neural network layer by layer."""
        for layer_idx, layer in enumerate(self.trained_net.layers):
            if layer.__class__.__name__ == "Dense":
                self._quantize_layer(layer_idx)
