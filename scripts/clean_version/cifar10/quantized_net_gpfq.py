"""Quantizing Neutal networks using Greedy path following quantization."""

import numpy as np
from keras.models import Model, clone_model
from tqdm import tqdm


def _quantize_neuron(w, analog_layer_input, quantized_layer_input, alphabet):
    """Quantize a single neuron.
    Parameters
    ----------
    w : np.array
        The weights of the neuron.
    analog_layer_input : np.array
        The activation of the previous unquantized layer.
    quantized_layer_input : np.array
        The activation of the previous quantized layer.
    alphabet : np.array
        The alphabet used for quantization.
    Returns
    -------
    q : np.array
        The quantized neuron.
    """
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
    def __init__(
        self,
        network: Model,
        batch_size,
        training_data,
        bits,
        deeper_layer,
        alphabet_scalar=1,
    ):
        """Wrapper class for tensorflow.keras.models.Model for
        quantizing the weights for Dense layers
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
        deeper_layer : bool
            Debug parameter
        alphabet_scalar : int, optional
            Scalar for the alphabet, by default 1
        """

        # pretrained network
        self.trained_net = network

        # Copies pretrained network structure and the weights
        self.quantized_net = clone_model(network)
        self.quantized_net.set_weights(network.get_weights())

        # Training data, used for preprocessing the layers
        self.training_data = training_data

        # How many training data points should be used for preprocessing the layers
        self.batch_size = batch_size
        self.alphabet_scalar = alphabet_scalar

        # Size of alphabet and alphabet
        self.bits = bits
        self.alphabet = np.linspace(-1, 1, num=int(round(2 ** (bits))))

        # If deeper_layer is true, the quantization input is the quantized layer
        self.deeper_layer = deeper_layer

        # Error evaluation per layer
        self.eval_layer = {"m": [], "error": [], "layer": []}

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

    def _quantize_layer(self, layer_idx: int):
        """Quantize a single layer.

        Parameters
        ----------
        layer_idx : int
            The layer index."""
        if layer_idx == 0:  # dont quantize flatten layer
            return

        W = self.trained_net.layers[layer_idx].get_weights()[0]
        Q = np.zeros(W.shape)

        # radius of alphabet
        rad = np.median(abs(W.flatten()))
        layer_alphabet = rad * self.alphabet * self.alphabet_scalar
        # quantized activation
        X_tilde = self._get_intermediate_output_q(layer_idx)
        # unquantized activation
        X = self._get_intermediate_output_a(layer_idx)
        for i in tqdm(range(W.shape[1])):
            Q[:, i] = _quantize_neuron(W[:, i], X, X_tilde, layer_alphabet)
        self._update_weights(layer_idx, Q)
        self._eval_layer_quant_error(Q, W, X, X_tilde, layer_idx)

    def quantize_network(self):
        """Quantize the network."""
        for layer_idx, layer in enumerate(self.trained_net.layers):
            if layer.__class__.__name__ == "Dense":  # only quantize dense layers
                self._quantize_layer(layer_idx)

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
