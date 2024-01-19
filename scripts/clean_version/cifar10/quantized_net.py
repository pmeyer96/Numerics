"""Quantize a neural network using OAQ."""
import numpy as np
from keras.models import Model, clone_model
from preprocess import preprocessing_layer


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


def quantizing_layer(layer, alphabet):
    """Quantize a layer.
    Parameters
    ----------
    layer : np.array
        The layer to be quantized.
    alphabet : np.array
        The alphabet used for quantization.
    Returns
    -------
    q : np.array
        The quantized layer.
    """
    quantized_layer = layer.copy()
    for i in range(layer.shape[1]):
        quantized_layer[:, i] = quantizing_neuron(quantized_layer[:, i], alphabet)
    return quantized_layer


def quantizing_neuron(neuron, alphabet):
    """Quantize a neuron.
    Parameters
    ----------
    neuron : np.array
        The neuron to be quantized.
    alphabet : np.array
        The alphabet used for quantization.
    Returns
    -------
    q : np.array
        The quantized neuron.
    """
    quantized_neuron = neuron.copy()
    for i in range(neuron.shape[0]):
        quantized_neuron[i] = quantizing_weight(quantized_neuron[i], alphabet)
    return quantized_neuron


class QuantizeNeuralNet:
    def __init__(self, network: Model, batch_size, training_data, bits, deeper_layer):
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
            If true, take quantized activation of previous layer as input for quantization
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

    def _quantize_layer(self, layer_idx: int, input_data):
        """Quantizes a layer of the neural network.

        Parameters
        ----------
        layer_idx : int
            The layer index
        input_data : np.array
            The input data for the first layer.
        """
        if layer_idx == 0:
            return
        W = self.trained_net.layers[layer_idx].get_weights()[0]
        Q = np.zeros(W.shape)
        if self.deeper_layer:  # use quantized activation
            data = np.asarray(self._get_intermediate_output_q(layer_idx, input_data))
            preprocessed_W, _ = preprocessing_layer(W, data)

        else:  # use analog activation
            data = np.asarray(self._get_intermediate_output_a(layer_idx, input_data))
            preprocessed_W, _ = preprocessing_layer(W, data)
        # Take biggest weight as alphabet scaling factor
        c = max(W.flatten(), key=abs)
        # calculation for relative quantization error evaluation
        data_q = np.asarray(self._get_intermediate_output_q(layer_idx, input_data))
        data_a = np.asarray(self._get_intermediate_output_a(layer_idx, input_data))
        alphabet = c * self.alphabet
        Q = quantizing_layer(preprocessed_W, alphabet)
        self._update_weights(layer_idx, Q)
        self._eval_layer_quant_error(Q, W, data_q, data_a, layer_idx)

    def _eval_layer_quant_error(
        self, quantized, unquantized, data_q, data_a, layer_idx
    ):
        """Evaluates the quantization error for every layer of a NN on a given data set
        Parameters
        ----------
        quantized : np.array
            The quantized weights
        unquantized : np.array
            The unquantized weights
        data_q : np.array
            The activation of quantizied NN
        data_a : np.array
            The activation of unquantizied NN
        """

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

    def _get_intermediate_output_q(self, layer_idx: int, data):
        """Generate the activation of the quantized layer.

        Parameters
        ----------
        layer_idx : int
            The layer index
        Returns
        -------
        intermediate_output : np.array
            The activation of the quantized layer"""
        if layer_idx == 0:
            return data

        inputs = self.quantized_net.input
        outputs = self.quantized_net.layers[layer_idx - 1].output
        intermediate_layer_model = Model(inputs=inputs, outputs=outputs)
        intermediate_output = intermediate_layer_model(data)

        return intermediate_output

    def _get_intermediate_output_a(self, layer_idx: int, data):
        """Generate the activation of the unquantized layer.

        Parameters
        ----------
        layer_idx : int
            The layer index
        Returns
        -------
        intermediate_output : np.array
            The activation of the unquantized layer"""
        if layer_idx == 0:
            return data

        inputs = self.trained_net.input
        outputs = self.trained_net.layers[layer_idx - 1].output
        intermediate_layer_model = Model(inputs=inputs, outputs=outputs)
        intermediate_output = intermediate_layer_model(data)

        return intermediate_output

    def quantize_network(self):
        """Quantize the whole network layer by layer."""
        data = self.training_data[: self.batch_size, :]

        for layer_idx, layer in enumerate(self.trained_net.layers):
            if layer.__class__.__name__ == "Dense":
                self._quantize_layer(layer_idx, data)
