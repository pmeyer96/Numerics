""" This file contains the class for the normal distributed data. """

import numpy as np
from preprocess import preprocessing_layer, preprocessing_Neuron_p_32bit


def _quantize_neuron_gpfq(w, input, alphabet):
    """Quantize a single neuron using the gpfq scheme.
    Parameters
    ----------
    w : np.array
        The weights of the neuron.
    input : np.array
        The input data.
    alphabet : np.array
        The alphabet used for quantization.
    Returns
    -------
    q : np.array
        The quantized neuron.
    """
    u = np.zeros(input.shape[0])
    q = np.zeros_like(w)
    for t in range(w.shape[0]):
        u += w[t] * input[:, t]

        norm = np.linalg.norm(input[:, t], 2) ** 2
        if norm > 0:
            q_arg = np.dot(input[:, t], u) / norm
        else:
            q_arg = 0
        q[t] = alphabet[np.argmin(abs(alphabet - q_arg))]
        u -= q[t] * input[:, t]
    return q


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


def quantize_neuron(neuron, alphabet):
    """Quantize a neuron using MSQ.
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


def quantize_layer(layer, alphabet):
    """Quantize a layer using MSQ.
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
        quantized_layer[:, i] = quantize_neuron(quantized_layer[:, i], alphabet)
    return quantized_layer


class NormalDistributedData:
    def __init__(self, A, W, batch_sizes, alphabet_scalar, bits):
        """Wrapper class for the quantization of a single layer.
        Parameters
        ----------
        A : np.array
            The input data.
        W : np.array
            The to be quantized layer.
        batch_sizes : list
            The batch sizes used for quantizing the layer.
        alphabet_scalar : float
            The scalar used for the scaling of the alphabet.
        bits : int
            The number of bits used for quantization.
        """
        self.data = A
        self.weights = W
        self.weights_m = W
        self.weights_g = W
        self.weights_msq = W
        self.batch_sizes = batch_sizes
        self.bitsize = bits
        self.alphabet_scalar = alphabet_scalar
        self.alphabet = np.linspace(
            -1, 1, num=int(round(2 ** (bits))), dtype=np.float32
        )
        self.eval_neuron_msq = []
        self.eval_neuron_maly_s = []
        self.eval_neuron_maly = []
        self.eval_neuron_gpfq = []
        self.eval_layer_msq = []
        self.eval_layer_maly = []
        self.eval_layer_maly_s = []
        self.eval_layer_gpfq = []

    def eval_layer_quant_error(self, Q, W, X):
        """Evaluate the quantization error of a layer.
        Parameters
        ----------
        Q : np.array
            The quantized layer.
        W : np.array
            The unquantized layer.
        X : np.array
            The input data."""
        numerator = np.linalg.norm(np.matmul(X, W) - np.matmul(X, Q), "fro")
        denominator = np.linalg.norm(np.matmul(X, W))
        return numerator / denominator

    def change_alphabet_scalar(self, alphabet_scalar):
        """Change the alphabet scalar.
        Parameters
        ----------
        alphabet_scalar : float
            The new alphabet scalar."""
        self.alphabet_scalar = alphabet_scalar

    def eval_neuron_quant_error(self, q, w, X):
        """Evaluate the quantization error of a neuron.
        Parameters
        ----------
        q : np.array
            The quantized neuron.
        w : np.array
            The unquantized neuron.
        X : np.array
            The input data."""
        numerator = np.linalg.norm(np.matmul(X, w) - np.matmul(X, q))
        denominator = np.linalg.norm(np.matmul(X, w))
        return numerator / denominator

    def quantize_layer_gpfq(self):
        """Quantize the layer using the gpfq scheme."""
        for i in self.batch_sizes:
            print(i)
            Q = np.zeros(self.weights.shape)
            rad = np.median(abs(self.weights.flatten()))
            layer_alphabet = rad * self.alphabet * self.alphabet_scalar
            # layer_alphabet = self.alphabet
            input = self.data[0:i, :]
            for j in range(self.weights.shape[1]):
                Q[:, j] = _quantize_neuron_gpfq(
                    self.weights[:, j], input, layer_alphabet
                )

                self.eval_neuron_gpfq.append(
                    {
                        "m": i,
                        "bits": self.bitsize,
                        "error": self.eval_neuron_quant_error(
                            Q[:, j], self.weights[:, j], input
                        ),
                        "c_alpha": self.alphabet_scalar,
                    }
                )
            self.eval_layer_gpfq.append(
                {
                    "m": i,
                    "bits": self.bitsize,
                    "error": self.eval_layer_quant_error(Q, self.weights, input),
                    "c_alpha": self.alphabet_scalar,
                }
            )

    def quantize_neurons_seperately_maly(self):
        """Quantize the neurons with seperate alphabets using the maly scheme."""
        for i in self.batch_sizes:
            Q = np.zeros(self.weights.shape)
            input_data = self.data[0:i, :]
            for j in range(self.weights.shape[1]):
                c = max(np.abs(self.weights[:, j]))
                Q[:, j] = quantize_neuron(
                    preprocessing_Neuron_p_32bit(input_data, self.weights[:, j], c)[0],
                    c * self.alphabet,
                )
                # print(self.weights[:, j])
                self.eval_neuron_maly_s.append(
                    {
                        "m": i,
                        "bits": self.bitsize,
                        "error": self.eval_neuron_quant_error(
                            Q[:, j], self.weights[:, j], input_data
                        ),
                    }
                )
            self.eval_layer_maly_s.append(
                {
                    "m": i,
                    "bits": self.bitsize,
                    "error": self.eval_layer_quant_error(Q, self.weights, input_data),
                }
            )

    def quantize_layer_maly(self):
        """Quantize the layer using the maly scheme, all neurons same alphabet."""
        for i in self.batch_sizes:
            Q = np.zeros(self.weights.shape)
            input_data = self.data[0:i, :]
            Q, _ = preprocessing_layer(self.weights, input_data)
            c = max(self.weights.flatten(), key=abs)
            alphabet = self.alphabet * c
            Q = quantize_layer(Q, alphabet)
            for j in range(self.weights.shape[1]):
                self.eval_neuron_maly.append(
                    {
                        "m": i,
                        "bits": self.bitsize,
                        "error": self.eval_neuron_quant_error(
                            Q[:, j], self.weights[:, j], input_data
                        ),
                    }
                )
            self.eval_layer_maly.append(
                {
                    "m": i,
                    "bits": self.bitsize,
                    "error": self.eval_layer_quant_error(Q, self.weights, input_data),
                }
            )

    def quantize_layer_msq(self):
        """Quantize the layer using the MSQ scheme."""
        for i in self.batch_sizes:
            Q = np.zeros(self.weights.shape)
            c = max(self.weights.flatten(), key=abs)
            alphabet = self.alphabet * c

            Q = quantize_layer(self.weights, alphabet)
            for j in range(self.weights.shape[1]):
                self.eval_neuron_msq.append(
                    {
                        "m": i,
                        "bits": self.bitsize,
                        "error": self.eval_neuron_quant_error(
                            Q[:, j], self.weights[:, j], self.data[0:i, :]
                        ),
                    }
                )
            self.eval_layer_msq.append(
                {
                    "m": i,
                    "bits": self.bitsize,
                    "error": self.eval_layer_quant_error(
                        Q, self.weights, self.data[0:i, :]
                    ),
                }
            )
