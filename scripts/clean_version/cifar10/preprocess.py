""" Module to preprocess the layers of a neural network."""

import sys

import numpy as np
from scipy.linalg import null_space
from tqdm import tqdm


def compute_alpha(z, b, J_rel, c):
    """Compute the alpha value for the preprocessing algorithm.
    Parameters
    ----------
    z : np.array
        The current neuron.
    b : np.array
        The current kernel vector.
    J_rel : np.array
        The current relevant indices.
    c : float
        Maximum value of the current neuron.
    Returns
    -------
    alpha : float
        The alpha value.
    idx : int
        The index of the to be erased entry of the other basis elements.
    """
    alpha_i = []
    j = 0
    for i in J_rel:
        if b[j] != 0 and -c - z[i] != 0 and c - z[i] != 0 and b[j] != 1:
            alpha_1 = np.float32((-c - z[i]) / (1.0 * b[j]))
            alpha_2 = np.float32((c - z[i]) / (1.0 * b[j]))
            alpha_i.append(min([alpha_1, alpha_2], key=abs))
            j += 1
        else:
            alpha_i.append(sys.float_info.max)
            j += 1
    alpha = min(alpha_i, key=abs)
    idx = alpha_i.index(alpha)
    return alpha, idx


def reduce_basis(B, idx):
    """Reduce the basis of the kernel by one generating a basis for the kernel with one more 0 entry.
    Parameters
    ----------
    B : np.array
        The current kernel.
    idx : int
        The index of the to be erased entry of the other basis elements.
    Returns
    -------
    B_prime : np.array
        The reduced basis."""
    if B[idx, 1] != 0:
        B_prime = np.reshape(B[:, 0] - B[idx, 0] / B[idx, 1] * B[:, 1], (B.shape[0], 1))
        B_prime = B_prime / np.linalg.norm(B_prime)
    else:
        B_prime = np.reshape(B[:, 1], (B.shape[0], 1))
    for i in range(2, B.shape[1]):
        if B[idx, i] != 0:
            b_i = np.reshape(
                (B[:, 0] - B[idx, 0] / B[idx, i] * B[:, i]), (B.shape[0], 1)
            )
            # normalize
            b_i = b_i / np.linalg.norm(b_i)
        else:
            b_i = np.reshape(B[:, i], (B.shape[0], 1))
            # normalize
            b_i = b_i / np.linalg.norm(b_i)

        B_prime = np.concatenate((B_prime, b_i), axis=1)

    B_prime[np.abs(B_prime) < 1e-8] = 0

    return B_prime


def add_z_alpha_b(z, alpha, b, J_rel):
    """Add the kernel vector times alpha to the current neuron on the relevant indices.
    Parameters
    ----------
    z : np.array
        The current neuron.
    alpha : float
        The alpha value.
    b : np.array
        The current kernel vector.
    J_rel : np.array
        The current relevant indices.
    Returns
    -------
    z : np.array
        The updated neuron."""
    j = 0
    for i in J_rel:
        z[i] = z[i] + alpha * b[j]
        j += 1
    return z


def preprocessing_Neuron_p_32bit(A_0, z_0, c, p_version=True):
    """Preprocess a single neuron.
    Parameters
    ----------
    A_0 : np.array
        data matrix
    z_0 : np.array
        The current neuron.
    c : float
        Maximum value of the current neuron.
    p_version : bool
        If true, use the version developed in the thesis/basisreduction, else the original algo.
    Returns
    -------
    z_it : np.array
        The preprocessed neuron.
    kernel_calculations : int
        The number of kernel calculations.
    sum : int
        Debug parameter.
    changes : list
        Debug parameter.
    """
    c_vec = np.ones(z_0.shape) * np.abs(c)
    m = A_0.shape[0]
    J_0 = np.where(~A_0.any(axis=0))[0]
    # print(J_0)
    # Step 2
    b = np.zeros(z_0.shape, dtype=np.float32)
    for i in J_0:
        b[i] = c - z_0[i]
    # Step 3
    z_it = z_0 + b
    # Step 4
    kernel_calculations = 0

    sum = 0
    changes = []
    if p_version:
        while (np.count_nonzero(np.isclose(np.abs(z_it), c_vec))) < A_0.shape[1] - m:
            kernel_calculations += 1
            J = np.where(np.isclose(np.abs(z_it), c_vec) == False)[0]
            if len(J) >= 2 * m:
                J_rel = J[: 2 * m]
            else:
                J_rel = J
            A_k = np.asanyarray(A_0[:, J_rel])
            B = null_space(A_k)

            alpha, idx = compute_alpha(z_it, B[:, 0], J_rel, c)
            z_it = add_z_alpha_b(z_it, alpha, B[:, 0], J_rel)

            for i in range(len(J_rel) - m - 1):
                B = reduce_basis(B, idx)
                alpha, idx = compute_alpha(z_it, B[:, 0], J_rel, c)
                z_it = add_z_alpha_b(z_it, alpha, B[:, 0], J_rel)

    else:
        while (np.count_nonzero(np.isclose(np.abs(z_it), c_vec))) < A_0.shape[1] - m:
            kernel_calculations += 1
            J = np.where(np.isclose(np.abs(z_it), c_vec) == False)[0]
            J_rel = J[: m + 1]
            A_k = np.asanyarray(A_0[:, J_rel])
            b = null_space(A_k)[:, 0]
            alpha, _ = compute_alpha(z_it, b, J_rel, c)
            z_it = add_z_alpha_b(z_it, alpha, b, J_rel)

    return z_it, kernel_calculations, sum, changes


def preprocessing_layer(layer, A, p_version=True):
    """Preprocess a layer.
    Parameters
    ----------
    layer : np.array
        The current layer.
    A : np.array
        The data matrix.
    p_version : bool
        If true, use the version developed in the thesis/basisreduction, else the original algo.
    Returns
    -------
    preprocessed_layer : np.array
        The preprocessed layer.
    kernel_calculations_sum : int
        The number of kernel calculations.
    """
    layer_flat = layer.flatten()
    c = max(layer_flat, key=abs)
    dim = layer.shape[0]
    preprocessed_layer, kernel_calculations_sum, _, _ = preprocessing_Neuron_p_32bit(
        A, layer[:, 0], c, p_version
    )
    preprocessed_layer = np.reshape(preprocessed_layer, (dim, 1))
    for i in tqdm(range(1, layer.shape[1])):
        neuron_i_proc, kernel_calculations, _, _ = preprocessing_Neuron_p_32bit(
            A, layer[:, i], c, p_version
        )
        kernel_calculations_sum += kernel_calculations
        neuron_i_proc = np.reshape(neuron_i_proc, (dim, 1))
        preprocessed_layer = np.concatenate((preprocessed_layer, neuron_i_proc), axis=1)
    return preprocessed_layer, kernel_calculations_sum
