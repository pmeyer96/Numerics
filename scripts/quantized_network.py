import numpy as np
def bit_round_paralell(t: float, alphabet: np.array) -> float:
    return alphabet[np.argmin(abs(alphabet-t))]