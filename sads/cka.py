import numpy as np


def _center_kernel(K: np.ndarray) -> np.ndarray:
    n = K.shape[0]
    H = np.eye(n) - np.ones((n, n)) / n
    return H @ K @ H

def cka_score(K: np.ndarray, y: np.ndarray, target_kernel: str = "rbf") -> float:
    Kc = _center_kernel(K)

    if target_kernel == "rbf":
        # Gaussian kernel on targets
        diffs = y[:, None] - y[None, :]
        n = K.shape[0]
        mask = ~np.eye(n, dtype=bool)
        sigma = np.median(np.abs(diffs[mask]))
        if sigma == 0:  # fallback to avoid divide-by-zero
            sigma = np.std(y) + 1e-12
        T = np.exp(-(diffs**2) / (2 * sigma**2 + 1e-12))

    elif target_kernel == "linear":
        # Linear kernel: yy^T
        T = np.outer(y, y)

    else:
        raise ValueError("target_kernel must be 'rbf' or 'linear'")

    Tc = _center_kernel(T)
    similarity = (Kc * Tc).sum()
    normalization = np.linalg.norm(Kc, 'fro') * np.linalg.norm(Tc, ord='fro') + 1e-12
    return similarity / normalization