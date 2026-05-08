import numpy as np


def compute_reflectivity(input_powers: np.ndarray, sbs_powers: np.ndarray) -> np.ndarray:
    """
    Compute per-beam reflectivity with safe divide-by-zero behavior.

    Both inputs are expected to have identical shapes, typically (beams,) or (beams, tpts).

    Note: This function is duplicated in sbsbs_train.py. The implementation there uses
    explicit indexing which is preferred. This version is kept for backward compatibility
    but should be consolidated.
    """
    input_powers = np.asarray(input_powers)
    sbs_powers = np.asarray(sbs_powers)
    reflectivity = np.zeros_like(input_powers, dtype=float)
    nonzero = input_powers != 0
    reflectivity[nonzero] = sbs_powers[nonzero] / input_powers[nonzero]
    return reflectivity


def masked_mse(pred: np.ndarray, target: np.ndarray, mask: np.ndarray) -> float:
    pred = np.asarray(pred, dtype=float)
    target = np.asarray(target, dtype=float)
    mask = np.asarray(mask, dtype=float)

    denom = float(np.maximum(mask.sum(), 1.0))
    return float(((pred - target) ** 2 * mask).sum() / denom)

