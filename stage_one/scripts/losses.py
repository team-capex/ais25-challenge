import numpy as np


def energy_rmse(
    predicted: np.ndarray, reference: np.ndarray, natoms: np.ndarray | None
) -> float:
    """Compute RMSE between predicted and reference energies."""
    diff = predicted - reference
    if natoms is not None:
        diff /= natoms
    return np.sqrt(np.mean(diff**2))


def force_rmse(predicted: np.ndarray, reference: np.ndarray) -> float:
    """Compute RMSE between predicted and reference forces."""
    return np.sqrt(
        np.mean(np.nanmean((predicted - reference) ** 2, axis=(1, 2)))
    )
