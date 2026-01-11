import numpy as np
from losses import force_rmse


def rmsd(
    predicted: np.ndarray,
    reference: np.ndarray,
    cells: np.ndarray,
    pbcs: np.ndarray,
    fixed: np.ndarray,
) -> float:
    """Compute RMSD between predicted and reference positions using the
    minimum image convention.
    """
    # Center structures
    pred_shift = np.nanmean(predicted, axis=1, keepdims=True)
    predicted = predicted - ~fixed[:, None, None] * pred_shift  # (B, N, 3)

    ref_shift = np.nanmean(reference, axis=1, keepdims=True)
    reference = reference - ~fixed[:, None, None] * ref_shift  # (B, N, 3)

    # Rotate structures. Set padded NaNs to zero for rotation calculation.
    # Atoms at the origin will not affect rotation.
    pred_zeroed = np.where(np.isnan(predicted), 0.0, predicted)  # (B, N, 3)
    ref_zeroed = np.where(np.isnan(reference), 0.0, reference)  # (B, N, 3)
    rot_mats = rotation_matrix_from_points(
        pred_zeroed, ref_zeroed
    )  # (B, 3, 3)

    # Don't rotate systems with fixed atoms or periodic boundary conditions
    rot_mats = np.where(
        (fixed | pbcs.any(axis=1))[:, None, None], np.eye(3), rot_mats
    )  # (B, 3, 3)
    pred_zeroed = np.einsum("bni, bji -> bnj", pred_zeroed, rot_mats)
    predicted = np.where(np.isnan(predicted), np.nan, pred_zeroed)

    # Generate all 3x3x3 = 27 shifts to neighboring cells
    unit_shifts = np.tile(
        [
            (i, j, k)
            for i in (-1, 0, 1)
            for j in (-1, 0, 1)
            for k in (-1, 0, 1)
        ],
        (predicted.shape[0], 1, 1),
    )  # (B, 27, 3)
    unit_shifts *= pbcs[:, np.newaxis, :]
    cell_shifts = np.einsum("bsk,bkj->bsj", unit_shifts, cells)

    # Find closest periodic image of each atom in predicted to reference
    expanded = (
        predicted[:, :, np.newaxis, :] + cell_shifts[:, np.newaxis, :, :]
    )  # (B, N, 27, 3)
    all_dists = np.linalg.norm(
        expanded - reference[:, :, np.newaxis, :], axis=-1
    )  # (B, N, 27)
    closest_pred = np.take_along_axis(
        expanded,
        np.argmin(all_dists, axis=-1)[:, :, np.newaxis, np.newaxis],
        axis=2,
    ).squeeze(2)  # (B, N, 3)
    return force_rmse(closest_pred, reference)


def wrap_positions(
    positions: np.ndarray, cell: np.ndarray, pbc: np.ndarray
) -> np.ndarray:
    """Wrap positions back to the unit cell."""
    # Convert cartesian to cell coordinates.
    fractional = np.swapaxes(
        np.linalg.solve(
            np.swapaxes(cell, -1, -2), np.swapaxes(positions, -1, -2)
        ),
        -1,
        -2,
    )  # (B, N, 3)

    # Floor the result to get the unit shifts.
    unit_shifts = np.floor(fractional)

    # Only wrap periodic dimensions.
    unit_shifts *= np.asarray(pbc[:, None, :], dtype=bool)

    # Move atoms a whole number of unit cell vectors to map them back to the
    # unit cell.
    cell_shifts = unit_shifts @ cell
    pos_wrapped = positions - cell_shifts
    return pos_wrapped


###############################################################################
#############    Vectorized implementation of ase.build.rotate   ##############
###############################################################################
def rotation_matrix_from_points(m0: np.ndarray, m1: np.ndarray) -> np.ndarray:
    """Returns a rigid transformation/rotation matrix that minimizes the
    RMSD between two set of points.

    m0 and m1 should be (..., npoints, 3) numpy arrays with
    coordinates as rows (the first axes are batch dimensions)::

        [(x1, y1, z1),
         (x2, y2, z2),
         (x3, y3, z3),
         ...
         (xN, yN, zN)]

    The centeroids should be set to origin prior to
    computing the rotation matrix.

    The rotation matrix is computed using quaternion
    algebra as detailed in::

        Melander et al. J. Chem. Theory Comput., 2015, 11,1055
    """
    # compute the rotation quaternion
    R11, R22, R33 = np.einsum("...ni, ...ni -> i...", m0, m1)
    R12, R23, R31 = np.einsum(
        "...ni, ...ni -> i...", m0, np.roll(m1, -1, axis=-1)
    )
    R13, R21, R32 = np.einsum(
        "...ni, ...ni -> i...", m0, np.roll(m1, -2, axis=-1)
    )

    f = [
        [R11 + R22 + R33, R23 - R32, R31 - R13, R12 - R21],
        [R23 - R32, R11 - R22 - R33, R12 + R21, R13 + R31],
        [R31 - R13, R12 + R21, -R11 + R22 - R33, R23 + R32],
        [R12 - R21, R13 + R31, R23 + R32, -R11 - R22 + R33],
    ]

    F = np.array(f)  # shape = (4, 4, ...)
    F = np.moveaxis(F, (0, 1), (-2, -1))  # shape = (..., 4, 4)

    w, V = np.linalg.eigh(F)
    # eigenvector corresponding to the most positive eigenvalue
    # q = V[:, np.argmax(w)]
    q = np.choose(
        np.argmax(w, axis=-1), np.moveaxis(V, (-2, -1), (1, 0))
    )  # shape = (4, ...)

    # Rotation matrix from the quaternion q
    R = quaternion_to_matrix(q)

    return R


def quaternion_to_matrix(q: np.ndarray) -> np.ndarray:
    """Returns a rotation matrix (..., 3, 3).

    Computed from a unit quaternion Input as (4, ...) numpy array.
    """
    q0, q1, q2, q3 = q
    R_q = [
        [
            q0**2 + q1**2 - q2**2 - q3**2,
            2 * (q1 * q2 - q0 * q3),
            2 * (q1 * q3 + q0 * q2),
        ],
        [
            2 * (q1 * q2 + q0 * q3),
            q0**2 - q1**2 + q2**2 - q3**2,
            2 * (q2 * q3 - q0 * q1),
        ],
        [
            2 * (q1 * q3 - q0 * q2),
            2 * (q2 * q3 + q0 * q1),
            q0**2 - q1**2 - q2**2 + q3**2,
        ],
    ]  # shape = (3, 3, ...)
    return np.moveaxis(np.array(R_q), (0, 1), (-2, -1))
