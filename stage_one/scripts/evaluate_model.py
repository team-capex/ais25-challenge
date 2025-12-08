from argparse import ArgumentParser
from dataclasses import dataclass

import numpy as np


@dataclass
class Results:
    test_indices: np.ndarray  # (B,)
    image_indices: np.ndarray  # (B,)
    model_energies: np.ndarray  # (B,)
    model_forces: np.ndarray  # (B, N, 3)
    model_positions: np.ndarray  # (B, N, 3)
    ref_energies: np.ndarray  # (B,)
    ref_forces: np.ndarray  # (B, N, 3)
    ref_positions: np.ndarray  # (B, N, 3)
    natoms: np.ndarray  # (B,)
    cells: np.ndarray  # (B, 3, 3)
    pbcs: np.ndarray  # (B, 3)

    def __getitem__(self, key) -> "Results":
        if isinstance(key, int):
            key = slice(key, key + 1)
        return Results(
            test_indices=self.test_indices[key],
            image_indices=self.image_indices[key],
            model_energies=self.model_energies[key],
            model_forces=self.model_forces[key],
            model_positions=self.model_positions[key],
            ref_energies=self.ref_energies[key],
            ref_forces=self.ref_forces[key],
            ref_positions=self.ref_positions[key],
            natoms=self.natoms[key],
            cells=self.cells[key],
            pbcs=self.pbcs[key],
        )


def results_from_npz(path_model: str, path_ref: str) -> Results:
    """Load model and reference results from .npz files."""
    results_model = np.load(path_model)
    results_ref = np.load(path_ref)

    # Sort model results first by test index, then by image index (for same
    # test index). NEB transition states are appended so the order is wrong
    # before sorting
    sorted_indices = np.lexsort(
        (results_model["image_indices"], results_model["test_indices"])
    )
    results_model_sorted = {
        key: results_model[key][sorted_indices] for key in results_model.files
    }

    # Same image ordering in reference and model results
    assert np.equal(
        results_ref["test_indices"], results_model_sorted["test_indices"]
    ).all()
    assert np.equal(
        results_ref["image_indices"], results_model_sorted["image_indices"]
    ).all()

    natoms = np.isfinite(results_model_sorted["forces"][:, :, 0]).sum(axis=1)

    res = Results(
        test_indices=results_model_sorted["test_indices"],
        image_indices=results_model_sorted["image_indices"],
        model_energies=results_model_sorted["energies"],
        model_forces=results_model_sorted["forces"],
        model_positions=results_model_sorted["positions"],
        ref_energies=results_ref["energies"],
        ref_forces=results_ref["forces"],
        ref_positions=results_ref["positions"],
        natoms=natoms,
        cells=results_model_sorted["cells"],
        pbcs=results_model_sorted["pbcs"],
    )
    results_model.close()
    results_ref.close()
    return res


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


def rmsd(
    predicted: np.ndarray,
    reference: np.ndarray,
    cells: np.ndarray,
    pbcs: np.ndarray,
) -> float:
    """Compute RMSD between predicted and reference positions using the
    minimum image convention. Note that rotations are not accounted for.
    """
    # Center non-periodic dimensions
    pred_shift = np.nanmean(predicted, axis=1, keepdims=True)
    pred_shift *= ~pbcs[:, np.newaxis, :]
    predicted = predicted - pred_shift  # (B, N, 3)

    ref_shift = np.nanmean(reference, axis=1, keepdims=True)
    ref_shift *= ~pbcs[:, np.newaxis, :]
    reference = reference - ref_shift  # (B, N, 3)

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


def get_loss_locality(res: Results, tol: float = 1e-4) -> dict[str, float]:
    """Get loss in test 1: isolated vs duplicated system"""
    # Check that energy is extensive
    energies_isolated = res.model_energies[::2]
    energies_double = res.model_energies[1::2]
    loss_e = np.max(
        np.abs((energies_double - 2 * energies_isolated) / res.natoms)
    )

    # Check that forces are identical
    forces_isolated = res.model_forces[::2]
    forces_double = res.model_forces[1::2]
    # Forces are padded with NaNs for smaller systems, so use nanmax
    loss_f = np.nanmax(np.abs(forces_double - forces_isolated))
    return dict(energy=max(loss_e, tol), force=max(loss_f, tol))


def get_loss_relax(res: Results) -> dict[str, float]:
    """Get loss in test 2: geometry relaxation"""
    assert np.all(res.natoms[::2] == res.natoms[1::2])
    natoms = res.natoms[::2]

    # Energy differences between paired polymorphs. Assume pairs are ordered
    # sequentially.
    model_energy_diff = res.model_energies[1::2] - res.model_energies[::2]
    ref_energy_diff = res.ref_energies[1::2] - res.ref_energies[::2]
    energy_loss = energy_rmse(model_energy_diff, ref_energy_diff, natoms)

    # RMSD between relaxed positions of model and reference
    rmsd1 = rmsd(
        res.model_positions[::2],
        res.ref_positions[::2],
        res.cells[::2],
        res.pbcs[::2],
    )
    rmsd2 = rmsd(
        res.model_positions[1::2],
        res.ref_positions[1::2],
        res.cells[1::2],
        res.pbcs[1::2],
    )
    return dict(energy=energy_loss, rmsd=(rmsd1 + rmsd2) / 2)


def get_loss_neb(res: Results) -> dict[str, float]:
    """Get loss in test 3: NEB"""
    assert np.all(res.natoms[::3] == res.natoms[1::3])
    assert np.all(res.natoms[::3] == res.natoms[2::3])

    # NEB images are ordered in triplets as: initial, transition state, final
    # Reaction and activation energy losses
    model_reac_energies = res.model_energies[2::3] - res.model_energies[::3]
    model_act_energies = res.model_energies[1::3] - res.model_energies[::3]
    ref_reac_energies = res.ref_energies[2::3] - res.ref_energies[::3]
    ref_act_energies = res.ref_energies[1::3] - res.ref_energies[::3]
    energy_loss_reaction = energy_rmse(
        model_reac_energies, ref_reac_energies, natoms=None
    )
    energy_loss_activation = energy_rmse(
        model_act_energies, ref_act_energies, natoms=None
    )

    # RMSD between initial, transition state, and final positions of model and
    # reference
    rmsd1 = rmsd(
        res.model_positions[::3],
        res.ref_positions[::3],
        res.cells[::3],
        res.pbcs[::3],
    )
    rmsd2 = rmsd(
        res.model_positions[1::3],
        res.ref_positions[1::3],
        res.cells[1::3],
        res.pbcs[1::3],
    )
    rmsd3 = rmsd(
        res.model_positions[2::3],
        res.ref_positions[2::3],
        res.cells[2::3],
        res.pbcs[2::3],
    )
    return dict(
        reaction_energy=energy_loss_reaction,
        activation_energy=energy_loss_activation,
        rmsd_ends=(rmsd1 + rmsd3) / 2,
        rmsd_ts=rmsd2,
    )


def get_loss_sp(res: Results) -> dict[str, float]:
    """Get loss in test 4: single point calculations"""
    energy_loss = energy_rmse(res.model_energies, res.ref_energies, res.natoms)
    force_loss = force_rmse(res.model_forces, res.ref_forces)
    return dict(energy=energy_loss, force=force_loss)


def main():
    parser = ArgumentParser(
        description="Evaluate model performance on test results."
    )
    parser.add_argument(
        "model_results",
        type=str,
        help="Path to .npz file containing model results.",
    )
    parser.add_argument(
        "reference_results",
        type=str,
        help="Path to .npz file containing reference results.",
    )
    args = parser.parse_args()

    res = results_from_npz(args.model_results, args.reference_results)

    # Slice results by test
    slice_bounds = np.searchsorted(res.test_indices, [1, 2, 3, 4, 5])
    slices = [slice(i, j) for i, j in zip(slice_bounds[:-1], slice_bounds[1:])]

    # Wrap positions back to unit cell to ensure the minimum image convention
    # works correctly when computing RMSD.
    res.model_positions = wrap_positions(
        res.model_positions,
        res.cells,
        res.pbcs,
    )
    res.ref_positions = wrap_positions(
        res.ref_positions,
        res.cells,
        res.pbcs,
    )

    # Compute losses for each test
    losses = {
        test_name: get_losses(res[s])
        for test_name, get_losses, s in zip(
            [
                "locality",
                "relaxation",
                "neb",
                "single_point",
            ],
            [
                get_loss_locality,
                get_loss_relax,
                get_loss_neb,
                get_loss_sp,
            ],
            slices,
        )
    }

    # Print results
    units = {
        "energy": "eV/atom",
        "force": "eV/Å",
        "rmsd": "Å",
        "rmsd_ts": "Å",
        "rmsd_ends": "Å",
        "reaction_energy": "eV",
        "activation_energy": "eV",
    }
    for test_name, loss_dict in losses.items():
        print(f"Test: {test_name}")
        for loss_name, loss_value in loss_dict.items():
            unit = units[loss_name]
            print(f"  {loss_name}: {loss_value:.6f} {unit}")


if __name__ == "__main__":
    main()
