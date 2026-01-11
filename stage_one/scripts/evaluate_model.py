from argparse import ArgumentParser
from dataclasses import dataclass

import numpy as np
from losses import energy_rmse, force_rmse
from positions import rmsd, wrap_positions


@dataclass
class Results:
    test_indices: np.ndarray  # (B,)
    image_indices: np.ndarray  # (B,)
    model_energies: np.ndarray  # (B,)
    model_forces: np.ndarray  # (B, N, 3)
    model_positions: np.ndarray  # (B, N, 3)
    model_bulk_energy: np.ndarray  # ()
    ref_energies: np.ndarray  # (B,)
    ref_forces: np.ndarray  # (B, N, 3)
    ref_positions: np.ndarray  # (B, N, 3)
    ref_bulk_energy: np.ndarray  # ()
    natoms: np.ndarray  # (B,)
    cells: np.ndarray  # (B, 3, 3)
    pbcs: np.ndarray  # (B, 3)
    fixed: np.ndarray  # (B,)

    def __getitem__(self, key) -> "Results":
        if isinstance(key, int):
            key = slice(key, key + 1)
        return Results(
            test_indices=self.test_indices[key],
            image_indices=self.image_indices[key],
            model_energies=self.model_energies[key],
            model_forces=self.model_forces[key],
            model_positions=self.model_positions[key],
            model_bulk_energy=self.model_bulk_energy,
            ref_energies=self.ref_energies[key],
            ref_forces=self.ref_forces[key],
            ref_positions=self.ref_positions[key],
            ref_bulk_energy=self.ref_bulk_energy,
            natoms=self.natoms[key],
            cells=self.cells[key],
            pbcs=self.pbcs[key],
            fixed=self.fixed[key],
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
    keys = results_model.files.copy()
    keys.remove("bulk_energy")  # Scalar, no need to sort
    results_model_sorted = {
        key: results_model[key][sorted_indices] for key in keys
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
        model_bulk_energy=results_model["bulk_energy"],
        ref_energies=results_ref["energies"],
        ref_forces=results_ref["forces"],
        ref_positions=results_ref["positions"],
        ref_bulk_energy=results_ref["bulk_energy"],
        natoms=natoms,
        cells=results_model_sorted["cells"],
        pbcs=results_model_sorted["pbcs"],
        fixed=results_ref["fixed"],
    )
    results_model.close()
    results_ref.close()
    return res


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
    res1 = res[::2]
    res2 = res[1::2]
    assert np.all(res1.natoms == res2.natoms)

    # Energy differences between paired polymorphs. Assume pairs are ordered
    # sequentially.
    model_energy_diff = res2.model_energies - res1.model_energies
    ref_energy_diff = res2.ref_energies - res1.ref_energies
    energy_loss = energy_rmse(model_energy_diff, ref_energy_diff, res1.natoms)

    # RMSD between relaxed positions of model and reference
    rmsd1 = rmsd(
        res1.model_positions,
        res1.ref_positions,
        res1.cells,
        res1.pbcs,
        res1.fixed,
    )
    rmsd2 = rmsd(
        res2.model_positions,
        res2.ref_positions,
        res2.cells,
        res2.pbcs,
        res2.fixed,
    )
    return dict(energy=energy_loss, rmsd=(rmsd1 + rmsd2) / 2)


def get_loss_neb(res: Results) -> dict[str, float]:
    """Get loss in test 3: NEB"""
    res_i = res[::3]
    res_ts = res[1::3]
    res_f = res[2::3]
    assert np.all(res_i.natoms == res_ts.natoms)
    assert np.all(res_i.natoms == res_f.natoms)

    # NEB images are ordered in triplets as: initial, transition state, final
    # Reaction and activation energy losses
    model_reac_energies = res_f.model_energies - res_i.model_energies
    model_act_energies = res_ts.model_energies - res_i.model_energies
    ref_reac_energies = res_f.ref_energies - res_i.ref_energies
    ref_act_energies = res_ts.ref_energies - res_i.ref_energies
    energy_loss_reaction = energy_rmse(
        model_reac_energies, ref_reac_energies, natoms=None
    )
    energy_loss_activation = energy_rmse(
        model_act_energies, ref_act_energies, natoms=None
    )

    # RMSD between initial, transition state, and final positions of model and
    # reference
    rmsd_i = rmsd(
        res_i.model_positions,
        res_i.ref_positions,
        res_i.cells,
        res_i.pbcs,
        res_i.fixed,
    )
    rmsd_ts = rmsd(
        res_ts.model_positions,
        res_ts.ref_positions,
        res_ts.cells,
        res_ts.pbcs,
        res_ts.fixed,
    )
    rmsd_f = rmsd(
        res_f.model_positions,
        res_f.ref_positions,
        res_f.cells,
        res_f.pbcs,
        res_f.fixed,
    )
    return dict(
        reaction_energy=energy_loss_reaction,
        activation_energy=energy_loss_activation,
        rmsd_ends=(rmsd_i + rmsd_f) / 2,
        rmsd_ts=rmsd_ts,
    )


def get_loss_sp(res: Results) -> dict[str, float]:
    """Get loss in test 4: single point calculations"""
    energy_loss = energy_rmse(
        res.model_energies - res.natoms * res.model_bulk_energy,
        res.ref_energies - res.natoms * res.ref_bulk_energy,
        res.natoms,
    )
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
