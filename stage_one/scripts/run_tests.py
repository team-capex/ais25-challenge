import os
import warnings
from argparse import ArgumentParser
from pathlib import Path
from typing import Dict, Iterable, List, MutableSequence, Sequence

import numpy as np
from ase import Atoms
from ase.calculators.calculator import Calculator
from ase.calculators.singlepoint import SinglePointCalculator
from ase.io import read
from ase.mep import NEB
from ase.optimize import FIRE

try:
    from tqdm import tqdm

    HAS_TQDM = True
except ImportError:
    HAS_TQDM = False

MAX_OPTIMIZER_STEPS = 10_000


def get_calculator() -> Calculator:
    """Create and return calculator."""
    ####################### REPLACE WITH OWN CALCULATOR #######################
    from mace.calculators import mace_mp

    return mace_mp()
    ###########################################################################


def split_by_test_type(images: Iterable[Atoms]) -> Dict[str, List[Atoms]]:
    """Split images by their test type."""
    test_type_images = {}
    for atoms in images:
        test_type_images.setdefault(atoms.info["test_type"], []).append(atoms)
    return test_type_images


def singlepoint(atoms: Atoms, calc: Calculator) -> None:
    """Run a singlepoint calculation."""
    atoms.calc = calc
    energy = atoms.get_potential_energy()
    forces = atoms.get_forces()
    atoms.calc = SinglePointCalculator(atoms, energy=energy, forces=forces)


def run_singlepoints(images: Iterable[Atoms], calc: Calculator) -> None:
    """Run singlepoint calculations on a sequence of images."""
    iterable = tqdm(images, desc="Running singlepoint") if HAS_TQDM else images
    for atoms in iterable:
        singlepoint(atoms, calc)


def relax(
    atoms: Atoms,
    calc: Calculator,
    fmax: float,
    trajectory: str | None = None,
    logfile: str | None = "-",
) -> None:
    """Relax atoms using the given calculator."""
    atoms.calc = calc
    opt = FIRE(atoms, logfile=logfile, trajectory=trajectory)
    opt.run(fmax=fmax, steps=MAX_OPTIMIZER_STEPS)
    if not opt.converged():
        warnings.warn(
            "Relaxation did not converge within the maximum number of steps. "
            f"For reference, {atoms.info=}, {atoms=}"
        )
    atoms.calc = SinglePointCalculator(
        atoms, energy=atoms.get_potential_energy(), forces=atoms.get_forces()
    )


def run_relaxations(
    images: Iterable[Atoms], calc: Calculator, relax_dir: str, fmax: float
) -> None:
    """Run relaxations on a sequence of images, logging the results."""
    os.makedirs(relax_dir, exist_ok=True)
    iterable = tqdm(images, desc="Relaxing images") if HAS_TQDM else images
    for atoms in iterable:
        name = f"relax_{atoms.info['image_index']}"
        trajectory = os.path.join(relax_dir, f"{name}.traj")
        logfile = os.path.join(relax_dir, f"{name}.log")
        relax(atoms, calc, fmax=fmax, trajectory=trajectory, logfile=logfile)


def _append_before_suffix(path: str, appendix: str) -> str:
    """Append *appendix* before the suffix of the given *path*."""
    p = Path(path)
    return str(p.with_name(p.stem + appendix + p.suffix))


def run_neb(
    initial: Atoms,
    final: Atoms,
    calculators: Sequence[Calculator],
    fmax_relax: float,
    fmax_neb: float,
    trajectory: str | None = None,
    logfile: str | None = "-",
) -> Atoms:
    """Run a NEB calculation between *initial* and *final* using the given
    calculators and return the transition state image.
    """
    # Relax endpoints
    relax(
        initial, calculators[0], fmax=fmax_relax, trajectory=None, logfile=None
    )
    relax(
        final, calculators[-1], fmax=fmax_relax, trajectory=None, logfile=None
    )

    # Set up NEB
    num_moving = len(calculators) - 2
    images = [initial] + [initial.copy() for _ in range(num_moving)] + [final]
    neb = NEB(images)
    neb.interpolate()

    # Assign calculators to moving images
    for image, calc in zip(images[1:-1], calculators[1:-1]):
        image.calc = calc

    # Run NEB
    opt = FIRE(neb, trajectory=trajectory, logfile=logfile)
    opt.run(fmax=fmax_neb, steps=MAX_OPTIMIZER_STEPS)
    if not opt.converged():
        warnings.warn(
            "NEB did not converge within the maximum number of steps. "
            f"For reference, {initial.info=}, {final.info=}"
        )
        return max(images, key=lambda atoms: atoms.get_potential_energy())

    # Finish with climbing image NEB
    neb.climb = True
    trajectory_climb = (
        trajectory
        if trajectory is None
        else _append_before_suffix(trajectory, "_climb")
    )
    logfile_climb = (
        logfile
        if logfile is None or logfile == "-"
        else _append_before_suffix(logfile, "_climb")
    )
    opt = FIRE(neb, trajectory=trajectory_climb, logfile=logfile_climb)
    opt.run(fmax=fmax_neb, steps=MAX_OPTIMIZER_STEPS)
    if not opt.converged():
        warnings.warn(
            "NEB climb did not converge within the maximum number of steps. "
            f"For reference, {initial.info=}, {final.info=}"
        )
    transition_state = max(
        images, key=lambda atoms: atoms.get_potential_energy()
    )
    # Make sure a singlepoint calculator is attached, so calculators can be reused
    singlepoint(transition_state, calculators[0])
    return transition_state


def run_nebs(
    images: MutableSequence[Atoms],
    neb_dir: str,
    fmax_relax: float,
    fmax_neb: float,
    num_images: int = 21,
) -> None:
    """Run NEB calculations on pairs of images in the given sequence and append
    the resulting transition states to the image sequence.
    """
    for atoms in images:
        assert (atoms.info["image_index"] % 3) % 2 == 0, "NEB endpoints only"
    os.makedirs(neb_dir, exist_ok=True)
    calcs = [get_calculator() for _ in range(num_images)]
    iterable = (
        tqdm(images[::2], desc="Running NEBs") if HAS_TQDM else images[::2]
    )
    for i, (initial, final) in enumerate(zip(iterable, images[1::2])):
        ts_index = 3 * i + 1
        name = f"neb_{initial.info['image_index']}-{final.info['image_index']}"
        trajectory = os.path.join(neb_dir, f"{name}.traj")
        logfile = os.path.join(neb_dir, f"{name}.log")
        transition_state = run_neb(
            initial,
            final,
            calcs,
            fmax_relax=fmax_relax,
            fmax_neb=fmax_neb,
            trajectory=trajectory,
            logfile=logfile,
        )
        transition_state.info["image_index"] = ts_index
        images.append(transition_state)


def gather_results(images: Sequence[Atoms], max_size: int):
    """Gather results from the given sequence of images into a dictionary.
    Arrays are padded with NaNs to the given maximum size.
    """
    # TODO: Use single precision to save file space? or compressed npz?
    test_indices = np.array(
        [atoms.info["test_index"] for atoms in images], dtype=int
    )
    image_indices = np.array(
        [atoms.info["image_index"] for atoms in images], dtype=int
    )
    energies = np.array(
        [atoms.get_potential_energy() for atoms in images], dtype=float
    )
    positions = np.full(
        (len(images), max_size, 3), fill_value=np.nan, dtype=float
    )
    forces = np.full(
        (len(images), max_size, 3), fill_value=np.nan, dtype=float
    )
    pbcs = np.zeros((len(images), 3), dtype=bool)
    cells = np.zeros((len(images), 3, 3), dtype=float)
    np.einsum("nii->ni", cells)[:] = 1.0
    for i, atoms in enumerate(images):
        natoms = len(atoms)
        positions[i, :natoms, :] = atoms.get_positions()
        forces[i, :natoms, :] = atoms.get_forces()
        pbc = atoms.get_pbc()
        pbcs[i, :] = pbc
        cells[i, pbc, :] = atoms.get_cell().array[pbc]
    results = dict(
        test_indices=test_indices,
        image_indices=image_indices,
        positions=positions,
        energies=energies,
        forces=forces,
        pbcs=pbcs,
        cells=cells,
    )
    return results


def main():
    parser = ArgumentParser(
        description="Run tests using your calculator and save results."
    )
    parser.add_argument(
        "input",
        type=str,
        help="Path to input trajectory file containing test images.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="../data/results.npz",
        help="Path to output file to save results.",
    )
    parser.add_argument(
        "--fmax_relax",
        type=float,
        default=0.001,
        help="Maximum force convergence criterion for relaxations.",
    )
    parser.add_argument(
        "--relax_dir",
        type=str,
        default="./relax",
        help="Directory to save relaxation trajectories and logs.",
    )
    parser.add_argument(
        "--neb_dir",
        type=str,
        default="./neb",
        help="Directory to save NEB trajectories and logs.",
    )
    parser.add_argument(
        "--fmax_neb",
        type=float,
        default=0.01,
        help="Maximum force convergence criterion for NEB calculations.",
    )
    parser.add_argument(
        "--num_images",
        type=int,
        default=21,
        help="Number of images to use in NEB calculations.",
    )
    args = parser.parse_args()

    images: List[Atoms] = read(args.input, index=":")  # type: ignore
    max_size = max(len(atoms) for atoms in images)
    calc = get_calculator()

    test_type_images = split_by_test_type(images)
    assert set(test_type_images.keys()) == {
        "sp_check",
        "sp_large",
        "relax",
        "neb",
    }
    run_singlepoints(test_type_images["sp_check"], calc)
    run_singlepoints(test_type_images["sp_large"], calc)
    run_relaxations(
        test_type_images["relax"],
        calc,
        relax_dir=args.relax_dir,
        fmax=args.fmax_relax,
    )
    run_nebs(
        test_type_images["neb"],
        neb_dir=args.neb_dir,
        fmax_neb=args.fmax_neb,
        fmax_relax=args.fmax_relax,
        num_images=args.num_images,
    )

    flattened_images = [
        atoms
        for atoms_list in test_type_images.values()
        for atoms in atoms_list
    ]
    np.savez(args.output, **gather_results(flattened_images, max_size))


if __name__ == "__main__":
    main()
