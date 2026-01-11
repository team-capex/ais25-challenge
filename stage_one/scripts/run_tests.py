import os
import warnings
from argparse import ArgumentParser
from pathlib import Path
from typing import Dict, Iterable, List, Sequence

import numpy as np
from ase import Atoms, units
from ase.build import bulk
from ase.calculators.calculator import Calculator
from ase.calculators.singlepoint import SinglePointCalculator
from ase.eos import EquationOfState, calculate_eos
from ase.io import read
from ase.mep import NEB
from ase.optimize import FIRE
from matplotlib import pyplot as plt
from matplotlib.axes import Axes

try:
    from tqdm import tqdm

    HAS_TQDM = True
except ImportError:
    HAS_TQDM = False

MAX_OPTIMIZER_STEPS = 1_000


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
    images: list[Atoms], calc: Calculator, relax_dir: str, fmax: float
) -> None:
    """Run relaxations on a sequence of images, logging the results."""
    images.sort(key=lambda atoms: atoms.info["image_index"])
    for a1, a2 in zip(images[:-1:2], images[1::2]):
        assert a2.info["image_index"] - a1.info["image_index"] == 1
    iterable = tqdm(images, desc="Relaxing images") if HAS_TQDM else images
    os.makedirs(relax_dir, exist_ok=True)
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
        transition_state = max(
            images, key=lambda atoms: atoms.get_potential_energy()
        ).copy()
        singlepoint(transition_state, calculators[0])
        return transition_state

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
    ).copy()
    # Make sure a singlepoint calculator is attached, so calculators can be reused
    singlepoint(transition_state, calculators[0])
    return transition_state


def run_nebs(
    images: list[Atoms],
    neb_dir: str,
    fmax_relax: float,
    fmax_neb: float,
    num_images: int = 21,
) -> None:
    """Run NEB calculations on pairs of images in the given sequence and append
    the resulting transition states to the image sequence.
    """
    images.sort(key=lambda atoms: atoms.info["image_index"])
    for atoms in images:
        assert (atoms.info["image_index"] % 3) % 2 == 0, "NEB endpoints only"
    for a1, a2 in zip(images[:-1:2], images[1::2]):
        assert a2.info["image_index"] - a1.info["image_index"] == 2, [
            atoms.info["image_index"] for atoms in images
        ]
    calcs = [get_calculator() for _ in range(num_images)]
    iterable = (
        tqdm(images[::2], desc="Running NEBs") if HAS_TQDM else images[::2]
    )
    os.makedirs(neb_dir, exist_ok=True)
    to_append = []
    for initial, final in zip(iterable, images[1::2]):
        ts_index = initial.info["image_index"] + 1
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
        to_append.append(transition_state)
    images.extend(to_append)
    images.sort(key=lambda atoms: atoms.info["image_index"])


def run_equation_of_state(calc: Calculator, eos_dir: str) -> float:
    """Run equation of state calculation for bulk FCC Au and save results."""
    atoms = bulk("Au", "fcc", a=4.1)
    atoms.calc = calc
    os.makedirs(eos_dir, exist_ok=True)
    eos = calculate_eos(
        atoms,
        npoints=21,
        eps=0.1,
        trajectory=os.path.join(eos_dir, "bulk.traj"),
    )

    dft_volumes = [
        15.507224999999993,
        15.679527499999997,
        15.851829999999996,
        16.0241325,
        16.196435,
        16.368737499999984,
        16.54104,
        16.7133425,
        16.885644999999993,
        17.057947499999997,
        17.230249999999995,
        17.402552499999988,
        17.57485500000001,
        17.747157499999986,
        17.91945999999999,
        18.091762499999994,
        18.26406499999998,
        18.436367500000003,
        18.608669999999996,
        18.780972499999997,
        18.95327499999999,
    ]
    dft_energies = [
        -3.77157789,
        -3.79390665,
        -3.81311483,
        -3.82933066,
        -3.8428309,
        -3.85393527,
        -3.86268755,
        -3.8692239,
        -3.8737403500000003,
        -3.8764576299999995,
        -3.87725922,
        -3.8765631099999998,
        -3.8742243899999997,
        -3.87056654,
        -3.86551633,
        -3.8592750700000003,
        -3.8518873899999995,
        -3.84343887,
        -3.8340592499999997,
        -3.8237333999999996,
        -3.81257209,
    ]
    eos_dft = EquationOfState(dft_volumes, dft_energies)

    fig, ax = plt.subplots()
    _plot_eos(eos, ax, color="tab:blue", label="MLIP")
    _plot_eos(eos_dft, ax, color="tab:orange", label="DFT")
    ax.secondary_xaxis(
        "top",
        functions=(lambda v: (4 * v) ** (1 / 3), lambda a: a**3 / 4),
    ).set_xlabel("Lattice constant [Å]")
    ax.set(
        xlabel="Volume [Å$^3$]",
        ylabel="Energy [eV]",
        title="Equation of State for bulk FCC Au",
    )
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(eos_dir, "eos.png"))
    plt.close(fig)

    return eos.e0


def _plot_eos(eos: EquationOfState, ax: Axes, color: str, label: str):
    if eos.v0 is None:
        eos.fit()

    x = np.linspace(np.min(eos.v), np.max(eos.v), 100)
    if eos.eos_string == "sj":
        y = eos.fit0(x ** -(1 / 3))
    else:
        y = eos.func(x, *eos.eos_parameters)

    label = f"{label} - {eos.eos_string}: E: {eos.e0:.3f} eV, V: {eos.v0:.3f} Å$^3$, B: {eos.B / units.kJ * 1.0e24:.3f} GPa"
    ax.plot(x, y, ls="-", color=color, label=label)
    ax.plot(eos.v, eos.e, ls="", marker="o", color=color)
    ax.axhline(eos.e0, ls="--", color=color)
    ax.axvline(eos.v0, ls="--", color=color)


def gather_results(images: Sequence[Atoms], max_size: int):
    """Gather results from the given sequence of images into a dictionary.
    Arrays are padded with NaNs to the given maximum size.
    """
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
        (len(images), max_size, 3), fill_value=np.nan, dtype=np.float32
    )
    forces = np.full(
        (len(images), max_size, 3), fill_value=np.nan, dtype=np.float32
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
        default=0.005,
        help="Maximum force convergence criterion for relaxations.",
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
        "--eos_dir",
        type=str,
        default="./eos",
        help="Directory to save equation of state data.",
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
    bulk_energy = run_equation_of_state(calc, args.eos_dir)
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
    np.savez_compressed(
        args.output,
        **gather_results(flattened_images, max_size),
        bulk_energy=bulk_energy,
    )


if __name__ == "__main__":
    main()
