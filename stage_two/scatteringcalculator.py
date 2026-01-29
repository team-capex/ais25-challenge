"""
Scattering profile calculation: compare experimental F(q) and G(r) data
with simulated profiles from an atomic structure using the Debye scattering equation.

Usage:
    python scatteringcalculator.py                    # run with defaults
    python scatteringcalculator.py --dataset 1       # use dataset index 1
    python scatteringcalculator.py --no-plot         # compute Rwp only, no plot
"""

from pathlib import Path
import argparse
import numpy as np
import matplotlib.pyplot as plt
from ase.io import read

# Optional: GPU acceleration if available (used by DebyeCalculator)
try:
    import torch
    _device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
except ImportError:
    _device = None

from debyecalculator import DebyeCalculator


# -----------------------------------------------------------------------------
# Default paths (relative to this script's directory)
# -----------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent
DATA_DIR = PROJECT_ROOT / "data"           # Experimental F(q) and G(r)
SYNTHESIS_PARAMETERS_DIR = PROJECT_ROOT / "synthesis_parameters"
STRUCTURES_DIR = PROJECT_ROOT / "structures"  # Atomic structures (e.g. XYZ)
DEFAULT_FQ_FILE = DATA_DIR / "Fq_data.npy"
DEFAULT_GR_FILE = DATA_DIR / "Gr_data.npy"
DEFAULT_STRUCTURE_FILE = STRUCTURES_DIR / "structure.xyz"

# Scattering parameters matching the experiment (do not change).
# See README "Scattering parameters" for full definitions.
#   qmin, qmax, qstep [Å⁻¹]: q-range and step for F(q).
#   qdamp [Å⁻¹]: instrumental damping in q-space.
#   rmin, rmax, rstep [Å]: r-range and step for G(r) (pair distribution function).
#   biso [Å²]: isotropic atomic displacement (Debye–Waller), thermal smearing.
EXPERIMENT_SCATTERING_PARAMS = {
    "qmin": 0.5,
    "qmax": 15.0,
    "qstep": 0.01,
    "qdamp": 0.0274,
    "rmin": 0.0,
    "rmax": 30.0,
    "rstep": 0.01,
    "biso": 0.3,
}


def calculate_scattering_pattern(
    cluster,
    qmin,
    qmax,
    qstep,
    qdamp,
    rmin,
    rmax,
    rstep,
    biso=0.3,
):
    """
    Compute F(q) and G(r) for an atomic cluster using the Debye scattering equation.

    Parameters
    ----------
    cluster : ase.Atoms
        Atomic structure (e.g. from an XYZ file).
    qmin, qmax, qstep : float
        q-range and step for the scattering pattern (Å⁻¹).
    qdamp : float
        Damping factor for the q-range.
    rmin, rmax, rstep : float
        r-range and step for the pair distribution function (Å).
    biso : float
        Isotropic atomic displacement parameter (Å²). Default 0.3.

    Returns
    -------
    Q : ndarray
        q values (Å⁻¹).
    F : ndarray
        F(q) intensity.
    r : ndarray
        r values (Å).
    G : ndarray
        G(r) intensity.
    """
    calc = DebyeCalculator(
        qmin=qmin,
        qmax=qmax,
        qstep=qstep,
        qdamp=qdamp,
        rmin=rmin,
        rmax=rmax,
        rstep=rstep,
        biso=biso,
    )
    r, Q, _, _, F, G = calc._get_all(structure_source=cluster)
    return Q, F, r, G


def calculate_loss(x_exp, x_sim, I_exp, I_sim, loss_type="rwp"):
    """
    Compare experimental and simulated patterns and compute a loss.

    Parameters
    ----------
    x_exp, x_sim : ndarray
        x-axis values (q or r) for experiment and simulation.
    I_exp, I_sim : ndarray
        Intensity values for experiment and simulation.
    loss_type : str
        One of "rwp" (weighted profile R-factor), "mae", "mse".

    Returns
    -------
    loss : float
        The loss value.
    I_sim_interp : ndarray
        Simulated intensity interpolated onto the experimental x-grid.
    """
    I_sim_interp = np.interp(x_exp, x_sim, I_sim)
    diff = I_exp - I_sim_interp

    if loss_type == "rwp":
        loss = np.sqrt(np.sum(diff**2) / np.sum(I_exp**2))
    elif loss_type == "mae":
        loss = np.mean(np.abs(diff))
    elif loss_type == "mse":
        loss = np.mean(diff**2)
    else:
        raise ValueError(f"Invalid loss_type: {loss_type}")

    return loss, I_sim_interp


def load_experimental_data(fq_path, gr_path, dataset_index=0):
    """
    Load experimental F(q) and G(r) from .npy files.

    Files are expected to be arrays of shape (n_datasets, n_points, 2)
    with columns [x, intensity].

    Parameters
    ----------
    fq_path : path-like
        Path to Fq_data.npy.
    gr_path : path-like
        Path to Gr_data.npy.
    dataset_index : int
        Which dataset to use (0, 1, 2, ...). Default 0.

    Returns
    -------
    q_exp, Fq_exp : ndarray
        Experimental q and F(q).
    r_exp, Gr_exp : ndarray
        Experimental r and G(r).
    """
    Fq_data = np.load(fq_path)
    Gr_data = np.load(gr_path)
    Fq_exp = Fq_data[dataset_index][:, 1]
    q_exp = Fq_data[dataset_index][:, 0]
    Gr_exp = Gr_data[dataset_index][:, 1]
    r_exp = Gr_data[dataset_index][:, 0]
    return q_exp, Fq_exp, r_exp, Gr_exp


def run(
    structure_path=DEFAULT_STRUCTURE_FILE,
    fq_path=DEFAULT_FQ_FILE,
    gr_path=DEFAULT_GR_FILE,
    dataset_index=0,
    params=None,
    show_plot=True,
):
    """
    Load data and structure, compute simulated pattern, compare with experiment.

    Parameters
    ----------
    structure_path : path-like
        Path to atomic structure (e.g. structure.xyz).
    fq_path, gr_path : path-like
        Paths to F(q) and G(r) .npy data.
    dataset_index : int
        Index of the dataset to use in the .npy arrays.
    params : dict, optional
        Scattering parameters; defaults to EXPERIMENT_SCATTERING_PARAMS (match experiment; do not change).
    show_plot : bool
        If True, show comparison plot; otherwise only print Rwp.

    Returns
    -------
    dict
        Contains "rwp_Fq", "rwp_Gr", "q_exp", "Fq_exp", "r_exp", "Gr_exp",
        "q_sim", "Fq_sim", "r_sim", "Gr_sim".
    """
    if params is None:
        params = EXPERIMENT_SCATTERING_PARAMS.copy()

    if _device is not None:
        print(f"Using device: {_device}")

    # Load experimental data
    q_exp, Fq_exp, r_exp, Gr_exp = load_experimental_data(
        fq_path, gr_path, dataset_index
    )
    print(f"Loaded experimental data (dataset index {dataset_index}) from {fq_path.name}, {gr_path.name}")

    # Load structure
    cluster = read(str(structure_path))
    print(f"Loaded structure: {structure_path.name} ({len(cluster)} atoms)")

    # Scattering parameters
    print("Scattering parameters:")
    for k, v in params.items():
        print(f"  {k}: {v}")

    # Simulate F(q) and G(r)
    q_sim, Fq_sim, r_sim, Gr_sim = calculate_scattering_pattern(
        cluster,
        params["qmin"],
        params["qmax"],
        params["qstep"],
        params["qdamp"],
        params["rmin"],
        params["rmax"],
        params["rstep"],
        params["biso"],
    )

    # Normalise for comparison
    Fq_exp_n = Fq_exp / np.max(Fq_exp)
    Gr_exp_n = Gr_exp / np.max(Gr_exp)
    Fq_sim_n = Fq_sim / np.max(Fq_sim)
    Gr_sim_n = Gr_sim / np.max(Gr_sim)

    # Rwp
    rwp_Fq, _ = calculate_loss(q_exp, q_sim, Fq_exp_n, Fq_sim_n, loss_type="rwp")
    rwp_Gr, _ = calculate_loss(r_exp, r_sim, Gr_exp_n, Gr_sim_n, loss_type="rwp")
    print(f"Rwp F(q): {rwp_Fq:.4f}")
    print(f"Rwp G(r): {rwp_Gr:.4f}")

    if show_plot:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        ax1.plot(q_sim, Fq_sim_n, label=f"Simulated (Rwp = {rwp_Fq:.3f})")
        ax1.plot(q_exp, Fq_exp_n, label="Experimental")
        ax1.set_xlabel(r"q (Å$^{-1}$)")
        ax1.set_ylabel("F(q)")
        ax1.set_yticks([])
        ax1.legend()
        ax2.plot(r_sim, Gr_sim_n, label=f"Simulated (Rwp = {rwp_Gr:.3f})")
        ax2.plot(r_exp, Gr_exp_n, label="Experimental")
        ax2.set_xlabel("r (Å)")
        ax2.set_ylabel("G(r)")
        ax2.set_yticks([])
        ax2.legend()
        plt.tight_layout()
        plt.show()

    return {
        "rwp_Fq": rwp_Fq,
        "rwp_Gr": rwp_Gr,
        "q_exp": q_exp,
        "Fq_exp": Fq_exp_n,
        "r_exp": r_exp,
        "Gr_exp": Gr_exp_n,
        "q_sim": q_sim,
        "Fq_sim": Fq_sim_n,
        "r_sim": r_sim,
        "Gr_sim": Gr_sim_n,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Scattering profile calculation: compare experimental F(q)/G(r) with simulated profiles using the Debye scattering equation."
    )
    parser.add_argument(
        "--dataset",
        type=int,
        default=0,
        help="Index of dataset in Fq_data.npy / Gr_data.npy (default: 0)",
    )
    parser.add_argument(
        "--structure",
        type=Path,
        default=DEFAULT_STRUCTURE_FILE,
        help=f"Path to structure file (default: {DEFAULT_STRUCTURE_FILE})",
    )
    parser.add_argument(
        "--fq",
        type=Path,
        default=DEFAULT_FQ_FILE,
        help=f"Path to F(q) data .npy (default: {DEFAULT_FQ_FILE})",
    )
    parser.add_argument(
        "--gr",
        type=Path,
        default=DEFAULT_GR_FILE,
        help=f"Path to G(r) data .npy (default: {DEFAULT_GR_FILE})",
    )
    parser.add_argument(
        "--no-plot",
        action="store_true",
        help="Only compute and print Rwp, do not show plot",
    )
    args = parser.parse_args()

    run(
        structure_path=args.structure,
        fq_path=args.fq,
        gr_path=args.gr,
        dataset_index=args.dataset,
        show_plot=not args.no_plot,
    )


if __name__ == "__main__":
    main()
