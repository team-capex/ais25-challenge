# Instructions for Stage 2 — Structure prediction from synthesis parameters

Stage 2 is open to the best teams from Stage 1. Your task is to build a **model that predicts the atomic structure that will stabilise** under given synthesis conditions.

## Task

- **Input:** Synthesis/experiment parameters (e.g. from `synthesis_parameters/parameters.npy`), corresponding to the conditions under which a sample was prepared.
- **Output:** A predicted atomic structure (e.g. in XYZ or other ASE-supported format) that you expect to stabilise under those conditions.

You must develop a model that, given the synthesis parameters for a dataset, predicts the atomic structure. The quality of your prediction is evaluated by comparing **experimental** X-ray scattering data (F(q) and G(r)) with **simulated** scattering profiles computed from your predicted structure using the Debye scattering equation.

## Evaluation

Submissions are evaluated using the **provided experimental data** (in `data/Fq_data.npy` and `data/Gr_data.npy`) versus the scattering patterns simulated from your predicted structure. The metrics are:

1. **Rwp** (weighted profile R-factor) — for F(q) and for G(r), measuring how well your simulated profile fits the experiment.
2. **MSE** (mean squared error) — between the experimental and simulated scattering data.

Lower Rwp and lower MSE indicate better agreement; models are ranked according to these metrics (details will be specified at evaluation time). The provided `scatteringcalculator.py` script computes both Rwp and MSE for a given structure and dataset, so you can assess your predictions locally.

The **three datasets** in this repository are **open data** for you to develop and evaluate your model. In addition, **data not revealed here** will be used as part of the final assessment; your model will be evaluated on both the open and the held-out data.

## What you have

- **Experimental scattering data:** F(q) and G(r) for three open datasets (`data/`). These are for development and local evaluation.
- **Synthesis parameters:** Parameters associated with each dataset (`synthesis_parameters/parameters.npy`).
- **Scattering calculator:** A script that simulates F(q) and G(r) from an atomic structure and compares them to the experiment using fixed, experiment-matched scattering parameters. Use it to compute both Rwp and MSE for your predicted structures (see [Getting Rwp and MSE](#getting-rwp-and-mse)).

You must **not** change the scattering parameters (q-range, r-range, damping, `biso`, etc.) used in the calculator when comparing against the provided data; they are fixed to match the experimental setup.

---

## Scattering profile calculation (tooling)

This folder provides **scattering profile calculation**: the **Debye scattering equation** is used to compare experimental F(q) and G(r) with simulated profiles from an atomic structure. The script computes Rwp and MSE and can plot experiment vs simulation.

> **Important:** The scattering parameters used in the simulation are **fixed to match the experimental setup**. They are defined in the code as `EXPERIMENT_SCATTERING_PARAMS` and **must not be changed** when comparing against the provided data.

## Project structure

```
stage_two/
├── README.md                 # This file
├── requirements.txt          # Python dependencies
├── scatteringcalculator.py   # Main script
├── data/                     # Experimental scattering data
│   ├── Fq_data.npy           # F(q) – multiple datasets
│   └── Gr_data.npy           # G(r) – multiple datasets
├── synthesis_parameters/     # Synthesis/experiment parameters
│   └── parameters.npy        # Metadata (optional; load manually if needed)
└── structures/               # Atomic structures
    └── structure.xyz         # Structure used for Debye simulation (e.g. cluster)
```

### Folder contents

| Folder | Contents |
|--------|----------|
| **data/** | F(q) and G(r) data: shape `(n_datasets, n_points, 2)`, columns `[q or r, intensity]` (intensity arbitrary scale). |
| **synthesis_parameters/** | Synthesis/experiment parameters (e.g. `parameters.npy`). Load with `np.load('synthesis_parameters/parameters.npy', allow_pickle=True)` if needed. |
| **structures/** | Atomic structures (XYZ or other ASE-supported). Default: `structure.xyz`. |

## Setup

1. **Clone or download** this folder.

2. **Install [uv](https://docs.astral.sh/uv/)** (if needed):

   ```bash
   curl -LsSf https://astral.sh/uv/install.sh | sh
   ```

3. **Create a virtual environment and install dependencies**:

   ```bash
   uv venv
   source .venv/bin/activate   # Linux/macOS
   # or: .venv\Scripts\activate   # Windows
   uv pip install -r requirements.txt
   ```

   This installs: `numpy`, `matplotlib`, `ase`, `debyecalculator`, and `torch`.  
   For GPU-accelerated Debye calculations, a CUDA-enabled PyTorch build is optional.

## How to run

From the project root (`stage_two/`):

```bash
python scatteringcalculator.py
```

- Uses **dataset index 0** by default (first dataset in `Fq_data.npy` and `Gr_data.npy`).
- Loads **structure** from `structures/structure.xyz`.
- Uses **fixed experiment-matched scattering parameters** (see [Scattering parameters](#scattering-parameters) — do not change).
- Prints scattering parameters, **Rwp** and **MSE** for F(q) and G(r), and opens a comparison plot.

### Command-line options

| Option | Description |
|--------|-------------|
| `--dataset N` | Use dataset index `N` (0, 1, 2, …) from the .npy files. |
| `--structure PATH` | Path to structure file (default: `structures/structure.xyz`). |
| `--fq PATH` | Path to F(q) .npy file (default: `data/Fq_data.npy`). |
| `--gr PATH` | Path to G(r) .npy file (default: `data/Gr_data.npy`). |
| `--no-plot` | Only compute and print Rwp and MSE; do not show the plot. |

**Examples:**

```bash
# Use the second dataset
python scatteringcalculator.py --dataset 1

# Compute Rwp and MSE only (e.g. for scripting)
python scatteringcalculator.py --no-plot

# Use your own structure and data
python scatteringcalculator.py --structure path/to/my.xyz --fq path/to/Fq.npy --gr path/to/Gr.npy
```

## Getting Rwp and MSE

Both **Rwp** (weighted profile R-factor) and **MSE** (mean squared error) are computed between the experimental and simulated F(q) and G(r) (after normalisation). You need both for evaluation.

### From the `run()` function

When you call `run()`, the returned dictionary contains `"rwp_Fq"`, `"rwp_Gr"`, `"mse_Fq"`, and `"mse_Gr"`:

```python
from pathlib import Path
from scatteringcalculator import run

results = run(dataset_index=0, show_plot=False)

# All four metrics
print("Rwp  F(q):", results["rwp_Fq"], "  G(r):", results["rwp_Gr"])
print("MSE  F(q):", results["mse_Fq"], "  G(r):", results["mse_Gr"])
```

### From `calculate_loss()` directly

If you have already computed experimental and simulated q/r and intensities (e.g. after normalising), you can get Rwp or MSE for a single profile with `calculate_loss()`:

```python
from scatteringcalculator import calculate_loss

# loss_type="rwp" or "mse"
rwp_value, I_sim_interp = calculate_loss(q_exp, q_sim, I_exp_norm, I_sim_norm, loss_type="rwp")
mse_value, I_sim_interp = calculate_loss(q_exp, q_sim, I_exp_norm, I_sim_norm, loss_type="mse")
```

Use the same normalisation as in `run()` (e.g. divide by max intensity) so that your metrics are comparable to the evaluation.

## Using the code as a module

You can call the main logic from another script:

```python
from pathlib import Path
from scatteringcalculator import run

# Default: dataset 0, experiment-matched scattering parameters
results = run(dataset_index=0, show_plot=True)

# Different structure or dataset; get both Rwp and MSE
results = run(
    structure_path=Path("structures/other_structure.xyz"),
    dataset_index=1,
    show_plot=False,
)
print("Rwp:", results["rwp_Fq"], results["rwp_Gr"])
print("MSE:", results["mse_Fq"], results["mse_Gr"])
```

## Scattering parameters

The scattering parameters (q-range, r-range, damping, `biso`, etc.) are **experiment-matched** and **must not be changed**; they are in `scatteringcalculator.py` as `EXPERIMENT_SCATTERING_PARAMS`. Do not edit or override unless using a different experiment.

| Parameter | Value |
|-----------|--------|
| qmin, qmax, qstep | 0.5, 15.0, 0.01 |
| qdamp | 0.0274 |
| rmin, rmax, rstep | 0, 30.0, 0.01 |
| biso | 0.3 |

For definitions of F(q), G(r), total scattering, PDF, and related theory, see [Anker et al., *Autonomous nanoparticle synthesis by design*](https://arxiv.org/abs/2505.13571).

## References

- **Anker et al.** [Autonomous nanoparticle synthesis by design](https://arxiv.org/abs/2505.13571) (arXiv:2505.13571) — total scattering, PDF, and structure-targeted synthesis.
- **DebyeCalculator**: [PyPI](https://pypi.org/project/debyecalculator), [GitHub](https://github.com/FrederikLizakJohansen/DebyeCalculator) — Debye scattering equation.
