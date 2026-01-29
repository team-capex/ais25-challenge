# Scattering profile calculation

This project does **scattering profile calculation**: it uses the **Debye scattering equation** to compare **experimental** X-ray scattering data (F(q) and G(r)) with **simulated** profiles from an atomic structure. The script computes the weighted profile R-factor (Rwp) and optionally plots experiment vs simulation.

> **Important:** The scattering parameters (q-range, r-range, damping, `biso`, etc.) used in the simulation are **fixed to match the experimental setup**. They are defined once in the code as `EXPERIMENT_SCATTERING_PARAMS` and **must not be changed** when comparing against the provided data. Changing them would make the comparison invalid.

## Project structure

```
AI4Science/
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

From the project root (`AI4Science/`):

```bash
python scatteringcalculator.py
```

- Uses **dataset index 0** by default (first dataset in `Fq_data.npy` and `Gr_data.npy`).
- Loads **structure** from `structures/structure.xyz`.
- Uses **fixed experiment-matched scattering parameters** (see [Scattering parameters](#scattering-parameters) — do not change).
- Prints scattering parameters, Rwp for F(q) and G(r), and opens a comparison plot.

### Command-line options

| Option | Description |
|--------|-------------|
| `--dataset N` | Use dataset index `N` (0, 1, 2, …) from the .npy files. |
| `--structure PATH` | Path to structure file (default: `structures/structure.xyz`). |
| `--fq PATH` | Path to F(q) .npy file (default: `data/Fq_data.npy`). |
| `--gr PATH` | Path to G(r) .npy file (default: `data/Gr_data.npy`). |
| `--no-plot` | Only compute and print Rwp; do not show the plot. |

**Examples:**

```bash
# Use the second dataset
python scatteringcalculator.py --dataset 1

# Compute Rwp only (e.g. for scripting)
python scatteringcalculator.py --no-plot

# Use your own structure and data
python scatteringcalculator.py --structure path/to/my.xyz --fq path/to/Fq.npy --gr path/to/Gr.npy
```

## Using the code as a module

You can call the main logic from another script:

```python
from pathlib import Path
from scatteringcalculator import run

# Default: dataset 0, experiment-matched scattering parameters
results = run(dataset_index=0, show_plot=True)

# Different structure or dataset
results = run(
    structure_path=Path("structures/other_structure.xyz"),
    dataset_index=1,
    show_plot=False,
)
print(results["rwp_Fq"], results["rwp_Gr"])
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
