# AIS25 AI4Materials Nanoparticle (NP) Challenge

This repository contains materials for the AIS25 AI4Materials Nanoparticle Challenge, organised in two stages.

## Stage 1 — `stage_one/`

**ML models for nanoparticle energy and forces.**  
Train or fine-tune models on a dataset of Au nanoparticles and low Miller-index surfaces to predict energy and forces with DFT-level accuracy (VASP, PBE, D3). Stage 1 is evaluated on tests for: no fictitious long-range interactions, geometry optimisation, NEB between polymorphs, and extrapolation to larger nanoparticles. The best teams proceed to Stage 2.

See **[stage_one/README.md](stage_one/README.md)** for instructions, deadlines, tests, and scoring.

## Stage 2 — `stage_two/`

**Scattering profile calculation.**  
Compare experimental X-ray scattering data (F(q) and G(r)) with simulated profiles from atomic structures using the **Debye scattering equation**. The code computes the weighted profile R-factor (Rwp) and can plot experimental vs simulated data. In the calculation of the simulated scattering patterns, the scattering parameters are fixed to match the experiment and must not be changed when comparing against the provided experimental data.

See **[stage_two/README.md](stage_two/README.md)** for setup, usage, and references.
