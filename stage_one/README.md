# Instructions for Stage 1 - New Deadline: January 12th, 2026

A training set of 1000 Au nanoparticles (NP) and low Miller-index surfaces are available [here](https://doi.org/10.11583/DTU.30480380).
This dataset can be used to fine-tune existing machine learning or foundation models to enable fast prediction of NP properties, such as energy and forces, with DFT-level accuracy (VASP, PBE, D3).

At the end of Stage 1 of the competition, each team must
1. Submit a hash of their final model in an e-mail to data-capex@dtu.dk no later than **January 12th, 2026, at noon, 12:00 (CET)**. This ensures that the test results can later be reproduced on the exact same model if necessary. Please use a SHA256 hash of the saved model file.
```bash
$ sha256sum model.pt
```
2. From the same e-mail address, submit the output file `data/results.npz` from the tests described below in an e-mail to data-capex@dtu.dk no later than **Januray 15th, 2026, at noon, 12:00 (CET)**. Please use the provided Python script to generate the output file in the expected format. See the [README](./scripts/README.md).
```bash
$ python run_tests.py ../data/test.traj
```

The tasks will be released on January 14, 2026, at noon, 12:00 (CET), i.e. pushed to this GitHub repository, and need to be completed within the following 24 hours. The time limit eliminates the possibility to cheat by using DFT directly. The tasks should be straightforward to complete with the provided scripts (see more below).

After the submission deadline, the three best teams will be crowned (see [scoring system](#scoringranking-models) below) and get to move on to Stage 2 of the competition.

There are 4 tests that measure different aspects of the model’s performance with one or more subtests, e.g. energy and interatomic force prediction. Each subtest is given a score, such that all models are compared across 10 metrics. 

## Tests to perform
Here is a brief description of the tasks that need to be performed.
The tests are described below along with the respective metric $\mathcal{L}$ for each subtest.
Root mean squared error (RMSE) metrics are detailed in the [Metrics](#metrics) section below.
Predicted labels are denoted with a hat, e.g. $\hat{E}$ for predicted energy, while true/reference labels are denoted without a hat, e.g. $E$ for reference energy.

1. *No fictitious, long-range interactions.* 
Two identical, uncharged nanoparticles placed far apart (50 Å) should not interact with each other.
The interatomic forces within each nanoparticle should be identical (within numerical precision) to an isolated nanoparticle.
Likewise, the total energy of the two-particle system should be twice the energy of an isolated nanoparticle.
In the following, $`\hat{E}_\mathrm{NP, \, isolated}`$ and $`\hat{F}_\mathrm{NP, \, isolated}`$ are the energy and forces of an isolated nanoparticle, $`\hat{F}_\mathrm{NP}`$ are the forces within a single nanoparticle in the two-particle system that has total energy $`\hat{E}_\mathrm{NP+NP}`$.
    1. $`\mathcal{L} = \max(\max \lvert \hat{E}_\mathrm{NP+NP} - 2 \hat{E}_\mathrm{NP, \, isolated} \rvert, 10^{-4} \ \mathrm{eV/atom})`$
    2. $`\mathcal{L} = \max(\max \lvert \hat{F}_\mathrm{NP} - \hat{F}_\mathrm{NP, \, isolated} \rvert, 10^{-4} \ \mathrm{eV/Å})`$

    If the maximum force or energy difference (element-wise) is below the tolerance of $10^{-4}$ eV/Å or eV/atom, respectively, the model gets full points.

2. *Geometry optimization.*
Predict the energy difference between different relaxed nanoparticle polymorphs. Starting from two different initial polymorphs A and B, relax the structures to A’ and B’. Then compare the energy and atomic position difference to the reference DFT calculation that found relaxed structures A’’ and B’’.
    1. $`\mathcal{L} = \mathrm{RMSE_E}(\hat{E}_\mathrm{B'} - \hat{E}_\mathrm{A'}, E_\mathrm{B''} - E_\mathrm{A''})`$
    2. $`\mathcal{L} = \frac{1}{2} \left( \mathrm{RMSD}(\hat{R}_\mathrm{A'}, R_\mathrm{A''}) + \mathrm{RMSD}(\hat{R}_\mathrm{B'}, R_\mathrm{B''}) \right)`$

3. *Nudged elastic band (NEB) between nanoparticle polymorphs.* Predict the reaction energy, activation energy, and atomic positions of the initial, final and transition states along the NEB path between two nanoparticle polymorphs *initial* and *final*.
    1. $`\mathcal{L} = \mathrm{RMSE_E}(\hat{E}_\mathrm{final} - \hat{E}_\mathrm{initial}, E_\mathrm{final} - E_\mathrm{initial})`$
    2. $`\mathcal{L} = \mathrm{RMSE_E}(\hat{E}_\mathrm{TS} - \hat{E}_\mathrm{initial}, E_\mathrm{TS} - E_\mathrm{initial})`$
    3. $`\mathcal{L} = \frac{1}{2} \left( \mathrm{RMSD}(\hat{R}_\mathrm{initial}, R_\mathrm{initial}) + \mathrm{RMSD}(\hat{R}_\mathrm{final}, R_\mathrm{final}) \right)`$
    4. $`\mathcal{L} = \mathrm{RMSD}(\hat{R}_\mathrm{TS}, R_\mathrm{TS})`$

    In this test, the energies are extensive, i.e. in eV, not eV/atom.

4. *Extrapolation to larger nanoparticles.* Predict the energy and forces of larger nanoparticles up to 10 nm in diameter that share the local motifs of the provided training set.
    1. $`\mathcal{L} = \mathrm{RMSE_E}(\hat{E}, E)`$
    2. $`\mathcal{L} = \mathrm{RMSE_F}(\hat{F}, F)`$

All relevant structures will be provided in the extended XYZ format. Each task is accompanied by a Python script (will be provided soon) that runs the test and saves the outputs in the expected format. These scripts rely on the Atomic Simulation Environment ([ASE](https://ase-lib.org/))  Python library. To use the scripts, the ML model should interface with ASE through an [ASE Calculator class](https://ase-lib.org/ase/calculators/calculators.html#module-ase.calculators)). For comparable results to the reference data, this might be especially important for tasks 2 and 3 that use specific optimization routines. Note that only energy and force evaluation is required to perform these tasks.

## Scoring/ranking models
Following the given tests, each team will be ranked according to a niching protocol. This means all tests and subtests within the same test are weighted equally, and each subtest score is ranked relative to all other submissions. The subtest score is given by the placement on the subtest leaderboard, i.e. a model that performs third best on a particular subtest will be given a score of 3 on that subtest. Note that everyone performing under the tolerance level in subtest 1 will be sharing the 1st spot in that subtest. Ranking in a whole test follows the same principle, i.e. the best combined score across all subtests in a test is ranked first in that test. The lowest combined score across the 4 tests is chosen as the best model.

## Metrics
For reference, we define energy and force root mean squared error (RMSE) functions and root mean squared deviation (RMSD) of atomic positions.

In the following, $B$ is the number of test samples in the subtest, while $N_b$, $E_b$ are the number of atoms and the energy of sample $`b \in \{1, ... ,B\}`$. Furthermore, force component $`i \in \{x, y, z\}`$ of atom $`n \in \{1, ..., N_b\}`$ of sample $b$ is written as $F_{b,ni}$. Likewise, $R_{b,ni}$ is position component $i$ of atom $n$ of sample $b$.

```math
\begin{aligned}
    \mathrm{RMSE_E}(\hat{E}, E) &= \sqrt{\frac{1}{B} \sum_{b=1}^{B} \left( \frac{\hat{E}_b - E_b}{N_b} \right)^2 } \\

    \mathrm{RMSE_F}(\hat{F}, F) &= \sqrt{\frac{1}{B} \sum_{b=1}^{B} \frac{1}{3N_b}\sum_{n=1}^{N_b} \sum_{i \in \{x,y,z\}} \left( \hat{F}_{b,ni} - F_{b,ni} \right)^2 } \\
    
    \mathrm{RMSD}(\hat{R}, R) &= \sqrt{\frac{1}{B} \sum_{b=1}^{B} \frac{1}{3N_b} \sum_{n=1}^{N_b} \sum_{i \in \{x,y,z\}} \left( \hat{R}_{b,ni} - R_{b,ni} \right)^2 }
\end{aligned}
```

Note that $E,F,R$ are the reference labels and $\hat{E}, \hat{F}, \hat{R}$ are the predicted labels.
