Please make sure your model is compatible with the Atomic Simulation Environment (ASE) by implementing an ASE Calculator class (see [ASE documentation](https://ase-lib.org/ase/calculators/calculators.html#module-ase.calculators)). The test script relies on this interface to evaluate energies and forces with your model - these two properties are the only required outputs from your model for the tests. Note that your model should support both periodic and non-periodic boundary conditions.

All test are run through the provided script [run_tests.py](./run_tests.py). Please redefine the function [get_calculator](./run_tests.py#L25) to return an ASE Calculator instance running your ML model.

To test your implementation, a [dummy test set](../data/dummy_test.traj) is provided along with [reference dft data](../data/dummy_results_dft.npz). You can run the tests on this dummy data set as follows:

```bash
$ python run_tests.py ../data/dummy_test.traj
$ python evaluate_model.py ../data/results.npz ../data/dummy_results_dft.npz
```

This will run all 4 tests on the dummy data set and save the predicted positions, energies and forces to `../data/results.npz`. If [tqdm](https://tqdm.github.io/) is installed, progress bars will be shown during the tests. The evaluation script will then compare your model predictions to the reference DFT data and print the losses for each subtest, corresponding to the loss metrics defined [here](../README.md#tests-to-perform). The output should look something like this:

```
Test: locality
  energy: 0.000100 eV/atom
  force: 0.000100 eV/Å
Test: relaxation
  energy: 0.001749 eV/atom
  rmsd: 0.024573 Å
Test: neb
  reaction_energy: 0.043661 eV
  activation_energy: 0.033837 eV
  rmsd_ts: 0.040639 Å
  rmsd_ends: 0.040498 Å
Test: single_point
  energy: 0.430836 eV/atom
  force: 0.069341 eV/Å
```

For sanity checks, the logs and trajectories of geometry optimizations and NEBs are saved in the `./relax/` and `./neb/` folders, respectively. The trajectories can be visualized with ASE’s `ase gui` from the command line:

```bash
$ ase gui relax/*.traj
$ ase gui neb/*climb.traj
```

It is also possible to change the convergence criteria and number of NEB images when running the tests.

```bash
$ python run_tests.py ../data/dummy_test.traj --fmax_relax 0.01 --fmax_neb 0.05 --num_images 7
```

You are free to change the script [run_tests.py](./run_tests.py) as needed to ensure compatibility with your model. The only requirement is that the NumPy arrays saved to the .npz output file is in the expected format. It needs to be readable by the evaluation script [evaluate_model.py](./evaluate_model.py), otherwise your submission may not be accepted.

The script has been tested using the following software versions:
- python 3.11.3
- ase 3.25.0
- numpy 1.25.1
- scipy 1.11.1
- tqdm 4.66.5