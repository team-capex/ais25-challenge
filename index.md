[Stage 1](#stage-1) | [Stage 2](#stage-2) | [Winner and prize](#winner-and-prize)

The Pioneer Center CAPeX, the Novo Nordisk Foundation and the Danish Center for AI Innovation (DCAI) is launching the two-stage "AIS25 AI4Materials Nanoparticle (NP) Challenge" machine learning competition for prediction and synthesis of nanoparticle catalysts.

At the launch of the challenge (November 4th, 11:00 am (CET)), a training set of 1.000 Au nanoparticles and low Miller-index surfaces will be made available [here](https://doi.org/10.11583/DTU.30480380). This dataset can be used to fine-tune existing machine learning or foundation models to enable fast prediction of NP properties, such as energy and forces, with DFT-level accuracy (VASP, PBE, D3).

## Stage 1

To enter Stage 1 of the competition, participating teams must submit a hash of their final model to data-capex@dtu.dk no later than December 12th at, noon, 12:00 (CET). On December 15th at noon, 12:00 (CET), participating teams will receive a complex NP test set and must submit their reply to data-capex@dtu.dk within 24:00 hours. Detailed instructions and scoring conditions for stage 1 are available [here](https://github.com/team-capex/ais25-challenge/tree/main/stage_one).

## Stage 2

The top three teams* following Stage 1 will be announced on January 9th, and qualify for the Stage 2, where they will be onboarded on the Danish Center for AI Innovation (DCAI) NVIDIA Superpod Gefion for training/fine-tuning of their model on a new >12.000.000 structure ab initio solid-liquid dataset (VASP, PBE D3) with solvated nanoparticles, Pt, Au, and Cu surfaces, as well as cat- and anions. During the competition, the teams will have access to the dataset before it is publicly released after the competition.

The DCAI Gefion SuperPOD has 1.528 DGX H100 GPUs plus a number of new B300 GPUs, and one of the world’s fastest INFINYBAND network and disk I/O. Compute credits will be provided by the Pioneer Center CAPeX and the Novo Nordisk Foundation (NNF) through grant #NNF25OC0105158 for training and inference on up to 512 H100s, as well as the new B300 GPUs.

The three teams will receive two sets of unpublished experimental synthesis conditions for Au nanoparticles, following the solution-based synthesis procedure described [here](https://doi.org/10.48550/arXiv.2505.13571), and must submit the three most stable atomic structures for each set of synthesis conditions, plus the total scattering and pair distribution function (PDF) (scripts will be provided), before February 20th, 2026. This part of the competition is high risk, and modifications to the scoring metric may be required.

## Winner and prize

The winner will be the team that is able to predict the best goodness-of-fit to the experimental scattering data as described in equation 4 of the [paper](https://doi.org/10.48550/arXiv.2505.13571). The experimentally measured scattering data is revealed after the competition ends.

Prize: On February 27th, 2026, the winner will be announced. The winning team will receive additional compute credits and will be invited to propose a specific nanoparticle structure to be synthesized during autonomous laboratory operation at the MAX IV Synchrotron in Lund.

**) To qualify for Stage 2, teams must agree to make their code openly available and participate in a joint scientific publication.*
