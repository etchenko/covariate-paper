# Causal Inference Project

An investigation into adjustment set selection and post-selection inference in causal effect estimation

## Overview

This project explores the validity of adjustment sets selected through conventional causal methods in differing cases. It compares different approaches over different types of DAGS, to investigate the relationship between adjustment set selection and post-selection inference

## Project Structure

```
.
├── README.md                    # This file
├── simulations.py              # Main simulation runner
├── causal_estimator.py         # Core causal estimation logic
├── feature_selector.py         # Feature selection implementations
├── graphs.py                   # DAG generation functions
├── gcm.py                      # Generalized Covariance Measure implementation
├── indtests.py                 # Independence testing functions
├── run.sh                      # SLURM script for single simulation
├── run_all.sh                  # SLURM script for parallel array jobs
├── env/                        # Virtual environment
└── old_code/                   # Archived previous implementations
```

## Installation

### Prerequisites

- Python 3.13 or higher
- Virtual environment (recommended)

### Setup

1. Clone the repository:
```bash
git clone <repository-url>
cd "Covariate paper"
```

2. Create and activate a virtual environment:
```bash
python3.13 -m venv env
source env/bin/activate
```

3. Install required packages:
```bash
pip install numpy pandas scikit-learn statsmodels tqdm boruta multiprocess
pip install causal-learn npeet-plus
```

## Quick Start

### Running Local Simulations

Run all linear model simulations:
```bash
python simulations.py linear
```

Run all non-linear (neural network) simulations:
```bash
python simulations.py nonlinear
```

### Running on HPC Cluster (SLURM)

For linear simulations:
```bash
sbatch run.sh
```

For parallel non-linear simulations:
```bash
sbatch run_all.sh
```

## Methodology

### Causal Estimation Methods

1. **Backdoor Adjustment**: Direct outcome regression adjusting for selected covariates
   - Estimates E[Y|A=1,Z] - E[Y|A=0,Z]

2. **Inverse Propensity Weighting (IPW)**: Reweights observations by inverse probability of treatment
   - Requires propensity score model P(A|Z)

3. **Augmented IPW (AIPW)**: Combines outcome regression and IPW for double robustness
   - Remains consistent if either the outcome or propensity model is correct

4. **Double Machine Learning (DML)**: Sample-splitting approach for nuisance parameter estimation
   - Controls for overfitting in feature selection

### Feature Selection Strategies

- **Treatment-based**: Select covariates predicting treatment assignment
- **Outcome-based**: Select covariates predicting the outcome
- **Union**: Use all features selected by either criterion
- **Intersection**: Use only features selected by both criteria
- **Different**: Use treatment features for propensity model, outcome features for outcome model
- **All/None**: Use all or no covariates (baselines)

### DAG Structures

**Graph 1**: Low-dimensional mediation structure
- 2 treatment confounders (W1, W2)
- 3 outcome predictors (O1, O2, O3)
- 1 mediator (M)
- 1 instrumental variable (W)

**Graph 2**: High-dimensional sparse structure
- 66+ random covariates from multivariate normal distribution
- Randomly assigned as treatment confounders, outcome predictors, or both
- 1 instrumental variable (W)

**Graph 3**: Outcome adjustment structure
- 4 latent confounders (U1, U2, U3, U4)
- 2 observed confounders (C1, C2)
- 1 instrumental variable (O1)

### Validation Testing

The Generalized Covariance Measure (GCM) tests whether the selected adjustment set satisfies the backdoor criterion by checking conditional independence:
- H₀: Y ⊥ W | (A, Z)
- Where W is an instrumental variable, A is treatment, Z is the adjustment set
- Rejection indicates the adjustment set is insufficient


## Acknowledgments

This project implements methods from:
- VanderWeele, T.J. (2019). Principles of confounder selection
- Chernozhukov et al. (2018). Double/debiased machine learning
- Shah and Peters (2020). The hardness of conditional independence testing
- Malinsky (2024). Generalized covariance measure extensions

