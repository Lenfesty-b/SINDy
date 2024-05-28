# Uncovering dynamical equations of stochastic decision models using data-driven SINDy algorithm
# Decision Models Simulation

## Introduction

This project simulates trials using four different decision models (DDM, LCA-DDM, LCA, and NLB) and fits SINDy models to the simulated data. Each model can run single trials and trial-averaged models using the derived average coefficients. The project includes Python scripts and Jupyter notebooks to demonstrate how to run these simulations and analyze the results.

## Requirements

- Python 3.8
- Required packages and versions are listed in `requirements.txt`.

## Directory Structure

- `codes/`: Contains the Python scripts for each model used and Jupyter notebooks demonstrating how to run single trials, multiple trials, and view the results.
    - `ddm_st.py`
    - `ddm_ave.py`
    - `lcaddm_st.py`
    - `lcaddm_ave.py`
    - `lca_st.py`
    - `lca_ave.py`
    - `nlb_st.py`
    - `nlb_ave.py`
    - `run_ddm.ipynb`
    - `run_lcaddm.ipynb`
    - `run_lca.ipynb`
    - `run_nlb.ipynb`
- `requirements.txt`: Lists the Python packages and versions required for the project.
- `README.md`: This file.

## Usage

To run the simulations, use the provided Jupyter notebooks in the `codes/` directory. Each notebook corresponds to one of the four decision models and includes the following steps and must be run sequentially:

1. **Single Trial Simulation**:
    - Demonstrates a sample decision trial and its output

2. **Multiple Single Trials**:
    - Run the simulations for multiple trials using the single trial Python script.
    - Store the coefficients returned from the simulations to be used for average model.

3. **Trial Averaged Model**:
    - Use the average coefficients obtained from the multiple single trials.
    - Run the trial-averaged model using the trial average model.
    - Compare the performance of the trial-averaged model to the original decision over many trials.

### Example Workflow

#### Single Trial Simulation

In the `run_ddm.ipynb` notebook for example, you will:
- Import the necessary functions from `ddm_st.py`.
- Define parameters for the trial.
- Run the `simulate_trial` function to simulate a single trial.
- Print and plot the results to visualize the decision variable trajectories.

#### Multiple Single Trials

In the same notebook, you will:
- Define parameters for running multiple trials.
- Run the `sessionDDM` function to simulate multiple single trials and fit the SINDy model.
- Print a summary of the results and plot sample trials to compare the DDM and SINDy models.

#### Trial Averaged Model

Also in the same notebook, you will:
- Use the coefficients obtained from multiple single trials.
- Run the `sessionDDM` function from `ddm_ave.py` to simulate the trial-averaged model using the average SINDy coefficients.
- Print a summary of the results and plot sample trials to compare the trial-averaged model with the original DDM model.

Repeat these steps for `run_lcaddm.ipynb`, `run_lca.ipynb`, and `run_nlb.ipynb`.

## Contact

For any questions or further assistance, please contact Brendan Lenfesty at lenfesty-b@ulster.ac.uk



