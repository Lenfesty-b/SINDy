import numpy as np
import scipy
import pysindy as ps
import pandas as pd

def simulate_trial(t, dt, tauX, epsilon, b, c, z):
    """
    Simulates a trial for the NLB model.

    Parameters:
        t (array-like): Time array.
        dt (float): Time step.
        tauX (float): Time constant.
        epsilon (float): Non-biased stimulus input.
        b (float): Biased stimulus input.
        c (float): Size of the noise.
        z (float): Decision threshold.

    Returns:
        x (array-like): Trajectory of the decision variable.
        trial_decision_time (float): Time at which the decision was made.
        trial_choice (int): Indicates whether the decision variable reached the positive threshold (1),
                            the negative threshold (0), or if there was no decision (2).
    """
    x = np.zeros_like(t)
    for p in range(len(x) - 1):
        x[p + 1] = x[p] + (dt / tauX) * ((epsilon * x[p]) + (x[p] ** 3) - (x[p] ** 5) + b) + c * np.sqrt(dt / tauX) * np.random.randn()
        if abs(x[p]) >= z:
            return x[:p + 1], p * dt, int(x[p] >= z)
    return None, 2, 2


def simulate_sindy(coefs, dt, tauX, c, z, t):
    """
    Simulates a trial using the SINDy model coefficients.

    Parameters:
        coefs (array-like): The coefficients of the SINDy model.
        dt (float): Time step.
        tauX (float): Time constant.
        c (float): Size of the noise.
        z (float): Decision threshold.
        t (array-like): Time array.

    Returns:
        sim (array-like): Trajectory of the decision variable.
        sindy_trial_decision_time (float): Time at which the decision was made.
        sindy_trial_choice (int): Indicates whether the decision variable reached the positive threshold (1),
                                  the negative threshold (0), or if there was no decision (2).
    """
    sim = np.zeros_like(t)
    for h in range(len(sim) - 1):
        sim[h+1] = sim[h] + (dt / tauX) * (coefs[0] + coefs[1] * sim[h] + coefs[2] * sim[h] ** 2 + coefs[3] * sim[h] ** 3 + coefs[4] * sim[h] ** 4 + coefs[5] * sim[h] ** 5) + c * np.sqrt(dt / tauX) * np.random.randn()
 
        # Uncomment the following lines to use polynomial order 4 or 6
#         sim[h+1] = sim[h] + (dt / tauX) * (coefs[0] + coefs[1] * sim[h] + coefs[2] * sim[h] ** 2 + coefs[3] * sim[h] ** 3 + coefs[4] * sim[h] ** 4) + c * np.sqrt(dt / tauX) * np.random.randn()
        # sim[h+1] = sim[h] + (dt / tauX) * (coefs[0] + coefs[1] * sim[h] + coefs[2] * sim[h] ** 2 + coefs[3] * sim[h] ** 3 + coefs[4] * sim[h] ** 4 + coefs[5] * sim[h] ** 5 + coefs[6] * sim[h] ** 6) + c * np.sqrt(dt / tauX) * np.random.randn()
        if abs(sim[h]) >= z:
            return sim[:h + 1], h * dt, int(sim[h] >= z)
        elif h > len(sim) - 3:
            return sim[:h + 1], h * dt, 2
    return sim, h * dt, 2

def sessionNLB(trials, signal, seeding, coefs):
    """
    Runs multiple trials of the NLB model and fits the SINDy model to the simulated data.

    Parameters:
        trials (int): Number of trials to be run.
        signal (float): Strength of the drift rate.
        coefs (array-like): The average coefficients obtained from single trial fitting.
        seeding (int): Seed for random number generation to ensure reproducibility.

    Returns:
        coef_mat (list of array-like): List of coefficients for each SINDy model fit.
        choice_trials (list of int): Choices made in each NLB trial.
        sindy_choice (list of int): Choices made in each SINDy trial.
        decision_time (list of float): Decision times for each NLB trial.
        sindy_dt (list of float): Decision times for each SINDy trial.
        model_data (list of array-like): Trajectories of the decision variable for sampled NLB trials.
        sindy_data (list of array-like): Trajectories of the decision variable for sampled SINDy trials.
    """
    # Parameter initialization
    epsilon = 0.05  # Non-biased stimulus input
    b = signal  # Biased stimulus input
    z = 0.75  # Decision threshold
    tauX = 20  # Time constant
    c = 0.01  # Noise size
    dt = 0.01  # Timestep
    T_Total = 20000  # Total time
    sample = 100  # Sample of decision variable activity

    # Data storage initialization
    coef_mat = []
    choice_trials, decision_time, model_data = [], [], []
    sindy_choice, sindy_dt, sindy_data = [], [], []

    # Loop through trials
    for model in range(trials):
        t = np.arange(0, T_Total, dt)
        np.random.seed(model + 20000 + seeding)

        # Simulate trial
        x, trial_decision_time, trial_choice = simulate_trial(t, dt, tauX, epsilon, signal, c, z)
        if trial_decision_time is not None:
            decision_time.append(trial_decision_time)
            choice_trials.append(trial_choice)
            if model % sample == 0:
                model_data.append(x)

        # Simulate SINDy trial using the provided coefficients
        sim, sindy_trial_decision_time, sindy_trial_choice = simulate_sindy(coefs, dt, tauX, c, z, t)
        sindy_choice.append(sindy_trial_choice)
        sindy_dt.append(sindy_trial_decision_time)
        if model % sample == 0:
            sindy_data.append(sim)

    return coef_mat, choice_trials, sindy_choice, decision_time, sindy_dt, model_data, sindy_data
