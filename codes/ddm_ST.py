import numpy as np
import scipy
import pysindy as ps
import pandas as pd

# SINDy setup
poly_order = 0  # Polynomial order for SINDy model (0, 1, or 2)
threshold = 0.00001  # Hyperparameter to control the level of sensitivity
smoothed_fd = ps.SmoothedFiniteDifference()  # Differentiation method to smooth noise

modelDDM = ps.SINDy(
    optimizer=ps.STLSQ(threshold=threshold),
    feature_library=ps.PolynomialLibrary(poly_order),
    differentiation_method=smoothed_fd
)

def simulate_trial(t, dt, A, c, z):
    """
    Simulates a trial for the Drift Diffusion Model (DDM).

    Parameters:
        t (array-like): Time array.
        dt (float): Time step.
        A (float): Drift rate.
        c (float): Size of the noise.
        z (float): Decision threshold.

    Returns:
        x (array-like): Trajectory of x  the decision variable.
        trial_decision_time (float): Time at which the decision was made.
        trial_choice (int): Indicates whether the positive threshold (1) or negative threshold (0) was reached,
                                  or if there was no decision (2).
    """
    x = np.zeros_like(t)
    for p in range(len(x) - 1):
        x[p+1] = x[p] + (dt * A) + c * np.sqrt(dt) * np.random.randn()
        if abs(x[p]) >= z:
            return x[:p + 1], p * dt, int(x[p] >= z)
    return None, 2, 2

def fit_sindy_model(x, t):
    """
    Fits a SINDy model to the given data and returns the model coefficients.

    Parameters:
        x (array-like): The data to fit, where each row is a time point and each column is a variable.
        t (array-like): The time points corresponding to the rows of x.

    Returns:
        array-like: The coefficients of the fitted SINDy model.
    """
    modelDDM.fit(x, t=t[0:len(x)])
    return np.squeeze(modelDDM.coefficients())

def simulate_sindy(coefs, dt, c, z, t):
    """
    Simulates a trial using the SINDy model coefficients.

    Parameters:
        coefs (array-like): The coefficients of the SINDy model.
        dt (float): Time step.
        c (float): Size of the noise.
        z (float): Decision threshold.
        t (array-like): Time array.

    Returns:
        sim (array-like): Trajectory of x the decision variable.
        sindy_trial_decision_time (float): Time at which the decision was made.
        sindy_trial_choice (int): Indicates whether the positive threshold (1) or negative threshold (0) was reached,
                                  or if there was no decision (2).
    """
    sim = np.zeros_like(t)
    for h in range(len(sim) - 1):
        sim[h+1] = sim[h] + (dt * coefs) + c * np.sqrt(dt) * np.random.randn()  # Polynomial order 0
        # Uncomment the following lines to use polynomial order 1 or 2
        # sim[h+1] = sim[h] + (dt * coefs[0] + (coefs[1] * sim[h])) + c * np.sqrt(dt) * np.random.randn()  # Polynomial order 1
        # sim[h+1] = sim[h] + (dt * coefs[0] + (coefs[1] * sim[h]) + (coefs[2] * sim[h]**2)) + c * np.sqrt(dt) * np.random.randn()  # Polynomial order 2
        if abs(sim[h]) >= z:
            return sim[:h + 1], h * dt, int(sim[h] >= z)
    return None, 2, 2

def sessionDDM(trials, signal):
    """
    Runs multiple trials of the DDM and fits the SINDy model to the simulated data.

    Parameters:
        trials (int): Number of trials to be run.
        signal (float): Strength of the drift rate.

    Returns:
        coef_mat (list of array-like): List of coefficients for each SINDy model fit.
        choice_trials (list of int): Choices made in each DDM trial.
        sindy_choice (list of int): Choices made in each SINDy trial.
        decision_time (list of float): Decision times for each DDM trial.
        sindy_dt (list of float): Decision times for each SINDy trial.
        model_data (list of array-like): Trajectories of x for sampled DDM trials.
        sindy_data (list of array-like): Trajectories of x for sampled SINDy trials.
    """
    # Parameter initialization
    z = 1  # Decision threshold
    c = 0.11  # Size of the noise
    A = signal  # Drift rate
    dt = 0.1  # Time step
    t_total = 10000  # Total trial time
    sample = 20  # Storing trial data

    # Data storage initialization
    coef_mat = []
    choice_trials, decision_time, model_data = [], [], []
    sindy_choice, sindy_dt, sindy_data = [], [], []

    # Loop through trials
    for model in range(trials):
        t = np.arange(0, t_total, dt)
        np.random.seed(model + 20000)

        # Simulate trial
        x, trial_decision_time, trial_choice = simulate_trial(t, dt, A, c, z)
        if trial_decision_time is not None:
            decision_time.append(trial_decision_time)
            choice_trials.append(trial_choice)
            if model % sample == 0:
                model_data.append(x)
                
        # Fit SINDy model and simulate
        coefs = fit_sindy_model(x, t)
        coef_mat.append((coefs))

        sim, sindy_trial_decision_time, sindy_trial_choice = simulate_sindy(coefs, dt, c, z, t)
        if sindy_trial_decision_time is not None:
            sindy_choice.append(sindy_trial_choice)
            sindy_dt.append(sindy_trial_decision_time)
            if model % sample == 0:
                sindy_data.append(sim)

    return coef_mat, choice_trials, sindy_choice, decision_time, sindy_dt, model_data, sindy_data
