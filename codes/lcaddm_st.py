import numpy as np
import scipy
import pysindy as ps
import pandas as pd

# SINDy setup
poly_order = 1
threshold = 0.0001
smoothed_fd = ps.SmoothedFiniteDifference()

modelLCA = ps.SINDy(
    optimizer=ps.STLSQ(threshold=threshold),
    feature_library=ps.PolynomialLibrary(degree=poly_order),
    differentiation_method=smoothed_fd
)

def simulate_trial(t, dt, S1, S2, b, k, c, z):
    """
    Simulates a trial for the LCA-DDM model.

    Parameters:
        t (array-like): Time array.
        dt (float): Time step.
        S1 (float): Stimulus input amplitude to x.
        S2 (float): Stimulus input amplitude to y.
        b (float): Mutual inhibitory coupling strength between x and y.
        k (float): Rate of decay for x and y.
        c (float): Size of the noise.
        z (float): Decision threshold.

    Returns:
        x (array-like): Trajectory of x the decision variable.
        y (array-like): Trajectory of y the decision variable.
        trial_decision_time (float): Time at which the decision was made.
        trial_choice (int): Indicates whether x (1) or y (0) reached the threshold first or if there was no decision (2).
    """
    x = np.zeros_like(t)
    y = np.zeros_like(t)
    
    for p in range(len(x) - 1):
        x[p+1] = x[p] + dt * (-k * x[p] - b * y[p] + S1) + c * np.sqrt(dt) * np.random.randn()
        y[p+1] = y[p] + dt * (-k * y[p] - b * x[p] + S2) + c * np.sqrt(dt) * np.random.randn()
        
        if x[p] >= z or y[p] >= z:
            return x[0:p + 1], y[0:p + 1], p * dt, int(x[p] >= z)
    
    return None, None, p*dt, 2

def fit_sindy_model(x, t):
    """
    Fits a SINDy model to the given data and returns the model coefficients.

    Parameters:
        x (array-like): The data to fit, where each row is a time point and each column is a variable.
        t (array-like): The time points corresponding to the rows of x.

    Returns:
        array-like: The coefficients of the fitted SINDy model.
    """
    modelLCA.fit(x, t[0:len(x)]
    return np.squeeze(modelLCA.coefficients())

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
        sim_x (array-like): Trajectory of x the decision variable.
        sim_y (array-like): Trajectory of y the decision variable.
        sindy_trial_decision_time (float): Time at which the decision was made.
        sindy_trial_choice (int): Indicates whether x (1) or y (0) reached the threshold first or if there was no decision (2).
    """
    sim_x = np.zeros_like(t)
    sim_y = np.zeros_like(t)
    coef_0, coef_1, coef_2, coef_3, coef_4, coef_5 = coefs
    
    for h in range(len(sim_x) - 1):
        sim_x[h+1] = sim_x[h] + dt * (coef_1 * sim_x[h] + coef_2 * sim_y[h] + coef_0) + c * np.sqrt(dt) * np.random.randn()  # Poly order 1
        sim_y[h+1] = sim_y[h] + dt * (coef_5 * sim_y[h] + coef_4 * sim_x[h] + coef_3) + c * np.sqrt(dt) * np.random.randn()  # Poly order 1
        
        # Uncomment the following lines to use polynomial order 0 or 2
        # sim_x[h+1] = sim_x[h] + dt * coef_1 + c * np.sqrt(dt) * np.random.randn()  # Poly order 0
        # sim_y[h+1] = sim_y[h] + dt * coef_2 + c * np.sqrt(dt) * np.random.randn()  # Poly order 0
        
        #sim_x[h+1] = sim_x[h]+dt*(coefs[0,1]*sim_x[h]+coefs[0,2]*sim_y[h]+coefs[0,0]) + c*np.sqrt(dt)*np.random.randn() poly order 2
        #sim_y[h+1] = sim_y[h]+dt*(coefs[1,2]*sim_y[h]+coefs[1,1]*sim_x[h]+coefs[1,0]) + c*np.sqrt(dt)*np.random.randn() poly order 2
        
        if sim_x[h] >= z or sim_y[h] >= z:
            return sim_x[:h + 1], sim_y[:h + 1], h * dt, int(sim_x[h] >= z)
    
    return None, None, h*dt, 2

def sessionLCADDM(trials, signal, seeding):
    """
    Runs multiple trials of the LCA-DDM model and fits the SINDy model to the simulated data.

    Parameters:
        trials (int): Number of trials to be run.
        signal (float): Strength of the drift rate.
        seeding (int): Seed for random number generation to ensure reproducibility.

    Returns:
        coef_mat (list of array-like): List of coefficients for each SINDy model fit.
        choice_trials (list of int): Choices made in each LCA-DDM trial.
        sindy_choice (list of int): Choices made in each SINDy trial.
        decision_time (list of float): Decision times for each LCA-DDM trial.
        sindy_dt (list of float): Decision times for each SINDy trial.
        model_data (list of list of array-like): Trajectories of x and y for sampled LCA-DDM trials.
        sindy_data (list of list of array-like): Trajectories of x and y for sampled SINDy trials.
    """
    # Parameter initialization
    S1 = 3 + signal  # Stimulus input amplitude to y1 
    S2 = 3  # Stimulus input amplitude to y2 
    b = 10  # Mutual inhibitory coupling strength between the y's
    k = 10  # Rate of decay of the y's
    z = 1  # Decision threshold
    c = 0.11  # Size of the noise
    dt = 0.01  # Timestep
    T_Total = 10000  # Total time
    sample = 42  # Storing trial data

    # Data storage initialization
    coef_mat = []
    choice_trials, decision_time, model_data = [], [], []
    sindy_choice, sindy_dt, sindy_data = [], [], []

    # Loop through trials
    for model in range(trials):
        t = np.arange(0, T_Total, dt)
        np.random.seed(model + 20000 + (seeding - 2000))

        # Simulate trial
        x, y, trial_decision_time, trial_choice = simulate_trial(t, dt, S1, S2, b, k, c, z)
        if trial_decision_time is not None:
            decision_time.append(trial_decision_time)
            choice_trials.append(trial_choice)
            if model % sample == 0:
                model_data.append([x, y])
   
        # Fit SINDy model and simulate
        coefs = fit_sindy_model(np.vstack((x, y)).T, t[0:len(x)])
        coef_mat.append(coefs)
        coefs=coefs.reshape(-1)
        sim_x, sim_y, sindy_trial_decision_time, sindy_trial_choice = simulate_sindy(coefs, dt, c, z, t)
        if sindy_trial_decision_time is not None:
            sindy_choice.append(sindy_trial_choice)
            sindy_dt.append(sindy_trial_decision_time)
            if model % sample == 0:
                sindy_data.append([sim_x, sim_y])

    return coef_mat, choice_trials, sindy_choice, decision_time, sindy_dt, model_data, sindy_data
##single trial dataset##
#WARNING this takes time to run#
# trials = 10000
# nt_var = [(trials, signal,2000) for signal in np.arange(0.000, 0.041, 0.001)]
# data=[]
# for i in range(len(nt_var)):
#     data.append(sessionLCADDM(nt_var[i]))
