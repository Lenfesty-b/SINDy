from joblib import Parallel, delayed
from multiprocessing import Pool
import multiprocessing as mp
import numpy as np
import scipy
import pysindy as ps
import pandas as pd
import time
import pickle
import os 

# SINDy setup
poly_order = 2#1
threshold = 0.001#0.001 
smoothed_fd=ps.SmoothedFiniteDifference()

modelLCA = ps.SINDy(
    optimizer=ps.STLSQ(threshold=threshold),
    feature_library=ps.PolynomialLibrary(degree=poly_order),
    differentiation_method=smoothed_fd
)

def fit_sindy_model(x):
    """
    Fits a SINDy model to the given data and returns the model coefficients.

    Parameters:
        x (array-like): The data to fit, where each row is a time point and each column is a variable.
        t (array-like): The time points corresponding to the rows of x.

    Returns:
        array-like: The coefficients of the fitted SINDy model.
    """
    modelLCA.fit(x, t=0.01,multiple_trajectories=True)
    
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
    sim_x = np.zeros_like(t,dtype=np.float32)
    sim_y = np.zeros_like(t,dtype=np.float32)
    #coef_0, coef_1, coef_2, coef_3, coef_4, coef_5 = coefs
    # coef_1, coef_2 = coefs
    coef_0=coefs[0]
    coef_1=coefs[1]
    coef_2=coefs[2]
    coef_3=coefs[3]
    coef_4=coefs[4]
    coef_5=coefs[5]
    coef_6=coefs[6]
    coef_7=coefs[7]
    coef_8=coefs[8]
    coef_9=coefs[9]
    coef_10=coefs[10]
    coef_11=coefs[11]
    
    
    for h in range(len(sim_x) - 1):
        #sim_x[h+1] = sim_x[h] + dt * (coef_1 * sim_x[h] + coef_2 * sim_y[h] + coef_0) + c * np.sqrt(dt) * np.random.randn()  # Poly order 1
        #sim_y[h+1] = sim_y[h] + dt * (coef_5 * sim_y[h] + coef_4 * sim_x[h] + coef_3) + c * np.sqrt(dt) * np.random.randn()  # Poly order 1
        
        # Uncomment the following lines to use polynomial order 0 or 2
        #sim_x[h+1] = sim_x[h] + dt * coef_1 + c * np.sqrt(dt) * np.random.randn()  # Poly order 0
        #sim_y[h+1] = sim_y[h] + dt * coef_2 + c * np.sqrt(dt) * np.random.randn()  # Poly order 0
        
        sim_x[h+1] = sim_x[h] + dt * (coef_0 * sim_x[h] + coef_1 * sim_y[h] + coef_2 + (coef_3 * sim_x[h]**2) + (coef_4 * sim_x[h] * sim_y[h]) + (coef_5 * sim_y[h]**2)) + c * np.sqrt(dt) * np.random.randn()  # Poly order 2
        sim_y[h+1] = sim_y[h] + dt * (coef_6 * sim_y[h] + coef_7 * sim_x[h] + coef_8 + (coef_9 * sim_x[h]**2) + (coef_10 * sim_x[h] * sim_y[h]) + (coef_11 * sim_y[h]**2)) + c * np.sqrt(dt) * np.random.randn()  # Poly order 2
        
        if sim_x[h] >= z or sim_y[h] >= z:
            return sim_x[:h + 1], sim_y[:h + 1], h * dt, int(sim_x[h] >= z)
    
    return None, None, h*dt, 2

def sessionLCA_sindy(trials, signal, model_data, seed):

    # Parameter initialization
    S1=1.85+signal#Stimulus input amplitude to y1 1.85
    S2=1.85#Stimulus input amplitude to y2 1.85
    b=4 #Mutual inhibitory coupling strength between the y's 4
    k=3 # Rate of decay of the y's 3
    z=1#3#.9#0.89881186# Decision threshold
    #z=2.5
    c=.11#0.075#.165# Size of the noise
    dt=.01#timestep
    Model_total=trials
    T_Total=10000 #Total time
    t = np.arange(0, T_Total, dt)
    # Data storage initialization
    coef_mat = []

    sindy_choice, sindy_dt, sindy_data = [], [], []

    # Loop through trials
    for model in range(trials):
   
        np.random.seed(model + 27500+seed)
        
        # Fit SINDy model and simulate
        coef_mat.append(model_data)
        coefs=model_data.reshape(-1)
        sim_x, sim_y, sindy_trial_decision_time, sindy_trial_choice = simulate_sindy(coefs, dt, c, z, t)
        if sindy_trial_decision_time is not None:
            sindy_choice.append(sindy_trial_choice)
            sindy_dt.append(sindy_trial_decision_time)
            sindy_data.append([sim_x, sim_y])

    return  sindy_choice, sindy_dt, sindy_data

def coefs_LCAMulti(model_data, trials):
    model_cos = np.empty(41, dtype=object)
    model_cos=fit_sindy_model(([np.array(model_data[2][i]).T for i in range(trials)]))
    return model_cos

def process_session_LCAMulti(trials, signal, model_data,seed):
    return sessionLCA_sindy(trials, signal, model_data, seed)

def process_coefs_LCAMulti(model_data, trials):
    return coefs_LCAMulti(model_data, trials)

# Define parameters for the session-LCA
trials = 100
seeds = [trials * i for i in range(100)]
signals = np.arange(0.000, 1.21, 0.03)  # 41 signals
filename = f'lca_timecourse_combined.pkl'
with open(filename, 'rb') as f:
    test = pickle.load(f)

print("coefs started")

model_coefs = Parallel(n_jobs=64)(delayed(process_coefs_LCAMulti) (test[i],10000) for i in range(41))

del test

print("loop started")
# Main loop to process each seed iteration and save to a file 
for seed in seeds: 
    # Create a list of parameters for the current seed 
    nt_var = [(trials, signals[i],model_coefs[i],seed+2000) for i in range(len(signals))] 

    # Time the parallel approach 
    start_time_parallel = time.time() 
    timecourse_parallel = Parallel(n_jobs=64)(
        delayed(process_session_LCAMulti)(nt_var[i][0], nt_var[i][1], nt_var[i][2],nt_var[i][3]) 
        for i in range(41) 
    ) 
    end_time_parallel = time.time() 
  
    print(f"Seed {seed}: Parallel approach time: {end_time_parallel - start_time_parallel} seconds") 
  
    # Save the output for this seed to a file 
    filename = f'lca_timecourse_seedmulti2_{seed}.pkl' 
    with open(filename, 'wb') as f: 
        pickle.dump(timecourse_parallel, f) 
    # Clear memory 
    del timecourse_parallel 
# Initialize the combined data structure 
combined_data = [] 
for _ in range(len(signals)): 
    combined_data.append([[], [], []])  # Placeholder for choice_trials, decision_time, model_data 

# Process each file and accumulate data 
for seed in seeds: 
    filename = f'lca_timecourse_seedmulti2_{seed}.pkl'    
    with open(filename, 'rb') as f: 
        test = pickle.load(f)  # Load the data for the current seed    
    # Combine data for each signal 
    for i in range(len(signals)): 

        combined_data[i][0].extend(test[i][0])  # Combine choice_trials 
        combined_data[i][1].extend(test[i][1])  # Combine decision_time 
        combined_data[i][2].extend(test[i][2])  # Combine model_data      

    # Optionally, delete the individual seed file after loading to save disk space 
#    os.remove(filename) 

# Save the combined data structures to a final output file 
with open('lca_timecourse_multi2_combined.pkl', 'wb') as f: 
    pickle.dump(combined_data, f) 
print("All seed iterations processed and combined into lca_timecourse_multi2_combined.pkl")