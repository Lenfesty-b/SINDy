from multiprocessing import Pool
import multiprocessing as mp
from joblib import Parallel, delayed
import numpy as np
import scipy
import pysindy as ps
import pandas as pd
import time
import pickle
import argparse

def simulate_trial(t, dt, S1,S2,b,k, c, z):
    x = np.zeros_like(t)
    y = np.zeros_like(t)
    
    for p in range(len(x) - 1):
        #print(p)
        x[p+1] = x[p]+dt*(-k*x[p]-b*y[p]+S1) + c*np.sqrt(dt)*np.random.randn()
        y[p+1] = y[p]+dt*(-k*y[p]-b*x[p]+S2) + c*np.sqrt(dt)*np.random.randn()
        
        if x[p] >= z or y[p]>=z :
            return x[0:p + 1],y[0:p+1], p * dt, int(x[p] >= z)
        
    return x,y, None, None
    
def sessionLCADDM(trials, signal,seeding):
    # Parameter initialization
    S1=3+signal#Stimulus input amplitude to y1 1.85
    S2=3#Stimulus input amplitude to y2 1.85
    b=10 #Mutual inhibitory coupling strength between the y's 4
    k=10 # Rate of decay of the y's 3
    #z=1#.9#0.89881186# Decision threshold
    z=1#2.5
    c=.11#0.075#.165# Size of the noise
    dt=.01#timestep
    Model_total=trials
    T_Total=10000 #Total time


    # Data storage initialization
    choice_trials, decision_time, model_data = [], [], []


    # Loop through trials
    for model in range(trials):
        t = np.arange(0, T_Total, dt)
        np.random.seed(model + 20000+(seeding-2000))

        # Simulate trial
        x,y, trial_decision_time, trial_choice = simulate_trial(t, dt,S1,S2,b,k, c, z)
        if trial_decision_time is not None:
            decision_time.append(trial_decision_time)
            choice_trials.append(trial_choice)
            #if model % sample == 0:
            model_data.append([x,y])


    return coef_mat, choice_trials, decision_time, model_data
    
if __name__ == '__main__':


    start = time.time()
    trials = 2000
    parser = argparse.ArgumentParser(description='Process some variables.')
    parser.add_argument('variable', type=int, help='An integer variable to process')

    seed = parser.parse_args()
    nt_var = [(trials, signal,seed.variable) for signal in np.arange(0.000, 0.041, 0.001)]

    with mp.Pool(processes=12) as p:
        output = p.starmap(sessionLCADDM, nt_var)
    
    end = time.time()
    print(f"Execution Time: {end - start} seconds")
    file_name = "multiprocessing_lcaddm_kelvin_full_"+str(seed.variable)+".pkl"
    open_file = open(file_name, "wb")
    pickle.dump(output, open_file)
    open_file.close()