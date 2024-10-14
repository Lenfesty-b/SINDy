import numpy as np
import scipy
import pysindy as ps
import os
import pandas as pd
from joblib import Parallel, delayed 
import time
import pickle

## SINDy setup
#poly_order = 5  # Polynomial order for SINDy model
#threshold = 0.000001  # Hyperparameter to control level of sensitivity
#smoothed_fd = ps.SmoothedFiniteDifference()  # Differentiation method to smooth noise
#
#modelNLB = ps.SINDy(
#    optimizer=ps.STLSQ(threshold=threshold),
#    feature_library=ps.PolynomialLibrary(degree=poly_order),
#    differentiation_method=smoothed_fd
#)
def fit_sindy_model(x,tauX):

    modelNLB.fit(x[2], t=0.01 / tauX, multiple_trajectories=True)
    return np.squeeze(modelNLB.coefficients())
    
def simulate_sindy(coefs, dt, tauX, c, z, t):

    sim = np.zeros_like(t)

    for h in range(len(sim) - 1):
        #sim[h + 1] = sim[h] + (dt / tauX) * (coefs[0] + coefs[1] * sim[h] + coefs[2] * sim[h] ** 2 + coefs[3] * sim[h] ** 3 + coefs[4] * sim[h] ** 4
         #                                    + coefs[5] * sim[h] ** 5) + c * np.sqrt(dt / tauX) * np.random.randn()
        
        #sim[h+1] = sim[h]+(dt/tauX)*(coefs[0]+(coefs[1]*sim[h])+(coefs[2]*sim[h]**2)+(coefs[3]*sim[h]**3)+(coefs[4]*sim[h]**4))+ c*np.sqrt((dt/tauX))*np.random.randn() #poly4
        sim[h+1] = sim[h]+(dt/tauX)*(coefs[0]+(coefs[1]*sim[h])+(coefs[2]*sim[h]**2)+(coefs[3]*sim[h]**3)+(coefs[4]*sim[h]**4)
                                         +(coefs[5]*sim[h]**5)+(coefs[6]*sim[h]**6))+ c*np.sqrt((dt/tauX))*np.random.randn() #poly6
        if abs(sim[h]) >= z:
            return sim[:h + 1], h * dt, int(sim[h] >= z)
        elif h > len(sim) - 3:
            return sim[:h + 1], h * dt, 2
    return sim, h * dt, 2

def sessionNLB(trials, signal, seeding, coefs):

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
    sindy_choice, sindy_dt, sindy_data = [], [], []

    # Loop through trials
    for model in range(trials):
        t = np.arange(0, T_Total, dt)
        np.random.seed(model + 20000 + seeding)


        # Fit SINDy model and simulate
        #coefs = fit_sindy_model(coefs_,  tauX)
        #coef_mat.append(coefs)

        sim, sindy_trial_decision_time, sindy_trial_choice = simulate_sindy(coefs, dt, tauX, c, z, t)
        sindy_choice.append(sindy_trial_choice)
        sindy_dt.append(sindy_trial_decision_time)
        sindy_data.append(sim)

    return   sindy_choice,  sindy_dt, sindy_data


def process_session_nlb(trials, signal, seed,coefs): 
    return sessionNLB(trials, signal, seed,coefs) 
    
trials = 10 
seeds = [10 * i for i in range(1000)] 



for half in range(2):
    # Define parameters for the session-nlb 
    filename = f'/users/blenfesty/SINDy/NLB/nlb_data/nlb_combined_coefs_poly6.pkl'
    with open(filename, 'rb') as f:
        test = pickle.load(f) 
    if half==0:
        signals=[round(i,4) for i in np.arange(0.000,.0041,.0002)]
        model_coefs=[test[i][2] for i in range(len(signals))] 
    elif half==1:
        signals=[round(i,4) for i in np.arange(0.0042,.0081,.0002)]
        model_coefs=[test[i+21][2] for i in range(len(signals))]  

        
    #model_coefs = [fit_sindy_model(test[j],20) for j in range(41)]#Parallel(n_jobs=64)(delayed(process_coefs_NLBMulti) (test[i],10000) for i in range(41))
    
    del test
    
    # Main loop to process each seed iteration and save to a file 
    for seed in seeds: 
        # Create a list of parameters for the current seed 
        nt_var = [(trials, signals[i],seed+1000,model_coefs[i]) for i in range(len(signals))] 
    
        # Time the parallel approach 
        start_time_parallel = time.time() 
        timecourse_parallel = Parallel(n_jobs=64)( 
            delayed(process_session_nlb)(nt_var[i][0], nt_var[i][1], nt_var[i][2],nt_var[i][3]) 
            for i in range(len(nt_var)) 
        ) 
        end_time_parallel = time.time() 
      
        print(f"Seed {seed}: Parallel approach time: {end_time_parallel - start_time_parallel} seconds") 
      
        # Save the output for this seed to a file 
        filename = f'/mnt/scratch2/users/blenfesty/nlb_timecourse_multi_seed_{seed}.pkl' 
        with open(filename, 'wb') as f: 
            pickle.dump(timecourse_parallel, f) 
        # Clear memory 
        del timecourse_parallel 
    # Initialize the combined data structure 
    combined_data = [] 
    combined_data_tm = []
    for _ in range(len(signals)): 
        combined_data.append([[], []])  # Placeholder for choice_trials, decision_time, model_data 
        combined_data_tm.append([])
    # Process each file and accumulate data 
    for seed in seeds: 
        filename = f'/mnt/scratch2/users/blenfesty/nlb_timecourse_multi_seed_{seed}.pkl'    
        with open(filename, 'rb') as f: 
            test = pickle.load(f)  # Load the data for the current seed    
        # Combine data for each signal 
        for i in range(len(signals)): 
    
            combined_data[i][0].extend(test[i][0])  # Combine choice_trials 
            combined_data[i][1].extend(test[i][1])  # Combine decision_time 
            combined_data_tm[i].extend(test[i][2])  # Combine model_data      
    
        # Optionally, delete the individual seed file after loading to save disk space 
        os.remove(filename) 
    
    # Save the combined data structures to a final output file 
    with open(f'/mnt/scratch2/users/blenfesty/nlb_timecourse_multi_combined_{half}_poly6.pkl', 'wb') as f: 
        pickle.dump(combined_data, f) 
        
        # Save the combined data structures to a final output file 
    with open(f'/mnt/scratch2/users/blenfesty/nlb_timecourse_multi_combined_{half}_tc_poly6.pkl', 'wb') as f: 
        pickle.dump(combined_data_tm, f) 
        
        
    print("All seed iterations processed and combined into nlb_timecourse_multi_combined.pkl") 