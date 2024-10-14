import numpy as np 
import argparse 
import pickle 
from joblib import Parallel, delayed 
import time 
import os 

def simulate_trial(t, dt, tauX, epsilon, b, c, z): 

    x = np.zeros_like(t,dtype=np.float16) 
    
    for p in range(len(x) - 1): 
    
        x[p + 1] = x[p] + (dt / tauX) * ((epsilon * x[p]) + (x[p] ** 3) - (x[p] ** 5) + b) + c * np.sqrt((dt / tauX)) * np.random.randn() 
        
        if abs(x[p]) >= z: 
            return x[:p + 1], p * dt, int(x[p] >= z)
             
    return x, None, None 

def sessionNLB(trials, signal,seeding): 

    # Parameter initialization 
    epsilon = 0.05 
    z = .75 
    tauX = 20 
    c = 0.01 
    dt = 0.01 
    T_Total = 20000 

    t = np.arange(0, T_Total, dt) 
    # Data storage initialization 
    choice_trials, decision_time, model_data = [], [], [] 

    # Loop through trials 
    for model in range(trials):
     
        
        np.random.seed(model + 20000+seeding) 
        # Simulate trial 
        x, trial_decision_time, trial_choice = simulate_trial(t, dt, tauX, epsilon, signal, c, z)
         
        if trial_decision_time is not None: 
            decision_time.append(trial_decision_time) 
            choice_trials.append(trial_choice) 
            model_data.append(x) 
            
    return choice_trials, decision_time, model_data 

#for j in range(len(signals)):
  #signals= np.arange(0.0040, 0.0081, 0.0002)
# Define the function to be run in parallel 
def process_session_nlb(trials, signal, seed): 
    return sessionNLB(trials, signal, seed) 
    
def process_signal(signal):
    # Define parameters for the session-nlb 
    trials = 10 
    seeds = [trials * i for i in range(1000)] 
  
    # Main loop to process each seed iteration and save to a file 
    for seed in seeds: 
        # Create a list of parameters for the current seed 
        nt_var = [(trials, signal,seed+1000) ]#for signal in signals[j]] 
    
        # Time the parallel approach 
        start_time_parallel = time.time() 
        timecourse_parallel = Parallel(n_jobs=60)( 
            delayed(process_session_nlb)(params[0], params[1], params[2]) 
            for params in nt_var) 
         
        end_time_parallel = time.time() 
      
        print(f"Seed {seed}: Parallel approach time: {end_time_parallel - start_time_parallel} seconds") 
      
        # Save the output for this seed to a file 
        filename = f'nlb_timecourse_seed_{signal}_{seed}.pkl' 
        with open(filename, 'wb') as f: 
            pickle.dump(timecourse_parallel, f) 
        # Clear memory 
        del timecourse_parallel 
    # Initialize the combined data structure 
    #combined_data = [] 
    #for _ in range(len(signals)): 
    combined_data=[[], [], []]  # Placeholder for choice_trials, decision_time, model_data 
    
    # Process each file and accumulate data 
    for seed in seeds: 
        filename = f'nlb_timecourse_seed_{signal}_{seed}.pkl'    
        with open(filename, 'rb') as f: 
            test = pickle.load(f)  # Load the data for the current seed    
        # Combine data for each signal 
        #for i in range(len(signals[j])): 
    
        combined_data[0].extend(test[0][0])  # Combine choice_trials 
        combined_data[1].extend(test[0][1])  # Combine decision_time 
        combined_data[2].extend(test[0][2])  # Combine model_data      
      
          # Optionally, delete the individual seed file after loading to save disk space 
        os.remove(filename) 
      
    output_filename = f'nlb_timecourse_combined_{signal}.pkl'
    with open(output_filename, 'wb') as f:
        pickle.dump(combined_data, f)
    print(f"All seed iterations processed and combined into {output_filename}")
    
      
signals=np.arange(0.0000, 0.0081, 0.0002)  
timecourse_parallel = Parallel(n_jobs=60)(delayed(process_signal)(signal) for signal in signals) 