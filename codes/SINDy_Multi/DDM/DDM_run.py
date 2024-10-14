import numpy as np 
import scipy 
import pandas as pd 
import pickle 
from joblib import Parallel, delayed 
import numpy as np 
import time 
import os 

def simulate_sindy(coefs, dt, c, z, t):

    sim = np.zeros_like(t,dtype=np.float32)
    for h in range(len(sim) - 1):
        sim[h+1] = sim[h] + (dt * coefs) + c * np.sqrt(dt) * np.random.randn()  # Polynomial order 0
        # Uncomment the following lines to use polynomial order 1 or 2
        # sim[h+1] = sim[h] + (dt * coefs[0] + (coefs[1] * sim[h])) + c * np.sqrt(dt) * np.random.randn()  # Polynomial order 1
        # sim[h+1] = sim[h] + (dt * coefs[0] + (coefs[1] * sim[h]) + (coefs[2] * sim[h]**2)) + c * np.sqrt(dt) * np.random.randn()  # Polynomial order 2
        if abs(sim[h]) >= z:
            return sim[:h + 1], h * dt, int(sim[h] >= z)
    return None, h*dt, 2

def sessionSINDY(trials, signal,model_data,seed):

    # Parameter initialization
    z = 1  # Decision threshold
    c = 0.11  # Size of the noise
    A = signal  # Drift rate
    dt = 0.1  # Time step
    t_total = 10000  # Total trial time

    # Data storage initialization
    coef_mat = []
    sindy_choice, sindy_dt, sindy_data = [], [], []
    t = np.arange(0, t_total, dt) 
                
#         # Fit SINDy model and simulate
    coefs = fit_sindy_model_multi(model_data, t)
    coef_mat.append(coefs)

    for model in range(trials):
        np.random.seed(model + 20000)

          
        sim, sindy_trial_decision_time, sindy_trial_choice = simulate_sindy(coefs, dt, c, z, t)
        if sindy_trial_decision_time is not None:
            sindy_choice.append(sindy_trial_choice)
            sindy_dt.append(sindy_trial_decision_time)
            sindy_data.append(sim)

                
    return coef_mat, sindy_choice, sindy_dt, sindy_data 

# Define parameters for the session-ddm 
trials = 1000 
seeds = [1000 * i for i in range(10)] 
signals= np.arange(0.0000, 0.041, 0.001) 
with open('ddm_timecourse.pkl', 'rb') as f:
    model_data=pickle.load(f)

# Define the function to be run in parallel 
def process_session_ddm(trials, signal, seed): 
    return sessionSINDY(trials, signal, model_data, seed) 

# Main loop to process each seed iteration and save to a file 
for seed in seeds: 
    # Create a list of parameters for the current seed 
    nt_var = [(trials, signal,model_data,seed) for signal in signals] 

    # Time the parallel approach 
    start_time_parallel = time.time() 
    timecourse_parallel = Parallel(n_jobs=60)( 
        delayed(process_session_ddm)(nt_var[i][0], nt_var[i][1], nt_var[i][2],nt_var[i][3]) 
        for i in range(len(nt_var)) 
    ) 
    end_time_parallel = time.time() 
  
    print(f"Seed {seed}: Parallel approach time: {end_time_parallel - start_time_parallel} seconds") 
  
    # Save the output for this seed to a file 
    filename = f'ddm_timecourse_seed_{seed}_multi.pkl' 
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
    filename = f'ddm_timecourse_seed_{seed}_multi.pkl'    
    with open(filename, 'rb') as f: 
        test = pickle.load(f)  # Load the data for the current seed    
    # Combine data for each signal 
    for i in range(len(signals)): 

        combined_data[i][0].extend(test[i][0])  # Combine choice_trials 
        combined_data[i][1].extend(test[i][1])  # Combine decision_time 
        combined_data[i][2].extend(test[i][2])  # Combine model_data      

    # Optionally, delete the individual seed file after loading to save disk space 
    os.remove(filename) 

# Save the combined data structures to a final output file 
with open('ddm_timecourse_multi_combined.pkl', 'wb') as f: 

    pickle.dump(combined_data, f) 
print("All seed iterations processed and combined into ddm_timecourse_combined_mutli.pkl") 