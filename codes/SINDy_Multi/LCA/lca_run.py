import pickle
from joblib import Parallel, delayed
import numpy as np
import time
import os

def simulate_trial(t, dt, S1,S2,b,k, c, z):
    x = np.zeros_like(t,dtype=np.float32)
    y = np.zeros_like(t,dtype=np.float32)
    for p in range(len(x) - 1):
        #print(p)
        x[p+1] = x[p]+dt*(-k*x[p]-b*y[p]+S1) + c*np.sqrt(dt)*np.random.randn()
        y[p+1] = y[p]+dt*(-k*y[p]-b*x[p]+S2) + c*np.sqrt(dt)*np.random.randn()
        if x[p] >= z or y[p]>=z :
            return x[0:p + 1],y[0:p+1], p * dt, int(x[p] >= z)
    return None,None, None, None
 
 

def sessionLCA(trials, signal,seed):
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
    sample=40
 
    # Data storage initialization
    choice_trials, decision_time= [], [] 
    model_data =  []
 
    # Loop through trials
    for model in range(trials):
        t = np.arange(0, T_Total, dt)
        np.random.seed(model + 27500 +seed)
#         print(model + 20000 + 7500+seed)
 
        # Simulate trial
        x,y, trial_decision_time, trial_choice = simulate_trial(t, dt,S1,S2,b,k, c, z)
        if trial_decision_time is not None:
            decision_time.append(trial_decision_time)
            choice_trials.append(trial_choice)
            model_data.append([x,y])
    return choice_trials,decision_time,model_data

# Define parameters for the session-LCA
trials = 500
seeds = [500 * i for i in range(20)]
# trials = 1000
# seeds = [trials * i for i in range(10)]
 
signals = np.arange(0.000, 1.21, 0.03)  # 41 signals
 
# Define the function to be run in parallel
def process_session_LCA(trials, signal, seed):
    return sessionLCA(trials, signal, seed)
 
# Main loop to process each seed iteration and save to a file
for seed in seeds:
    # Create a list of parameters for the current seed
    nt_var = [(trials, signal, 2000+seed) for signal in signals]
 
    # Time the parallel approach
    start_time_parallel = time.time()
    timecourse_parallel = Parallel(n_jobs=16)(
        delayed(process_session_LCA)(nt_var[i][0], nt_var[i][1], nt_var[i][2])
        for i in range(len(nt_var))
    )
    end_time_parallel = time.time()
 
    print(f"Seed {seed}: Parallel approach time: {end_time_parallel - start_time_parallel} seconds")
 
    # Save the output for this seed to a file
    filename = f'lca_timecourse_seed_{seed}.pkl'
    with open(filename, 'wb') as f:
        pickle.dump(timecourse_parallel, f)
 
    # Clear memory
    del timecourse_parallel
 
 
# Initialize the combined data structu re
combined_data = []
for _ in range(len(signals)):
    combined_data.append([[], [], []])  # Placeholder for choice_trials, decision_time, model_data
 
# Process each file and accumulate data
for seed in seeds:
    filename = f'lca_timecourse_seed_{seed}.pkl'
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
with open('lca_timecourse_combined.pkl', 'wb') as f:
    pickle.dump(combined_data, f)
 
print("All seed iterations processed and combined into lca_timecourse_combined.pkl")