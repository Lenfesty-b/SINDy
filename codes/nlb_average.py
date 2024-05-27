import numpy as np
import scipy
import pysindy as ps
import pandas as pd

def simulate_trial(t, dt, tauX, epsilon, b, c, z):
    x = np.zeros_like(t)
    for p in range(len(x) - 1):
        x[p + 1] = x[p] + (dt / tauX) * ((epsilon * x[p]) + (x[p] ** 3) - (x[p] ** 5) + b) + c * np.sqrt((dt / tauX)) * np.random.randn()
        if abs(x[p]) >= z:
            return x[:p + 1], p * dt, int(x[p] >= z)
    return x, None, None

def fit_sindy_model(x, t, tauX):
    # Assuming modelNLB is defined and accessible
    modelNLB.fit(x, t=t[0:len(x)]/tauX)
    return np.squeeze(modelNLB.coefficients())

def simulate_sindy(coefs, dt, tauX, c, z, t):
    sim = np.zeros_like(t)
    for h in range(len(sim) - 1):
        sim[h+1] = sim[h]+(dt/tauX)*(coefs[0]+(coefs[1]*sim[h])+(coefs[2]*sim[h]**2)+(coefs[3]*sim[h]**3)+(coefs[4]*sim[h]**4))+ c*np.sqrt((dt/tauX))*np.random.randn()
#        sim[h+1] = sim[h]+(dt/tauX)*(coefs[0]+(coefs[1]*sim[h])+(coefs[2]*sim[h]**2)+(coefs[3]*sim[h]**3)+(coefs[4]*sim[h]**4)
#                                         +(coefs[5]*sim[h]**5))+ c*np.sqrt((dt/tauX))*np.random.randn()
#        sim[h+1] = sim[h]+(dt/tauX)*(coefs[0]+(coefs[1]*sim[h])+(coefs[2]*sim[h]**2)+(coefs[3]*sim[h]**3)+(coefs[4]*sim[h]**4)
#                                         +(coefs[5]*sim[h]**5)+(coefs[6]*sim[h]**6))+ c*np.sqrt((dt/tauX))*np.random.randn()
        if abs(sim[h]) >= z:
            return sim[:h + 1], h * dt, int(sim[h] >= z)
        elif h>len(sim)-3:
            return sim[:h + 1],  h * dt, 2      
    return sim, h * dt, 2 

def sessionNLB(trials, signal,seeding,coefs):
    #trials - number of trials to be run
    #signal - strenght of drift rate
    #coefs - the average coefficents obtained from single trial
    #seedings -  seeded noise value for repilcability
    
    # Parameter initialization
    epsilon = 0.05 #non biased stimulus input
    b=signal # biased stimulus input
    z = .75 #decison threshold
    tauX = 20 #time constant
    c = 0.01 #noise size
    dt = 0.01 #timestep
    T_Total = 20000 #total time
    sample = 100 #sample  of decision variable activity

    # Data storage initialization
    coef_mat = []
    choice_trials, decision_time, model_data = [], [], []
    sindy_choice, sindy_dt, sindy_data = [], [], []

    # Loop through trials
    for model in range(trials):
        t = np.arange(0, T_Total, dt)
        np.random.seed(model + 20000+seeding)

        # Simulate trial
        x, trial_decision_time, trial_choice = simulate_trial(t, dt, tauX, epsilon, signal, c, z)
        if trial_decision_time is not None:
            decision_time.append(trial_decision_time)
            choice_trials.append(trial_choice)
            if model % sample == 0:
                model_data.append(x)


        sim, sindy_trial_decision_time, sindy_trial_choice = simulate_sindy(coefs, dt, tauX, c, z, t)
        sindy_choice.append(sindy_trial_choice)
        sindy_dt.append(sindy_trial_decision_time)
        if model % sample == 0:
            sindy_data.append(sim)

    return coef_mat, choice_trials, sindy_choice, decision_time, sindy_dt, model_data, sindy_data