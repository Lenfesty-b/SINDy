import numpy as np
import scipy
import pandas as pd


def simulate_trial(t, dt, A, c, z):
    ##simulates trial for ddm
    
    x = np.zeros_like(t)
    for p in range(len(x) - 1):
        x[p+1] = x[p]+(dt*A)+ c*np.sqrt(dt)*np.random.randn()
        if abs(x[p]) >= z:
            return x[:p + 1], p * dt, int(x[p] >= z)
    return None, 2, 2

def simulate_sindy(coefs, dt, c, z, t):
    ##simulates trial for SINDy model
    
    sim = np.zeros_like(t)
    coef_0,coef_1,coef_2=coefs
    for h in range(len(sim) - 1):
        sim[h+1] = sim[h]+(dt*coefs)+ c*np.sqrt(dt)*np.random.randn() #Polynomial order 0
        #sim[h+1] = sim[h]+(dt*coef_0+(coef_1*sim[h]))+ c*np.sqrt((dt))*np.random.randn()#Polynomial order 1
#         sim[h+1] = sim[h]+(dt*coef_0+(coef_1*sim[h])+(coef_2*sim[h]**2))+ c*np.sqrt((dt))*np.random.randn()#Polynomial order 2
        if abs(sim[h]) >= z:
            return sim[:h + 1], h * dt, int(sim[h] >= z)
    return None, 2, 2

def sessionDDM(trials, signal,coefs,seedings):
    #trials - number of trials to be run
    #signal - strenght of drift rate
    #coefs - the average coefficents obtained from single trial
    #seedings -  seeded noise value for repilcability
    
    # Parameter initialization
    
    z=1 # decision threshold
    c=0.11  # Size of the noise
    A=signal  # Drift rate
    x0=0  # initial condition of decision variable x
    dt=0.1  # Time step
    t_total=10000 #total trial time
    trial_total=trials
    sample=20 #storing trial data

    # Data storage initialization
    coef_mat = [] #coefficent matrix
    choice_trials, decision_time, model_data = [], [], []
    sindy_choice, sindy_dt, sindy_data = [], [], []

    # Loop through trials
    for model in range(trial_total):
        t = np.arange(0, t_total, dt)
        np.random.seed(model + 20000+seedings)

        # Simulate trial
        x, trial_decision_time, trial_choice = simulate_trial(t, dt,A, c, z)
        if trial_decision_time is not None:
            decision_time.append(trial_decision_time)
            choice_trials.append(trial_choice)
            if model % sample == 0:
                model_data.append(x)
        coefs=coefs
        sim, sindy_trial_decision_time, sindy_trial_choice = simulate_sindy(coefs, dt,  c, z, t)
        if sindy_trial_decision_time is not None:
            sindy_choice.append(sindy_trial_choice)
            sindy_dt.append(sindy_trial_decision_time)
            if model % sample == 0:
                sindy_data.append(sim)

    return coef_mat, choice_trials, sindy_choice, decision_time, sindy_dt, model_data, sindy_data