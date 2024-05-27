import numpy as np
import scipy
import pysindy as ps
import pandas as pd


def simulate_trial(t, dt, S1,S2,b,k, c, z):
    ##simulates trial for LCA-DDM model
    x = np.zeros_like(t)
    y = np.zeros_like(t)
    
    for p in range(len(x) - 1):
        #print(p)
        x[p+1] = x[p]+dt*(-k*x[p]-b*y[p]+S1) + c*np.sqrt(dt)*np.random.randn()
        y[p+1] = y[p]+dt*(-k*y[p]-b*x[p]+S2) + c*np.sqrt(dt)*np.random.randn()
        
        if x[p] >= z or y[p]>=z :
            return x[0:p + 1],y[0:p+1], p * dt, int(x[p] >= z)
        
    return None, None, 2, 2

def fit_sindy_model(x, t):
    
    modelLCA.fit(x, t)
    return np.squeeze(modelLCA.coefficients())

def simulate_sindy(coefs, dt, c, z, t):
    ##simulates trial for SINDy model
    sim_x = np.zeros_like(t)
    sim_y = np.zeros_like(t)
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
       sim_x[h+1] = sim_x[h]+dt*(coef_1*sim_x[h]+coef_2*sim_y[h]+coef_0) + c*np.sqrt(dt)*np.random.randn()#poly1
       sim_y[h+1] = sim_y[h]+dt*(coef_5*sim_y[h]+coef_4*sim_x[h]+coef_3) + c*np.sqrt(dt)*np.random.randn()#poly1
#        sim_x[h+1] = sim_x[h]+(dt*coef_1) + c*np.sqrt(dt)*np.random.randn() #poly0
#        sim_y[h+1] = sim_y[h]+(dt*coef_2) + c*np.sqrt(dt)*np.random.randn() #poly0             
#         sim_x[h+1] = sim_x[h]+dt*(coef_0*sim_x[h]+coef_1*sim_y[h]+coef_2+(coef_3*sim_x[h]**2)+(coef_4*sim_x[h]*sim_y[h])+(coef_5*sim_y[h]**2)) + c*np.sqrt(dt)*np.random.randn()#poly2
#         sim_y[h+1] = sim_y[h]+dt*(coef_6*sim_y[h]+coef_7*sim_x[h]+coef_8+(coef_9*sim_x[h]**2)+(coef_10*sim_x[h]*sim_y[h])+(coef_11*sim_y[h]**2)) + c*np.sqrt(dt)*np.random.randn()#poly2  

        if sim_x[h] >= z or sim_y[h] >= z :
            return sim_x[:h + 1],sim_y[:h + 1], h * dt, int(sim_x[h] >= z)
    
    return None, None, 2, 2

def sessionLCADDM(trials, signal,coefs,seedings):
    #trials - number of trials to be run
    #signal - strenght of drift rate
    #coefs - the average coefficents obtained from single trial
    #seedings -  seeded noise value for repilcability
    
    # Parameter initialization
    S1=3+signal #Stimulus input amplitude to y1 
    S2=3        #Stimulus input amplitude to y2 
    b=10 #Mutual inhibitory coupling strength between the y's 4
    k=10 # Rate of decay of the y's 3
    z=1  # Decision threshold

    c=.11 # Size of the noise
    dt=.01 #timestep
    Model_total=trials# number of trials
    T_Total=10000 #Total time
    sample=42 #storing trial data

    # Data storage initialization
    coef_mat = []
    choice_trials, decision_time, model_data = [], [], []
    sindy_choice, sindy_dt, sindy_data = [], [], []

    # Loop through trials
    for model in range(trials):
        t = np.arange(0, T_Total, dt)
        np.random.seed(model + 20000+seedings)

        # Simulate trial
        x,y, trial_decision_time, trial_choice = simulate_trial(t, dt,S1,S2,b,k, c, z)
        if trial_decision_time is not None:
            decision_time.append(trial_decision_time)
            choice_trials.append(trial_choice)
            if model % sample == 0:
                model_data.append([x,y])
                

        coefs=coefs
        sim_x,sim_y, sindy_trial_decision_time, sindy_trial_choice = simulate_sindy(coefs, dt,  c, z, t)
        if sindy_trial_decision_time is not None:
            sindy_choice.append(sindy_trial_choice)
            sindy_dt.append(sindy_trial_decision_time)
            if model % sample == 0:
                sindy_data.append([sim_x,sim_y])

    return coef_mat, choice_trials, sindy_choice, decision_time, sindy_dt, model_data, sindy_data