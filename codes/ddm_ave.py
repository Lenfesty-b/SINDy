import numpy as np
import scipy
import pandas as pd

def simulate_trial(t, dt, A, c, z):
    """
    Simulates a trial for the Drift Diffusion Model (DDM).

    Parameters:
        t (array-like): Time array.
        dt (float): Time step.
        A (float): Drift rate.
        c (float): Size of the noise.
        z (float): Decision threshold.

    Returns:
        x (array-like): Trajectory of the decision variable .
        trial_decision_time (float): Time at which the decision was made.
        trial_choice (int): Indicates whether the positive threshold (1) or negative threshold (0) was reached,
                            or if there was no decision (2).
    """
    x = np.zeros_like(t)
    for p in range(len(x) - 1):
        x[p+1] = x[p] + (dt * A) + c * np.sqrt(dt) * np.random.randn()
        if abs(x[p]) >= z:
            return x[:p + 1], p * dt, int(x[p] >= z)
    return x, p*dt, 2  # No decision case

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
        sim (array-like): Trajectory of the decision variable.
        sindy_trial_decision_time (float): Time at which the decision was made.
        sindy_trial_choice (int): Indicates whether the positive threshold (1) or negative threshold (0) was reached,
                                  or if there was no decision (2).
    """
    sim = np.zeros_like(t)
      # Assuming polynomial order 2
    for h in range(len(sim) - 1):
        sim[h+1] = sim[h] + (dt * coefs) + c * np.sqrt(dt) * np.random.randn()  # Polynomial order 0
        # Uncomment the following lines to use polynomial order 1 or 2
        # sim[h+1] = sim[h] + (dt * coefs[0] + (coef[1] * sim[h])) + c * np.sqrt(dt) * np.random.randn()  # Polynomial order 1
        # sim[h+1] = sim[h] + (dt * coef[0] + (coef[1] * sim[h]) + (coef[2] * sim[h]**2)) + c * np.sqrt(dt) * np.random.randn()  # Polynomial order 2
        if abs(sim[h]) >= z:
            return sim[:h + 1], h * dt, int(sim[h] >= z)
    return sim, h*dt, 2  # No decision case

def sessionDDM(trials, signal, coefs, seedings):
    """
    Runs multiple trials of the DDM and fits the SINDy model to the simulated data.

    Parameters:
        trials (int): Number of trials to be run.
        signal (float): Strength of the drift rate.
        coefs (array-like): The average coefficients obtained from single trial fitting.
        seedings (int): Seed for random number generation to ensure reproducibility.

    Returns:
        coef_mat (list of array-like): List of coefficients for each SINDy model fit.
        choice_trials (list of int): Choices made in each DDM trial.
        sindy_choice (list of int): Choices made in each SINDy trial.
        decision_time (list of float): Decision times for each DDM trial.
        sindy_dt (list of float): Decision times for each SINDy trial.
        model_data (list of array-like): Trajectories of the decision variable for sampled DDM trials.
        sindy_data (list of array-like): Trajectories of the decision variable for sampled SINDy trials.
    """
    # Parameter initialization
    z = 1  # Decision threshold
    c = 0.11  # Size of the noise
    A = signal  # Drift rate
    dt = 0.1  # Time step
    t_total = 10000  # Total trial time
    sample = 20  # Storing trial data

    # Data storage initialization
    coef_mat = []  # Coefficient matrix
    choice_trials, decision_time, model_data = [], [], []
    sindy_choice, sindy_dt, sindy_data = [], [], []

    # Loop through trials
    for model in range(trials):
        t = np.arange(0, t_total, dt)
        np.random.seed(model + 20000 + seedings)

        # Simulate trial
        x, trial_decision_time, trial_choice = simulate_trial(t, dt, A, c, z)
        if trial_decision_time != 2:
            decision_time.append(trial_decision_time)
            choice_trials.append(trial_choice)
            if model % sample == 0:
                model_data.append(x)

        # Simulate SINDy trial using the provided coefficients
        sim, sindy_trial_decision_time, sindy_trial_choice = simulate_sindy(coefs, dt, c, z, t)
        if sindy_trial_decision_time != 2:
            sindy_choice.append(sindy_trial_choice)
            sindy_dt.append(sindy_trial_decision_time)
            if model % sample == 0:
                sindy_data.append(sim)

    return coef_mat, choice_trials, sindy_choice, decision_time, sindy_dt, model_data, sindy_data
##trial average dataset##
#WARNING this takes time to run#
# trials = 10000
# coefs=[-0.0002550803312826866, 0.001639659866047647, 0.0034691280937529764, 0.005352963039526989, 0.007251549049884441, 0.009056453260515641, 0.010782709514096756, 0.012394510935003574, 0.014108232905555188, 0.015722103423218878, 0.017241446683745117, 0.01869445765813389, 0.02024525363971882, 0.0217249808299592, 0.02312440435654821, 0.024437598886822662, 0.02579356287473701, 0.02707672065975406, 0.028293358001629222, 0.029533270369067794, 0.03076281012204078, 0.03194984777631033, 0.033120492873633785, 0.03430631716135452, 0.035411084779392, 0.03652609218450582, 0.03762576158688378, 0.03873546613742232, 0.039810913046592, 0.040891341882716606, 0.04192118231919146, 0.04298019121653966, 0.04402114191380068, 0.045127104731484076, 0.046177931741489125, 0.0472094283939012, 0.04821454082922745, 0.04927210046195163, 0.05027756755098404, 0.051288343256507135, 0.052346361438025006]
#signal=np.arange(0.000, 0.041, 0.001)
# nt_var = [(trials, signal[i],coefs[i]) for i in len(signal)]
# data=[]
# for i in range(len(nt_var)):
#     data.append(sessionDDM(nt_var[i]))

# coefs=[[-0.00053918, -0.04032936, -0.00458844],
#        [ 0.00156725, -0.0403852 ,  0.00586076],
#        [ 0.00351999, -0.03972968,  0.01511209],
#        [ 0.00563896, -0.04081258,  0.02633106],
#        [ 0.00786341, -0.04156398,  0.03739479],
#        [ 0.00991774, -0.04314671,  0.04745402],
#        [ 0.01178705, -0.04298663,  0.05605065],
#        [ 0.01364761, -0.04413727,  0.06375269],
#        [ 0.01549754, -0.04564129,  0.07208164],
#        [ 0.01734395, -0.0464958 ,  0.07967658],
#        [ 0.01898734, -0.04836429,  0.08669995],
#        [ 0.02066174, -0.05068335,  0.09349356],
#        [ 0.02259253, -0.05295282,  0.10194311],
#        [ 0.02434701, -0.05501766,  0.10822454],
#        [ 0.02600845, -0.05653418,  0.11271265],
#        [ 0.02777535, -0.0589243 ,  0.11747253],
#        [ 0.02940903, -0.06004162,  0.12224716],
#        [ 0.03101186, -0.06144016,  0.12661974],
#        [ 0.03266733, -0.06333453,  0.13061663],
#        [ 0.03427615, -0.06613527,  0.13610635],
#        [ 0.03599496, -0.06875437,  0.14099535],
#        [ 0.03771137, -0.07202531,  0.14602144],
#        [ 0.03933534, -0.07480244,  0.15051581],
#        [ 0.04093185, -0.07797296,  0.15487121],
#        [ 0.04243555, -0.08044205,  0.1588453 ],
#        [ 0.04393014, -0.08218686,  0.16112396],
#        [ 0.04539834, -0.08466273,  0.16390413],
#        [ 0.04682777, -0.08690131,  0.16704808],
#        [ 0.04833259, -0.08955188,  0.17013778],
#        [ 0.04979881, -0.09122467,  0.17137561],
#        [ 0.05116182, -0.09207167,  0.1713716 ],
#        [ 0.05260484, -0.09448666,  0.17518411],
#        [ 0.05393215, -0.09558792,  0.17666203],
#        [ 0.05547038, -0.09806739,  0.17978806],
#        [ 0.05678485, -0.09939309,  0.18103432],
#        [ 0.05813763, -0.10074915,  0.18252053],
#        [ 0.05947948, -0.10273574,  0.18441668],
#        [ 0.0608439 , -0.10495684,  0.18702551],
#        [ 0.06209851, -0.10579994,  0.18752142],
#        [ 0.06357617, -0.10871686,  0.19072838],
#        [ 0.06495034, -0.11112724,  0.19380482]]poly2
#       [[0.00016518, 0.04403046],
#       [0.00129728, 0.04408339],
#       [0.0024063 , 0.04409221],
#       [0.00350808, 0.0445293 ],
#       [0.00467506, 0.04454163],
#       [0.00568927, 0.04498385],
#       [0.0067291 , 0.04540364],
#       [0.00771732, 0.04602515],
#       [0.00873235, 0.04653803],
#       [0.00965908, 0.04704829],
#       [0.01049417, 0.04771073],
#       [0.01140517, 0.04829368],
#       [0.01231418, 0.04922995],
#       [0.0132665 , 0.04988675],
#       [0.01418008, 0.05026558],
#       [0.01502725, 0.05080635],
#       [0.01587594, 0.0516393 ],
#       [0.01663784, 0.0520387 ],
#       [0.01740332, 0.05267847],
#       [0.01810349, 0.05343998],
#       [0.0188716 , 0.0543816 ],
#       [0.01957995, 0.05511304],
#       [0.02036539, 0.05580017],
#       [0.02107619, 0.05664274],
#       [0.02186162, 0.05709014],
#       [0.02267404, 0.05762641],
#       [0.02340207, 0.05815924],
#       [0.02423819, 0.05864316],
#       [0.02496421, 0.05926505],
#       [0.02585598, 0.05951667],
#       [0.02674038, 0.05952804],
#       [0.02752125, 0.06023028],
#       [0.02840171, 0.06050982],
#       [0.02916995, 0.06115929],
#       [0.02993404, 0.06157543],
#       [0.03081598, 0.06169639],
#       [0.03160179, 0.06207831],
#       [0.03236171, 0.06262655],
#       [0.03326185, 0.06252106],
#       [0.03399002, 0.06309173],
#       [0.03471744, 0.06385308]]poly1
