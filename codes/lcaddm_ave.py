import numpy as np
import scipy
import pysindy as ps
import pandas as pd


def simulate_trial(t, dt, S1, S2, b, k, c, z):
    """
    Simulates a trial for the LCA-DDM model.

    Parameters:
        t (array-like): Time array.
        dt (float): Time step.
        S1 (float): Stimulus input amplitude to x.
        S2 (float): Stimulus input amplitude to y.
        b (float): Mutual inhibitory coupling strength between x and y.
        k (float): Rate of decay for x and y.
        c (float): Size of the noise.
        z (float): Decision threshold.

    Returns:
        x (array-like): Trajectory of x the decision variable.
        y (array-like): Trajectory of y the decision variable.
        trial_decision_time (float): Time at which the decision was made.
        trial_choice (int): Indicates whether x (1) or y (0) reached the threshold first or if there was no decision (2).
    """
    x = np.zeros_like(t)
    y = np.zeros_like(t)
    
    for p in range(len(x) - 1):
        x[p+1] = x[p] + dt * (-k * x[p] - b * y[p] + S1) + c * np.sqrt(dt) * np.random.randn()
        y[p+1] = y[p] + dt * (-k * y[p] - b * x[p] + S2) + c * np.sqrt(dt) * np.random.randn()
        
        if x[p] >= z or y[p] >= z:
            return x[0:p + 1], y[0:p + 1], p * dt, int(x[p] >= z)
    
    return None, None, p*dt, 2

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
    sim_x = np.zeros_like(t)
    sim_y = np.zeros_like(t)
    coef_0, coef_1, coef_2, coef_3, coef_4, coef_5= coefs
    
    for h in range(len(sim_x) - 1):
        sim_x[h+1] = sim_x[h] + dt * (coef_1 * sim_x[h] + coef_2 * sim_y[h] + coef_0) + c * np.sqrt(dt) * np.random.randn()  # Poly order 1
        sim_y[h+1] = sim_y[h] + dt * (coef_5 * sim_y[h] + coef_4 * sim_x[h] + coef_3) + c * np.sqrt(dt) * np.random.randn()  # Poly order 1
        
        # Uncomment the following lines to use polynomial order 0 or 2
        # sim_x[h+1] = sim_x[h] + dt * coef_1 + c * np.sqrt(dt) * np.random.randn()  # Poly order 0
        # sim_y[h+1] = sim_y[h] + dt * coef_2 + c * np.sqrt(dt) * np.random.randn()  # Poly order 0
        
        # sim_x[h+1] = sim_x[h] + dt * (coef_0 * sim_x[h] + coef_1 * sim_y[h] + coef_2 + (coef_3 * sim_x[h]**2) + (coef_4 * sim_x[h] * sim_y[h]) + (coef_5 * sim_y[h]**2)) + c * np.sqrt(dt) * np.random.randn()  # Poly order 2
        # sim_y[h+1] = sim_y[h] + dt * (coef_6 * sim_y[h] + coef_7 * sim_x[h] + coef_8 + (coef_9 * sim_x[h]**2) + (coef_10 * sim_x[h] * sim_y[h]) + (coef_11 * sim_y[h]**2)) + c * np.sqrt(dt) * np.random.randn()  # Poly order 2

        if sim_x[h] >= z or sim_y[h] >= z:
            return sim_x[:h + 1], sim_y[:h + 1], h * dt, int(sim_x[h] >= z)
    
    return None, None, h*dt, 2

def sessionLCADDM(trials, signal, coefs, seedings):
    """
    Runs multiple trials of the LCA-DDM model and fits the SINDy model to the simulated data.

    Parameters:
        trials (int): Number of trials to be run.
        signal (float): Strength of the drift rate.
        coefs (array-like): The average coefficients obtained from single trial fitting.
        seedings (int): Seed for random number generation to ensure reproducibility.

    Returns:
        coef_mat (list of array-like): List of coefficients for each SINDy model fit.
        choice_trials (list of int): Choices made in each LCA-DDM trial.
        sindy_choice (list of int): Choices made in each SINDy trial.
        decision_time (list of float): Decision times for each LCA-DDM trial.
        sindy_dt (list of float): Decision times for each SINDy trial.
        model_data (list of list of array-like): Trajectories of x and y for sampled LCA-DDM trials.
        sindy_data (list of list of array-like): Trajectories of x and y for sampled SINDy trials.
    """
    # Parameter initialization
    S1 = 3 + signal  # Stimulus input amplitude to y1 
    S2 = 3  # Stimulus input amplitude to y2 
    b = 10  # Mutual inhibitory coupling strength between the y's
    k = 10  # Rate of decay of the y's
    z = 1  # Decision threshold
    c = 0.11  # Size of the noise
    dt = 0.01  # Timestep
    T_Total = 10000  # Total time
    sample = 42  # Storing trial data

    # Data storage initialization
    coef_mat = []
    choice_trials, decision_time, model_data = [], [], []
    sindy_choice, sindy_dt, sindy_data = [], [], []

    # Loop through trials
    for model in range(trials):
        t = np.arange(0, T_Total, dt)
        np.random.seed(model + 20000 + seedings)

        # Simulate trial
        x, y, trial_decision_time, trial_choice = simulate_trial(t, dt, S1, S2, b, k, c, z)
        if trial_decision_time != 2:
            decision_time.append(trial_decision_time)
            choice_trials.append(trial_choice)
            if model % sample == 0:
                model_data.append([x, y])
                
        # Simulate SINDy trial using the provided coefficients
        sim_x, sim_y, sindy_trial_decision_time, sindy_trial_choice = simulate_sindy(coefs, dt, c, z, t)
        if sindy_trial_decision_time != 2:
            sindy_choice.append(sindy_trial_choice)
            sindy_dt.append(sindy_trial_decision_time)
            if model % sample == 0:
                sindy_data.append([sim_x, sim_y])

    return coef_mat, choice_trials, sindy_choice, decision_time, sindy_dt, model_data, sindy_data

##trial average dataset##
#WARNING this takes time to run#
# trials = 10000
# coefs=[[0.17747871867215675, -0.5531563723936244, -0.584503576738585, 0.18021292558819674, -0.593383044120184, -0.5620566080478654], [0.17862896244954773, -0.5550002523141973, -0.5856886272815048, 0.18040191058270655, -0.5963731835756152, -0.5640232522292504], [0.1800527496350613, -0.5580577233609252, -0.5880140618397781, 0.18017002501157633, -0.5978225072086387, -0.5646090087152044], [0.18125693350448147, -0.5603369154482609, -0.5895065005232842, 0.18016059564118395, -0.5998841150507638, -0.5660237506505624], [0.18306087694889844, -0.5646249335386115, -0.5931900347119116, 0.18027418155058061, -0.6023722677862433, -0.5675828391602199], [0.18526145421401885, -0.570140701033317, -0.5979665075502292, 0.18151213476367903, -0.6084499585729676, -0.5726978757046319], [0.1874906557975659, -0.5756036778091043, -0.6028489647714123, 0.18258869605290082, -0.6141725141826888, -0.5774039882844239], [0.18992390049600497, -0.5817597343379554, -0.608554929199205, 0.1843306412000883, -0.6219204997841228, -0.5841769346946016], [0.1930554717490067, -0.5901718312450952, -0.6163909391228537, 0.18681997624370375, -0.6318521819180793, -0.5930947736979507], [0.19664457821637146, -0.5998979377059405, -0.6258470720090963, 0.1896744485823969, -0.6430494472530236, -0.6031081917630252], [0.2006402847400654, -0.6112294357515771, -0.6368618276241909, 0.19262092645128318, -0.654348219744605, -0.6132617468491108], [0.2049176158410598, -0.6232887545526473, -0.6485087223430994, 0.1963170996753955, -0.6681115620167216, -0.6259624634157078], [0.20925257930677954, -0.6356890083992459, -0.6606275047855724, 0.19958251743870722, -0.6802305082579696, -0.636978294044777], [0.21380135894766272, -0.6488679781436636, -0.6734802615948552, 0.20348688130640785, -0.6944690223161538, -0.6499653091231974], [0.21847183711719254, -0.6623701820134719, -0.6864050859986022, 0.2070532919709679, -0.7074924801692933, -0.6622191981504036], [0.22327440455764014, -0.6763147966336146, -0.6999512801899852, 0.2112508665378779, -0.7224685713871906, -0.6760908171653706], [0.22819526495632572, -0.6905983290028742, -0.7137535759940185, 0.21545977159414423, -0.7375870479657788, -0.6901274439980224], [0.2334402225880393, -0.705902176587296, -0.7286685896034436, 0.2197651599521378, -0.752913184749161, -0.704508391026003], [0.23954768235578522, -0.7236835765630967, -0.7461895142016456, 0.22475785563038744, -0.7707532694250471, -0.7211093699218555], [0.24482573559326828, -0.7389383606257356, -0.7610499305917431, 0.22945522258680576, -0.7875148463647591, -0.7367556362185255], [0.2505911793790434, -0.7559350456775789, -0.7779070563605922, 0.23407873728638937, -0.8038266089724091, -0.7519047329774238], [0.25646877184746136, -0.7732511957672465, -0.7948031712375843, 0.238583455784662, -0.8196200471078073, -0.7666957966742277], [0.2624161889477768, -0.7908314884568551, -0.8122314280857299, 0.24414528310121292, -0.8388824787569898, -0.7847455257077207], [0.267957775898203, -0.80704457728014, -0.8281053999516745, 0.24892545794281895, -0.8554789172288125, -0.8003210143188967], [0.27346086122042296, -0.8231477724583952, -0.8437010960692505, 0.2539314855539964, -0.8728833148021027, -0.816845797451413], [0.2796936450628329, -0.8416532374534949, -0.8619344000900777, 0.25923815747583356, -0.8911537953878409, -0.8338355776614772], [0.28605977736009247, -0.8604663097992635, -0.8805273378815333, 0.264855091322289, -0.9104694679270063, -0.8520796689391411], [0.2919022672972454, -0.8776403259119981, -0.8972993936212419, 0.27016715595417723, -0.9287866659460736, -0.8694803773559016], [0.29833957686611157, -0.8967673482030905, -0.9160364249364533, 0.27559697923777404, -0.9473973754193387, -0.8870709015628908], [0.3045221738611268, -0.9149607421434696, -0.9340697138989154, 0.2808116681199676, -0.9654919344881635, -0.9041229157508546], [0.31113808164419954, -0.934523886877993, -0.9532058254247854, 0.28656686887162053, -0.9852206006406917, -0.9229752543142471], [0.31735608888906025, -0.9528266533645355, -0.9709114500656085, 0.29220473711082356, -1.0045625329037449, -0.9416834212301411], [0.32394626033539486, -0.9722478663350679, -0.9901389985438281, 0.29776681238349495, -1.0237765720097385, -0.9599720769780382], [0.3301989563962627, -0.9906486008362843, -1.00806014455851, 0.30306339897829754, -1.0419666108379684, -0.9774365911777292], [0.33652335902811364, -1.0093024595759834, -1.0263336133824859, 0.3089904947379634, -1.0620819750505779, -0.9967271978510934], [0.3428347183737526, -1.0278647092770952, -1.044474769629416, 0.31434759332452383, -1.0805081975655586, -1.0143828034141351], [0.3492359812386792, -1.0466576418859008, -1.0629281102614883, 0.32016372536943244, -1.1003833832652394, -1.0334342965926935], [0.35609227668171195, -1.0670204526955278, -1.0831564200599808, 0.3259488390907494, -1.1200387762322142, -1.0519638832339226], [0.36271986127755146, -1.086489713789304, -1.1022923079288047, 0.3313047032910616, -1.1385881785267247, -1.0693675427884821], [0.36924831795330054, -1.1057410644790766, -1.1212304112676739, 0.3369577542207768, -1.1578361805615696, -1.0878975633671053], [0.37573240240582406, -1.124842766120545, -1.1399420915383893, 0.3428564228012467, -1.177908781736894, -1.1072627510540638]]
# signal=np.arange(0.000, 0.041, 0.001)
# nt_var = [(trials, signal[i],coefs[i]) for i in len(signal)]
# data=[]
# for i in range(len(nt_var)):
#     data.append(sessionLCADDM(nt_var[i]))


    # coefs=[[  2.4333602 , -15.73016016, -15.70908075,  25.26200496,
    #      50.48936058,  25.27032184,   2.44948037, -15.79633805,
    #     -15.82376585,  25.38963588,  50.74127369,  25.39786946],
    #    [  2.43802158, -15.75887375, -15.70998912,  25.31519825,
    #      50.50664789,  25.24162435,   2.45269701, -15.8351485 ,
    #     -15.83788104,  25.46681182,  50.82681914,  25.3993571 ],
    #    [  2.44187428, -15.78765019, -15.7153694 ,  25.3778974 ,
    #      50.55460243,  25.23437861,   2.45595415, -15.87071414,
    #     -15.84833206,  25.53297865,  50.88999927,  25.38969195],
    #    [  2.44908259, -15.83830157, -15.74087141,  25.4746958 ,
    #      50.66694708,  25.25580464,   2.46026925, -15.91201562,
    #     -15.86395222,  25.60639006,  50.96696979,  25.38786483],
    #    [  2.45786689, -15.89626252, -15.77495216,  25.57767661,
    #      50.79637269,  25.28860172,   2.46707293, -15.97223838,
    #     -15.89879069,  25.71608014,  51.11523968,  25.42152938],
    #    [  2.46798923, -15.96149932, -15.81497215,  25.69234391,
    #      50.94351514,  25.32861935,   2.4765532 , -16.04596896,
    #     -15.94685295,  25.8406994 ,  51.29271932,  25.46788649],
    #    [  2.47718578, -16.02254663, -15.84789086,  25.8019757 ,
    #      51.07161235,  25.35446097,   2.48595936, -16.12343808,
    #     -15.99885743,  25.97651464,  51.49643436,  25.52889321],
    #    [  2.49359036, -16.12802687, -15.92939032,  25.97899956,
    #      51.34869836,  25.46125285,   2.50095256, -16.23224771,
    #     -16.08234964,  26.15690183,  51.78725584,  25.63406291],
    #    [  2.51200842, -16.2460069 , -16.02461024,  26.17611339,
    #      51.6693879 ,  25.5914521 ,   2.5188309 , -16.35564253,
    #     -16.18474582,  26.35486873,  52.12587585,  25.76879008],
    #    [  2.53314002, -16.38243216, -16.13452356,  26.40163873,
    #      52.03876082,  25.74039206,   2.53736368, -16.48450719,
    #     -16.29157453,  26.56593945,  52.48327176,  25.91128351],
    #    [  2.55014015, -16.49042651, -16.22150165,  26.57981232,
    #      52.32910197,  25.85943027,   2.55463965, -16.60484893,
    #     -16.38878591,  26.760964  ,  52.80870137,  26.03643424],
    #    [  2.57268372, -16.63218833, -16.33923858,  26.8084332 ,
    #      52.7124191 ,  26.01947302,   2.577321  , -16.7525495 ,
    #     -16.51671909,  26.99062632,  53.20926528,  26.20557842],
    #    [  2.59453237, -16.77009212, -16.45251112,  27.03107368,
    #      53.08215427,  26.17303103,   2.597192  , -16.88425438,
    #     -16.62828743,  27.19835413,  53.56464738,  26.34920471],
    #    [  2.61816433, -16.91833348, -16.57680968,  27.26969982,
    #      53.48378089,  26.34337485,   2.61969914, -17.03311963,
    #     -16.7573657 ,  27.43306479,  53.97790679,  26.5236474 ],
    #    [  2.63897998, -17.0457364 , -16.68416913,  27.46998972,
    #      53.82349294,  26.48695827,   2.63976271, -17.16622162,
    #     -16.87081598,  27.64301185,  54.3396956 ,  26.67420825],
    #    [  2.66260237, -17.19122493, -16.80579697,  27.69905632,
    #      54.20848957,  26.64984737,   2.66302271, -17.31963348,
    #     -17.00437308,  27.88500974,  54.76506529,  26.85325426],
    #    [  2.68453496, -17.32442985, -16.91857224,  27.9070384 ,
    #      54.56166777,  26.80060573,   2.6859962 , -17.46788252,
    #     -17.13411279,  28.1139109 ,  55.16793123,  27.02552377],
    #    [  2.70541591, -17.45132703, -17.02391567,  28.10440633,
    #      54.89244837,  26.93936226,   2.70588949, -17.59801273,
    #     -17.24422008,  28.3160468 ,  55.51196578,  27.16602407],
    #    [  2.73205206, -17.61379999, -17.16625599,  28.35793132,
    #      55.33850543,  27.13698347,   2.73200374, -17.76589958,
    #     -17.39520996,  28.57507552,  55.9810905 ,  27.37422235],
    #    [  2.75316949, -17.73915905, -17.27108139,  28.54887666,
    #      55.65852282,  27.2716089 ,   2.75312666, -17.90305237,
    #     -17.51570444,  28.78758466,  56.35630527,  27.53556797],
    #    [  2.7784577 , -17.89357409, -17.40598713,  28.78822239,
    #      56.07891763,  27.45704638,   2.77577749, -18.04906715,
    #     -17.64530058,  29.01397321,  56.75992574,  27.71159944],
    #    [  2.80017347, -18.02176531, -17.51518919,  28.98135307,
    #      56.40836831,  27.59860124,   2.79746213, -18.18785939,
    #     -17.76728991,  29.22672661,  57.1351016 ,  27.87189794],
    #    [  2.82457892, -18.16815946, -17.64075007,  29.20418183,
    #      56.79296488,  27.76589158,   2.82078926, -18.33523801,
    #     -17.9012645 ,  29.450664  ,  57.54354421,  28.05540369],
    #    [  2.84688361, -18.30171228, -17.75589608,  29.40792801,
    #      57.14644778,  27.91971358,   2.84230358, -18.47127099,
    #     -18.02089446,  29.65655842,  57.90478912,  28.20983651],
    #    [  2.86694934, -18.41757681, -17.85436414,  29.57887599,
    #      57.43643299,  28.04382302,   2.86110977, -18.59062483,
    #     -18.12437417,  29.83744493,  58.21778842,  28.34116161],
    #    [  2.88773375, -18.53886918, -17.95907494,  29.75880514,
    #      57.74881764,  28.18161027,   2.88279224, -18.72648371,
    #     -18.24455533,  30.04177348,  58.57761924,  28.49465182],
    #    [  2.91171697, -18.67963823, -18.08063827,  29.96839107,
    #      58.11228021,  28.34136859,   2.90490279, -18.86522447,
    #     -18.36653469,  30.25088653,  58.94484074,  28.65231186],
    #    [  2.93125934, -18.79273698, -18.17913986,  30.13511254,
    #      58.4041397 ,  28.47008288,   2.92467487, -18.98750197,
    #     -18.47577978,  30.43227618,  59.26684779,  28.79290426],
    #    [  2.95036007, -18.90140571, -18.27334095,  30.29237787,
    #      58.67798966,  28.59094851,   2.94448982, -19.11124642,
    #     -18.58535795,  30.6173042 ,  59.59292586,  28.93319016],
    #    [  2.96992449, -19.01625896, -18.36962364,  30.46476279,
    #      58.9684437 ,  28.71399852,   2.9633137 , -19.22720095,
    #     -18.68887324,  30.78705588,  59.89461402,  29.0641428 ],
    #    [  2.98964057, -19.12804112, -18.46974441,  30.62571766,
    #      59.25801553,  28.84520085,   2.9809447 , -19.33510014,
    #     -18.78406875,  30.94456941,  60.17141455,  29.18476941],
    #    [  3.00797631, -19.23200356, -18.55536277,  30.7762139 ,
    #      59.50548596,  28.94532684,   2.99948087, -19.44963366,
    #     -18.88538606,  31.11307664,  60.46862792,  29.31474829],
    #    [  3.02704064, -19.3400419 , -18.64974447,  30.9314547 ,
    #      59.77792187,  29.06700752,   3.01733706, -19.55786961,
    #     -18.98209782,  31.26998022,  60.74784491,  29.43780481],
    #    [  3.04434428, -19.43733129, -18.73269834,  31.07087733,
    #      60.01584026,  29.16878551,   3.03329404, -19.65418521,
    #     -19.06732108,  31.40740602,  60.98841384,  29.54124435],
    #    [  3.06238552, -19.53887346, -18.81886137,  31.21582029,
    #      60.26170227,  29.27285932,   3.05088548, -19.76017243,
    #     -19.1626831 ,  31.55935281,  61.26032761,  29.66298854],
    #    [  3.07956128, -19.63383188, -18.90117345,  31.34906686,
    #      60.49197464,  29.37241638,   3.06678635, -19.8554258 ,
    #     -19.24560203,  31.69425963,  61.49386902,  29.76289884],
    #    [  3.09636703, -19.7280519 , -18.9812213 ,  31.4836066 ,
    #      60.72126518,  29.47120484,   3.08260426, -19.94978973,
    #     -19.32735842,  31.82738297,  61.72170305,  29.85929052],
    #    [  3.11224626, -19.8164914 , -19.06055969,  31.6087454 ,
    #      60.94740528,  29.57644752,   3.0973393 , -20.03768023,
    #     -19.40178471,  31.95137625,  61.92751578,  29.94064616],
    #    [  3.12735247, -19.89983758, -19.13316284,  31.72648291,
    #      61.15334583,  29.66809524,   3.11109457, -20.11990994,
    #     -19.47081734,  32.06617473,  62.1170807 ,  30.01434322],
    #    [  3.14292558, -19.98523695, -19.20459524,  31.84512126,
    #      61.35278413,  29.75374013,   3.1259144 , -20.2073501 ,
    #     -19.54635016,  32.1880819 ,  62.32394198,  30.10029156],
    #    [  3.15792587, -20.06618174, -19.27378289,  31.95540664,
    #      61.54095765,  29.83477849,   3.13926305, -20.28432701,
    #     -19.61309673,  32.29135526,  62.50047654,  30.17546605]]poly2
    #[[ 2.86095780e-03,  2.77084232e-03],
#       [ 3.84527267e-03,  1.79729367e-03],
#       [ 4.74376968e-03,  9.17800803e-04],
#       [ 5.67521060e-03,  8.64456511e-06],
#       [ 6.56794831e-03, -8.38208482e-04],
#       [ 7.49449500e-03, -1.71273711e-03],
#       [ 8.43372694e-03, -2.60373433e-03],
#       [ 9.34049351e-03, -3.43629667e-03],
#       [ 1.02140286e-02, -4.23951812e-03],
#       [ 1.11081177e-02, -5.03900902e-03],
#       [ 1.19836946e-02, -5.80904978e-03],
#       [ 1.28360768e-02, -6.51641250e-03],
#       [ 1.36409264e-02, -7.19673844e-03],
#       [ 1.44576479e-02, -7.87637549e-03],
#       [ 1.52300928e-02, -8.51376499e-03],
#       [ 1.59881446e-02, -9.13055512e-03],
#       [ 1.67375165e-02, -9.72712973e-03],
#       [ 1.74830521e-02, -1.03050416e-02],
#       [ 1.82611788e-02, -1.09134773e-02],
#       [ 1.90153131e-02, -1.14901668e-02],
#       [ 1.97391960e-02, -1.20465143e-02],
#       [ 2.04661304e-02, -1.26103801e-02],
#       [ 2.11745001e-02, -1.31229667e-02],
#       [ 2.18512998e-02, -1.36262275e-02],
#       [ 2.25319022e-02, -1.41307744e-02],
#       [ 2.32156429e-02, -1.46418246e-02],
#       [ 2.39205819e-02, -1.51292199e-02],
#       [ 2.45512876e-02, -1.55729719e-02],
#       [ 2.52113388e-02, -1.60489429e-02],
#       [ 2.58897557e-02, -1.65247859e-02],
#       [ 2.65447927e-02, -1.69772452e-02],
#       [ 2.72123816e-02, -1.74389073e-02],
#       [ 2.78571484e-02, -1.78686102e-02],
#       [ 2.84564581e-02, -1.82807354e-02],
#       [ 2.90773263e-02, -1.86804786e-02],
#       [ 2.96763693e-02, -1.90820045e-02],
#       [ 3.03127317e-02, -1.94859066e-02],
#       [ 3.09852136e-02, -1.99495165e-02],
#       [ 3.16521284e-02, -2.04368508e-02],
#       [ 3.22836379e-02, -2.08390917e-02],
#       [ 3.28860351e-02, -2.12172871e-02]] poly0
