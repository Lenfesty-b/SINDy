import pickle 
import time
import pysindy as ps
import numpy as np
import argparse
from joblib import Parallel, delayed 
import os
import faulthandler

faulthandler.enable()
trial_amount=4000
# Now load fitted data and extract necessary columns
with open('/mnt/scratch2/users/blenfesty/nlb_timecourse_combined.pkl', 'rb') as f:
    fitted_data = pickle.load(f)


combined_data = [[] for _ in range(len(fitted_data))]  # Initialize as a list of empty lists

for snr_index in range(len(fitted_data)):

    # SINDy setup
    poly_order = 5  # Polynomial order for SINDy model
    threshold = 0.000001  # Hyperparameter to control level of sensitivity
    smoothed_fd = ps.SmoothedFiniteDifference()  # Differentiation method to smooth noise

    modelNLB = ps.SINDy(
       optimizer=ps.STLSQ(threshold=threshold),
       feature_library=ps.PolynomialLibrary(degree=poly_order),
       differentiation_method=smoothed_fd
    )

    fitted_snr = fitted_data[snr_index]
    # Extract choice and decision time from the fitted data
    fitted_choice = fitted_snr[0]   # Fitted choice in the first position
    fitted_time = fitted_snr[1]    # Fitted time in the second position
    fitted_coefs = fitted_snr[2]    # Fitted time in the second position

   # Update combined data with fitted values
    combined_data[snr_index].append(fitted_choice)
    combined_data[snr_index].append(fitted_time)
    modelNLB.fit(fitted_coefs[0:trial_amount], t=0.01 / 20, multiple_trajectories=True)
    combined_data[snr_index].append(np.squeeze(modelNLB.coefficients()))
    del fitted_choice,fitted_time,fitted_coefs

# Save the final combined dataset to a file
with open('nlb_combined_coefs.pkl', 'wb') as f:
    pickle.dump(combined_data, f)
del combined_data

combined_data = [[] for _ in range(len(fitted_data))]  # Initialize as a list of empty lists

for snr_index in range(len(fitted_data)):

    # SINDy setup
    poly_order = 4  # Polynomial order for SINDy model
    threshold = 0.000001  # Hyperparameter to control level of sensitivity
    smoothed_fd = ps.SmoothedFiniteDifference()  # Differentiation method to smooth noise

    modelNLB = ps.SINDy(
       optimizer=ps.STLSQ(threshold=threshold),
       feature_library=ps.PolynomialLibrary(degree=poly_order),
       differentiation_method=smoothed_fd
    )

    fitted_snr = fitted_data[snr_index]
    # Extract choice and decision time from the fitted data
    fitted_choice = fitted_snr[0]   # Fitted choice in the first position
    fitted_time = fitted_snr[1]    # Fitted time in the second position
    fitted_coefs = fitted_snr[2]    # Fitted time in the second position

   # Update combined data with fitted values
    combined_data[snr_index].append(fitted_choice)
    combined_data[snr_index].append(fitted_time)
    modelNLB.fit(fitted_coefs[0:trial_amount], t=0.01 / 20, multiple_trajectories=True)
    combined_data[snr_index].append(np.squeeze(modelNLB.coefficients()))
    del fitted_choice,fitted_time,fitted_coefs
    
# Save the final combined dataset to a file
with open('nlb_combined_coefs_poly4.pkl', 'wb') as f:
    pickle.dump(combined_data, f)
del combined_data


combined_data = [[] for _ in range(len(fitted_data))]  # Initialize as a list of empty lists

for snr_index in range(len(fitted_data)):

    # SINDy setup
    poly_order = 5  # Polynomial order for SINDy model
    threshold = 0.000001  # Hyperparameter to control level of sensitivity
    smoothed_fd = ps.SmoothedFiniteDifference()  # Differentiation method to smooth noise

    modelNLB = ps.SINDy(
       optimizer=ps.STLSQ(threshold=threshold),
       feature_library=ps.PolynomialLibrary(degree=poly_order),
       differentiation_method=smoothed_fd
    )

    fitted_snr = fitted_data[snr_index]
    # Extract choice and decision time from the fitted data
    fitted_choice = fitted_snr[0]   # Fitted choice in the first position
    fitted_time = fitted_snr[1]    # Fitted time in the second position
    fitted_coefs = fitted_snr[2]    # Fitted time in the second position

   # Update combined data with fitted values
    combined_data[snr_index].append(fitted_choice)
    combined_data[snr_index].append(fitted_time)
    modelNLB.fit(fitted_coefs[0:trial_amount], t=0.01 / 20, multiple_trajectories=True)
    combined_data[snr_index].append(np.squeeze(modelNLB.coefficients()))
    del fitted_choice,fitted_time,fitted_coefs

# Save the final combined dataset to a file
with open('nlb_combined_coefs_poly6.pkl', 'wb') as f:
    pickle.dump(combined_data, f)

print("Data combined and saved successfully")