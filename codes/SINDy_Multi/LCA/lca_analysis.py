import pickle
import numpy as np
import pysindy as ps


def fit_sindy_model(x,poly_order):
    """
    Fits a SINDy model to the given data and returns the model coefficients.

    Parameters:
        x (array-like): The data to fit, where each row is a time point and each column is a variable.
        t (array-like): The time points corresponding to the rows of x.

    Returns:
        array-like: The coefficients of the fitted SINDy model.
    """
    poly_order = poly_order
    threshold = 0.001#0.001 
    smoothed_fd=ps.SmoothedFiniteDifference()
    
    modelLCA = ps.SINDy(
        optimizer=ps.STLSQ(threshold=threshold),
        feature_library=ps.PolynomialLibrary(degree=poly_order),
        differentiation_method=smoothed_fd
    )
    modelLCA.fit(x, t=0.01,multiple_trajectories=True)
    
    return np.squeeze(modelLCA.coefficients())


combined_data = [[] for _ in range(41)]  # Initialize as a list of empty lists

# Load original data in chunks and extract necessary columns
with open('lca_timecourse_combined.pkl', 'rb') as f:
    original_data = pickle.load(f)

for snr_index in range(len(original_data)):
    original_snr = original_data[snr_index]
    
    # Extract choice and decision time from the original data, ignoring the large third column
    original_choice = original_snr[0]  # Assume choice is in the first position
    original_time = original_snr[1]    # Assume time is in the second position
    fitted_coefs =([np.array(original_snr[2][i]).T for i in range(10000)]) 

    # Store extracted data temporarily
    combined_data[snr_index].append(original_choice)
    combined_data[snr_index].append(original_time)
    combined_data[snr_index].append(fit_sindy_model(fitted_coefs,1))
    combined_data[snr_index].append(fit_sindy_model(fitted_coefs,0))
    combined_data[snr_index].append(fit_sindy_model(fitted_coefs,2))

# Clear original_data from memory
del original_data

# Now load fitted data and extract necessary columns
with open('lca_timecourse_multi_combined.pkl', 'rb') as f:
    fitted_data = pickle.load(f)

for snr_index in range(len(fitted_data)):
    fitted_snr = fitted_data[snr_index]
    
    # Extract choice and decision time from the fitted data
    fitted_choice = fitted_snr[0]   # Fitted choice in the first position
    fitted_time = fitted_snr[1]    # Fitted time in the second position
    # Update combined data with fitted values
    combined_data[snr_index].append(fitted_choice)
    combined_data[snr_index].append(fitted_time)
    

# Clear fitted_data from memory
del fitted_data,fitted_choice,fitted_time,fitted_snr,fitted_coefs

# Now load fitted data and extract necessary columns
with open('lca_timecourse_multi0_combined.pkl', 'rb') as f:
    fitted_data = pickle.load(f)

for snr_index in range(len(fitted_data)):
    fitted_snr = fitted_data[snr_index]
    
    # Extract choice and decision time from the fitted data
    fitted_choice = fitted_snr[0]   # Fitted choice in the first position
    fitted_time = fitted_snr[1]    # Fitted time in the second position

    # Update combined data with fitted values
    combined_data[snr_index].append(fitted_choice)
    combined_data[snr_index].append(fitted_time)

# Clear fitted_data from memory
del fitted_data,fitted_choice,fitted_time,fitted_snr

# Now load fitted data and extract necessary columns
with open('lca_timecourse_multi2_combined.pkl', 'rb') as f:
    fitted_data = pickle.load(f)

for snr_index in range(len(fitted_data)):
    fitted_snr = fitted_data[snr_index]
    
    # Extract choice and decision time from the fitted data
    fitted_choice = fitted_snr[0]   # Fitted choice in the first position
    fitted_time = fitted_snr[1]    # Fitted time in the second position

    # Update combined data with fitted values
    combined_data[snr_index].append(fitted_choice)
    combined_data[snr_index].append(fitted_time)

# Clear fitted_data from memory
del fitted_data,fitted_choice,fitted_time,fitted_snr

# Save the final combined dataset to a file
with open('lca_combined_multi_beh_coefs.pkl', 'wb') as f:
    pickle.dump(combined_data, f)

print("Data combined and saved successfully.")