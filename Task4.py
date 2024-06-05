import pandas as pd
import numpy as np
from scipy.optimize import minimize

# Load the dataset
data = pd.read_csv('Loan_Data.csv')

# Data inspection and analysis
# Already done in Task3.py

# Define the number of buckets
num_buckets = 10

# Determine the range for FICO scores
fico_min = data['fico_score'].min()
fico_max = data['fico_score'].max()

# Initialize bucket boundaries
bucket_boundaries = np.linspace(fico_min, fico_max, num_buckets + 1)

# Assign each FICO score to the nearest bucket
data['bucket'] = np.digitize(data['fico_score'], bucket_boundaries, right=True) - 1

# Calculate the mean FICO score for each bucket
bucket_means = data.groupby('bucket')['fico_score'].mean()

# Calculate the MSE for the initial buckets
data = data.merge(bucket_means.rename('bucket_mean'), on='bucket')
data['squared_error'] = (data['fico_score'] - data['bucket_mean']) ** 2
initial_mse = data['squared_error'].mean()

# Define a function to calculate MSE given a set of boundaries using pandas DataFrame for mapping
def calculate_mse(boundaries):
    # Ensure boundaries are sorted and include the min and max FICO scores
    boundaries = np.concatenate(([fico_min], np.sort(boundaries), [fico_max]))
    
    # Assign each FICO score to the nearest bucket
    data['bucket'] = np.digitize(data['fico_score'], boundaries, right=True) - 1
    
    # Calculate the mean FICO score for each bucket
    bucket_means = data.groupby('bucket')['fico_score'].mean().to_dict()
    
    # Map bucket means back to the original data
    data['bucket_mean'] = data['bucket'].map(bucket_means)
    
    # Calculate the MSE
    mse = ((data['fico_score'] - data['bucket_mean']) ** 2).mean()
    return mse

# Initial guess for boundaries (excluding the first and last boundaries)
initial_guess = bucket_boundaries[1:-1]

# Perform the optimization to minimize MSE
result = minimize(calculate_mse, initial_guess, method='L-BFGS-B', bounds=[(fico_min, fico_max)] * (num_buckets - 1))

# Extract the optimized boundaries
optimized_boundaries = np.concatenate(([fico_min], np.sort(result.x), [fico_max]))

# Recalculate bucket means and MSE with optimized boundaries
data['bucket'] = np.digitize(data['fico_score'], optimized_boundaries, right=True) - 1
bucket_means = data.groupby('bucket')['fico_score'].mean()
data['bucket_mean'] = data['bucket'].map(bucket_means)
optimized_mse = ((data['fico_score'] - data['bucket_mean']) ** 2).mean()

# Define a function to calculate Log-Likelihood given a set of boundaries
def calculate_log_likelihood(boundaries):
    # Ensure boundaries are sorted and include the min and max FICO scores
    boundaries = np.concatenate(([fico_min], np.sort(boundaries), [fico_max]))
    
    # Assign each FICO score to the nearest bucket
    data['bucket'] = np.digitize(data['fico_score'], boundaries, right=True) - 1
    
    # Calculate the number of records and defaults in each bucket
    bucket_stats = data.groupby('bucket').agg(n_i=('fico_score', 'size'), k_i=('default', 'sum')).reset_index()
    
    # Calculate the probability of default in each bucket
    bucket_stats['p_i'] = bucket_stats['k_i'] / bucket_stats['n_i']
    
    # Calculate the log-likelihood
    log_likelihood = (bucket_stats['k_i'] * np.log(bucket_stats['p_i']) + 
                      (bucket_stats['n_i'] - bucket_stats['k_i']) * np.log(1 - bucket_stats['p_i'])).sum()
    
    # Handle any log(0) cases by setting log-likelihood to a very low value (not valid)
    if np.isinf(log_likelihood) or np.isnan(log_likelihood):
        log_likelihood = -np.inf
    
    return -log_likelihood  # Negative because we minimize in optimization

# Perform the optimization to maximize Log-Likelihood (minimize negative Log-Likelihood)
result = minimize(calculate_log_likelihood, initial_guess, method='L-BFGS-B', bounds=[(fico_min, fico_max)] * (num_buckets - 1))

# Extract the optimized boundaries
optimized_boundaries_ll = np.concatenate(([fico_min], np.sort(result.x), [fico_max]))

# Recalculate bucket stats and Log-Likelihood with optimized boundaries
data['bucket'] = np.digitize(data['fico_score'], optimized_boundaries_ll, right=True) - 1
bucket_stats = data.groupby('bucket').agg(n_i=('fico_score', 'size'), k_i=('default', 'sum')).reset_index()
bucket_stats['p_i'] = bucket_stats['k_i'] / bucket_stats['n_i']
optimized_log_likelihood = (bucket_stats['k_i'] * np.log(bucket_stats['p_i']) + 
                            (bucket_stats['n_i'] - bucket_stats['k_i']) * np.log(1 - bucket_stats['p_i'])).sum()

# Print results
print("Optimized Bucket Boundaries (MSE):", optimized_boundaries)
print("Bucket Means (MSE):", bucket_means)
print("Optimized MSE:", optimized_mse)
print("\nOptimized Bucket Boundaries (Log-Likelihood):", optimized_boundaries_ll)
print("Bucket Statistics (Log-Likelihood):", bucket_stats)
print("Optimized Log-Likelihood:", optimized_log_likelihood)
