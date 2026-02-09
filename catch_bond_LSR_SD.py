import numpy as np
from scipy.optimize import least_squares

# Read in data points
lifetimes_03 = np.array([0.048, 0.047, 0.044, 0.043, 0.039, 0.038])
lifetimes_3 = np.array([0.140, 0.101, 0.052, 0.040, 0.040, 0.032, 0.030])
sd_03 = np.array([0.04, 0.034, 0.042, 0.038, 0.036, 0.036])
sd_3 = np.array([0.089, 0.084, 0.049, 0.033, 0.039, 0.022, 0.029])

# Define pooled lifetimes and standard deviation lists
lifetimes_pooled = np.concatenate((lifetimes_03, lifetimes_3))
sd_pooled = np.concatenate((sd_03, sd_3))

# Define SD as function of Lifetime
def SD(lifetime, A, B):
    return A * lifetime + B

# Define residual function
def residuals(params, lifetimes_pooled, sd_pooled):
    A, B = params

    # Compute model values
    model_values = np.array([SD(lifetimes_pooled[i], A, B) for i in range(len(lifetimes_pooled)) ])

    # Define residuals
    residuals = model_values - sd_pooled
    
    # Return residuals
    return residuals

# Initial guesses
initial_guess = [1, 0.01]

# Perform least squares optimization
result = least_squares(residuals, initial_guess, args=(lifetimes_pooled, sd_pooled))

# Extract parameter values
A_opt, B_opt = result.x

# Print results
print(f"Optimized parameters: A={A_opt}, B={B_opt}")