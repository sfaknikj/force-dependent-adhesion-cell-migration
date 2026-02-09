import numpy as np
from scipy.optimize import least_squares

# Lifetime as function of Force
def L(force, A, B):
    return A *np.exp(force) + B

# Data points to match
force_values = np.array([2.5, 4.64, 7.32, 10.18])
L_values = np.array([0.029, 0.024, 0.028, 0.088])

# Define residual function with adjusted weighting
def residuals(params, force_values, L_values):
    A, B = params
    
    # Penalize negative values on A, B
    penalty = 0
    for param in [A, B]:
        if param < 0:
            penalty += 100000 * (-param)

    # Compute model values
    model_values = np.array([L(force_values[i], A, B) for i in range(len(force_values)) ])

    # Define residuals
    residuals = model_values - L_values
    
    # Return residuals
    return np.append(residuals, penalty)

# Initial guesses
initial_guess = [0.0001, 0.1]

# Perform least squares regression
result = least_squares(residuals, initial_guess, args=(force_values, L_values))

# Extract optimized parameters and round to 10 decimal places
A_opt, B_opt = result.x

# Print results
print(f"Optimized parameters: A={A_opt}, B={B_opt}")