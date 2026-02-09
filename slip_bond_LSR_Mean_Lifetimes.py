import numpy as np
from scipy.optimize import differential_evolution

# Lifetime function from logistic PDE
def L(force, tau, A, B, C, D):
    return A*np.exp(B*tau - C*force) / (1+ D*np.exp(B*tau))

# Data
force_values = np.array([13.4, 17.8, 23.8, 29.9, 35.4, 48.2, 14.9, 22.8, 30.3, 38.1, 44.2, 54.9, 68.7]) / 2.8
tau_values = np.array([0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 3, 3, 3, 3, 3, 3, 3])
L_values = np.array([0.047, 0.044, 0.048, 0.043, 0.039, 0.038, 0.14, 0.101, 0.052, 0.032, 0.04, 0.04, 0.03])

# Objective function for DE
def objective(params):
    A, B, C, D = params

    # Skip invalid parameter values
    if any(p <= 0 for p in params):
        return np.inf

    L_pred = L(force_values, tau_values, A, B, C, D)
    residuals = L_pred - L_values
    
    # Want to minimize the sum of squared residuals
    error = np.sum(residuals**2)
    return error

# Parameter bounds
bounds = [(1e-4, 2), (1e-3, 10), (1e-4, 2), (1e-3, 10)]

# DE settings
result = differential_evolution(objective, bounds, strategy='best1bin', maxiter=2000, popsize=500, tol=1e-7, mutation=(0.5,1.99), recombination=0.7, workers=-1, updating='deferred', polish=True, disp = True)

print("Optimal Parameters:")
A_opt, B_opt, C_opt, D_opt = result.x
print(f"A = {A_opt:.12f}")
print(f"B = {B_opt:.12f}")
print(f"C = {C_opt:.12f}")
print(f"D = {D_opt:.12f}")
print(f"Total weighted error = {result.fun:.6e}")