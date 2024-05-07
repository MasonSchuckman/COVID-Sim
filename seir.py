import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint

# Updated parameters for COVID-19
beta = 2.5 / 14  # Transmission rate approximated from R0
sigma = 1 / 5    # Inverse of average incubation period in days
gamma = 1 / 14   # Inverse of average infectious period in days
delta = 1 / 365  # Assuming immunity lasts for a year
N = 100000        # Population size

# Initial conditions
I0 = 1
R0 = 0
E0 = 10
S0 = N - I0 - E0 - R0

def get_params(t, beta, sigma, gamma, delta):
    if 300 <= t <= 330:
        beta *= 4
    
    if t > 400:
        gamma *= 0.75

    if t > 800:
        delta /= 3
        sigma *= 1.5

    return beta, sigma, gamma, delta
    
    

# SEIRS model differential equations
def deriv(y, t, N, beta, sigma, gamma, delta):
    S, E, I, R = y
    
    beta, sigma, gamma, delta = get_params(t, beta, sigma, gamma, delta)
    
    dSdt = -beta * S * I / N + delta * R
    dEdt = beta * S * I / N - sigma * E
    dIdt = sigma * E - gamma * I
    dRdt = gamma * I - delta * R
    return dSdt, dEdt, dIdt, dRdt

# Time grid (in days)
t = np.linspace(0, 2000, 200)

# Initial conditions vector
y0 = S0, E0, I0, R0

# Integrate the SEIRS equations over the time grid t
ret = odeint(deriv, y0, t, args=(N, beta, sigma, gamma, delta))
#print(ret)
S, E, I, R = ret.T

# Plotting results
plt.figure(figsize=(10, 6))
plt.plot(t, S, 'b', alpha=0.7, linewidth=2, label='Susceptible')
plt.plot(t, E, 'y', alpha=0.7, linewidth=2, label='Exposed')
plt.plot(t, I, 'r', alpha=0.7, linewidth=2, label='Infectious')
plt.plot(t, R, 'g', alpha=0.7, linewidth=2, label='Recovered')
plt.title("SEIRS Model for COVID-19 Spread")
plt.xlabel('Time (days)')
plt.ylabel('Number of people')
plt.legend()
plt.show()
