import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint

class Event:
    def __init__(self, start, end, param_changes):
        self.start = start
        self.end = end
        self.param_changes = param_changes

    def apply_event(self, params):
        for key, value in self.param_changes.items():
            params[key] = value
        return params

# Parameters for COVID-19, can be adjusted dynamically
params = {
    'beta': 2.5 / 14,  # Transmission rate
    'sigma': 1 / 5,    # Incubation period
    'gamma': 1 / 14,   # Infectious period
    'delta': 1 / 365,  # Loss of immunity
    'N': 100000        # Population size
}

# Initial conditions
I0 = 1
R0 = 0
E0 = 10
S0 = params['N'] - I0 - E0 - R0
y0 = S0, E0, I0, R0

# Differential equations for the SEIRS model
def deriv(y, t, params):
    S, E, I, R = y
    N, beta, sigma, gamma, delta = params['N'], params['beta'], params['sigma'], params['gamma'], params['delta']
    dSdt = -beta * S * I / N + delta * R
    dEdt = beta * S * I / N - sigma * E
    dIdt = sigma * E - gamma * I
    dRdt = gamma * I - delta * R
    return dSdt, dEdt, dIdt, dRdt

# List of events
events = [
    
    Event(500, 1000, {'beta': 1.5 / 14}),   # Increased beta from day 500 to 1000
    Event(1000, 1500, {'gamma': 1 / 7})     # Improved recovery rate from day 1000 to 1500
]

# Simulation function
def simulate_seirs_events(y0, params, events):
    t_final = 2000
    t = np.linspace(0, t_final, 1000)
    results = []
    current_params = params.copy()

    for ti in t:
        # Check for any ongoing events
        for event in events:
            if event.start <= ti < event.end:
                current_params = event.apply_event(current_params.copy())
            elif event.end <= ti:
                current_params = params.copy()  # Reset to original after event ends
        
        # Solve for the next step
        result = odeint(deriv, y0, [ti, ti + (t_final / 1000)], args=(current_params,))
        y0 = result[-1]
        results.append(result[-1])

    results = np.array(results)
    return t, results

# Running the dynamic simulation with events
t, results = simulate_seirs_events(y0, params, events)
S, E, I, R = results.T

# Plotting results
plt.figure(figsize=(10, 6))
plt.plot(t, S, 'b', alpha=0.7, linewidth=2, label='Susceptible')
plt.plot(t, E, 'y', alpha=0.7, linewidth=2, label='Exposed')
plt.plot(t, I, 'r', alpha=0.7, linewidth=2, label='Infectious')
plt.plot(t, R, 'g', alpha=0.7, linewidth=2, label='Recovered')
plt.title("Dynamic SEIRS Model with Events for COVID-19 Spread")
plt.xlabel('Time (days)')
plt.ylabel('Number of people')
plt.legend()
plt.show()