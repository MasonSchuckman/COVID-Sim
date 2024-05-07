import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint

class Event:
    def __init__(self, start, end, param_changes):
        self.start = start
        self.end = end
        self.param_changes = param_changes

    def is_active(self, t):
        return self.start <= t < self.end

class SEIRSModel:
    def __init__(self, params, events, hospital_capacity):
        self.params = params
        self.events = events
        self.hospital_capacity = hospital_capacity

    def deriv(self, y, t):
        S, E, I, R, D = y
        
        N = self.params['N'] - D
        beta = self.params['beta']
        sigma = self.params['sigma']
        gamma = self.params['gamma']
        delta = self.params['delta']
        mu = self.params['mu_base']  # Base mortality rate

        # Adjust parameters based on active events
        current_params = self.params.copy()
        for event in self.events:
            if event.is_active(t):
                current_params.update(event.param_changes)
        beta, gamma = current_params['beta'], current_params['gamma']

        # Adjust mortality rate based on hospital capacity
        mu = mu + max(0, (I - self.hospital_capacity) / N * self.params['mu_factor'])

        dSdt = -beta * S * I / N + delta * R
        dEdt = beta * S * I / N - sigma * E
        dIdt = sigma * E - gamma * I - mu * I
        dRdt = gamma * I - delta * R
        dDdt = mu * I
        return dSdt, dEdt, dIdt, dRdt, dDdt

    def simulate(self, y0, t):
        results = odeint(self.deriv, y0, t)
        return t, results

params = {
    'N': 100000,
    'beta': 3 / 14,
    'sigma': 1 / 5,
    'gamma': 1 / 14,
    'delta': 1 / 365,
    'mu_base': 0.001 / 14,  # Base mortality rate per day
    'mu_factor': 0.05  # Additional mortality factor when over capacity
}
hospital_capacity = 2000  # Hospital capacity for serious cases

events = [
    # Holidays
    Event(100, 120, {'beta': 10 / 14}),
    Event(300, 320, {'beta': 10 / 14}),
    Event(500, 520, {'beta': 10 / 14}),
    Event(700, 720, {'beta': 10 / 14}),


    Event(250, 500, {'gamma': 1 / 7})
]

model = SEIRSModel(params, events, hospital_capacity)

# Simulation setup
I0, R0, E0, D0 = 1, 0, 10, 0
S0 = params['N'] - I0 - E0 - R0 - D0
y0 = S0, E0, I0, R0, D0
#t = np.linspace(0, 800, 800)


# Simulation function
def simulate_seirs_events(y0, t):
    #t_final = 1000
    #t = np.linspace(0, t_final, 1000)
    results = []

    for ti in t:

        # Solve for the next step
        t_end, result = model.simulate(y0, [ti, ti + (t[-1] / 1000)])        
        y0 = result[-1]
        #print("y0 = ", y0)
        results.append(y0)

    results = np.array(results)
    return t, results


t = np.linspace(0, 1000, 1000)
print("end t = ", t[-1])
t_end, results = simulate_seirs_events(y0, t)

#print(results)
S, E, I, R, D = results.T

print("Total deaths = ", D[-1])


# Plotting results
plt.figure(figsize=(12, 8))
plt.subplot(2, 1, 1)
plt.plot(t, S, 'b', alpha=0.7, linewidth=2, label='Susceptible')
plt.plot(t, E, 'y', alpha=0.7, linewidth=2, label='Exposed')
plt.plot(t, I, 'r', alpha=0.7, linewidth=2, label='Infectious')
plt.plot(t, R, 'g', alpha=0.7, linewidth=2, label='Recovered')
plt.plot(t, D, 'k', alpha=0.7, linewidth=2, label='Deceased')
plt.axhline(y=hospital_capacity, color='c', linestyle='--', label='Hospital Capacity')
plt.title("Enhanced SEIRS Model with Hospital Capacity and Mortality")
plt.xlabel('Time (days)')
plt.ylabel('Number of people')
plt.legend()

# 100% Stacked Area Chart
plt.subplot(2, 1, 2)
plt.stackplot(t, S/params['N'], E/params['N'], I/params['N'], R/params['N'], D/params['N'], colors=['b', 'y', 'r', 'g', 'k'], labels=['Susceptible', 'Exposed', 'Infectious', 'Recovered', 'Deceased'], alpha=0.5)
plt.title("100% Stacked Area Chart of SEIRS Model")
plt.xlabel('Time (days)')
plt.ylabel('Proportion of Population')
plt.legend(loc='upper right')

plt.tight_layout()
plt.show()
