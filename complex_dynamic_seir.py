import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint

class Event:
    def __init__(self, start, end, param_multipliers):
        self.start = start
        self.end = end
        self.param_multipliers = param_multipliers  # Multipliers for the parameters

    def is_active(self, t):
        return self.start <= t < self.end

class SEIRSModel:
    def __init__(self, params, events, hospital_capacity):
        self.params = params
        self.events = events
        self.hospital_capacity = hospital_capacity

    def deriv(self, y, t):
        S, E, I, R, D = y
        N = S + E + I + R

        # Start with the base parameters
        current_params = self.params.copy()

        # Adjust parameters based on active events
        for event in self.events:

            if event.is_active(t):
                #print(t)
                # Apply multipliers to the parameters
                for param, multiplier in event.param_multipliers.items():
                    
                    current_params[param] *= multiplier

        # Extract the necessary parameters
        beta, sigma, gamma, delta, mu = current_params['beta'], current_params['sigma'], current_params['gamma'], current_params['delta'], current_params['mu_base']

        # Adjust mortality rate based on hospital capacity
        mu += max(0, (I - self.hospital_capacity) / N * self.params['mu_factor'])

        dSdt = -beta * S * I / N + delta * R
        dEdt = beta * S * I / N - sigma * E
        dIdt = sigma * E - gamma * I - mu * I
        dRdt = gamma * I - delta * R
        dDdt = mu * I
        return dSdt, dEdt, dIdt, dRdt, dDdt

    def simulate(self, y0, t, rtol=1e-3, atol=1e-6):
        results = odeint(self.deriv, y0, t, full_output=0, rtol=rtol, atol=atol)
        return results


# Simulation function
def simulate_seirs_events(y0, t):
    return model.simulate(y0, t)
    # results = []

    # for ti in t:

    #     # Solve for the next step
    #     t_end, result = model.simulate(y0, [ti, ti + (t[-1] / 1000)])        
    #     y0 = result[-1]
    #     results.append(y0)

    # results = np.array(results)
    # return t, results

params = {
    'N': 331002647,
    'beta': .2,
    'sigma': 1 / 5,
    'gamma': 1 / 14,
    'delta': 1 / 180,
    'mu_base': 0.001 / 14,  # Base mortality rate per day
    'mu_factor': 0.01  # Additional mortality factor when over capacity
}
hospital_capacity = 200000  # Hospital capacity for serious cases

events = [
    # Lockdowns
    # Event(20, 25, {'beta': .5}), 
    # Event(25, 30, {'beta': .3}),   
    # Event(100, 800, {'gamma': 2}),   
    # Event(50, 70, {'beta': .2}),  
    # Event(70, 100, {'beta': .3}),  
]

# for i in range(160):
#     events.append(Event(30 + i * 5, 30 + (1+i) * 5, {'beta': .2 + i*.01}), )



model = SEIRSModel(params, events, hospital_capacity)

# Simulation setup
I0, R0, E0, D0 = 1, 0, 10, 0
S0 = params['N'] - I0 - E0 - R0 - D0
y0 = S0, E0, I0, R0, D0
#t = np.linspace(0, 800, 800)




MAX_T = 800
t = np.linspace(0, MAX_T, MAX_T)
print("end t = ", t[-1])
results = simulate_seirs_events(y0, t)

S, E, I, R, D = results.T

# Calculate daily new exposures and infections
new_E = E#np.diff(E, prepend=E[0])
new_I = I#np.diff(I, prepend=I[0])

plt.figure(figsize=(12, 10))
plt.subplot(3, 1, 1)
plt.plot(t, S, 'b', alpha=0.7, linewidth=2, label='Susceptible')
plt.plot(t, E, 'y', alpha=0.7, linewidth=2, label='Exposed')
plt.plot(t, I, 'r', alpha=0.7, linewidth=2, label='Infectious')
plt.plot(t, R, 'g', alpha=0.7, linewidth=2, label='Recovered')
plt.plot(t, D, 'k', alpha=0.7, linewidth=2, label='Deceased')
plt.axhline(y=hospital_capacity, color='c', linestyle='--', label='Hospital Capacity')
#plt.title("Enhanced SEIRS Model with Hospital Capacity and Mortality")
plt.title("SEIRS Model with Mortality and Hospital Capacity")

plt.xlabel('Time (days)')
plt.ylabel('Number of People')
plt.legend()

plt.subplot(3, 1, 2)
plt.stackplot(t, S/params['N'], E/params['N'], I/params['N'], R/params['N'], D/params['N'], colors=['b', 'y', 'r', 'g', 'k'], labels=['Susceptible', 'Exposed', 'Infectious', 'Recovered', 'Deceased'], alpha=0.5)
plt.title("100% Stacked Area Chart of SEIRS Model")
plt.xlabel('Time (days)')
plt.ylabel('Proportion of Population')
plt.legend(loc='upper right')

plt.subplot(3, 1, 3)
plt.plot(t, new_E, 'y--', label='New Exposures')
plt.plot(t, new_I, 'r--', label='New Infections')
plt.title("New Daily Cases (Exposures and Infections)")
plt.xlabel('Time (days)')
plt.ylabel('Number of New Cases')
plt.legend()

plt.tight_layout()
plt.show()