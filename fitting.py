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
def simulate_seirs_events(model, y0, t):
    return model.simulate(y0, t)
    # results = []

    # for ti in t:

    #     # Solve for the next step
    #     t_end, result = model.simulate(y0, [ti, ti + (t[-1] / 1000)])        
    #     y0 = result[-1]
    #     results.append(y0)

    # results = np.array(results)
    # return t, results
pop=331002647
params = {
    'N': 331002647,
    'beta': 2,
    'sigma': 1 / 5,
    'gamma': 1 / 14,
    'delta': 1 / 365,
    'mu_base': 0.001 / 14,  # Base mortality rate per day
    'mu_factor': 0.00  # Additional mortality factor when over capacity
}
hospital_capacity = 2000  # Hospital capacity for serious cases

events = [
    # Lockdowns
    #Event(20, 25, {'gamma': 1}), 
    #Event(70, 600, {'gamma': 1.5, 'beta': 0.7}),   
    # Event(50, 70, {'beta': .2}),  
    # Event(70, 100, {'beta': .3}),  

    # 2022 Christmas
    Event(715, 725, {'gamma': 0.8, 'beta': 1.1}), 

    Event(725, 740, {'gamma': 0.5, 'beta': 1.5}), 
    Event(750, 765, {'gamma': 2, 'beta': 0.8}), 
    Event(765, 775, {'gamma': 1.5, 'beta': 0.9}), 

]

# for i in range(16):
#     events.append(Event(60 + i * 5, 60 + (1+i) * 5, {'gamma': .2 + i*.05}), )
# print(events)
model = SEIRSModel(params, events, hospital_capacity)





import pandas as pd
import matplotlib.pyplot as plt
from scipy.integrate import odeint
from scipy.optimize import minimize


# Load the data
df = pd.read_csv('H:/downloads/US.csv')
# Convert 'date' to datetime and sort just in case
df['date'] = pd.to_datetime(df['date'])
df.sort_values('date', inplace=True)

# Calculate 7-day average of new confirmed cases
df['confirmed_7d_avg'] = df['new_confirmed'].rolling(window=7).sum()



# Simulation setup
I0, R0, E0, D0 = 1e5, 2e8, 1.2e5, 0
S0 = params['N'] - I0 - E0 - R0 - D0
y0 = S0, E0, I0, R0, D0
print("N = {}".format(params['N']))

t_fit_start = 620
t_fit_end = 850
t = np.linspace(t_fit_start, t_fit_end, t_fit_end - t_fit_start + 1)
#simulate_seirs_events(model, y0, np.linspace(0, t_fit_start, t_fit_start + 1))[-1]
actual_data = df['confirmed_7d_avg'].iloc[t_fit_start:t_fit_end + 1].values



# MAX_T = 991
# t = np.linspace(0, MAX_T, MAX_T)
# print("end t = ", t[-1])
# t_end, results = simulate_seirs_events(y0, t)

# S, E, I, R, D = results.T

# # Calculate daily new exposures and infections
# new_E = np.diff(E, prepend=E[0])
# new_I = np.diff(I, prepend=I[0])


# Assuming new_confirmed data aligns with new_I from your model
# Data from simulation
#model_output = results[:, 2]  # Assuming I (infectious) is in the third column from results

# Actual data (you might need to adjust indexing based on your dataframe)
#actual_data = df['confirmed_7d_avg'].values

# Objective function to minimize
def objective_function(params):
    # Update model parameters
    beta, sigma, gamma, mu_base, I0, R0, E0, D0 = params
    model.params.update({
        'beta': beta,
        'sigma': sigma,
        'gamma': gamma,
        'mu_base': mu_base
    })
    S0 = pop - I0 - E0 - R0 - D0
    y0 = S0, E0, I0, R0, D0
    #print(y0)
    model_results = simulate_seirs_events(model, y0, t)
    S, E, I, R, D = np.array(model_results).T
    model_I = (1.5*I)  # Extracting only the Infectious column
    #print("SSE = {}".format(model_I))
    # Calculate the sum of squared differences
    print(np.sum(I - actual_data) / len(I))
    return np.sum(((model_I - actual_data)**2))

# Initial parameter guesses
initial_guess = [params['beta'], params['sigma'], params['gamma'], params['mu_base'], I0, R0, E0, D0]

# Bounds for the parameters to ensure they are positive and within a reasonable range
bounds = [(.1, 3), (0.1, 1/4), (0.1, 1/5), (0.0000001, 0.01), (5e5,1e6),(1e7,2e9),(5e5,1e6),(1e4,1e7)]

# Run the optimization
result = minimize(objective_function, initial_guess, bounds=bounds, method='L-BFGS-B', options={'maxiter': 100000})

# Print optimized parameters
print("Result : ", result)
print("Optimized parameters:", result.x)
params.update({
        'beta': result.x[0],
        'sigma': result.x[1],
        'gamma': result.x[2],
        'mu_base': result.x[3]
    })
#print(params)



model = SEIRSModel(params, events, hospital_capacity)


# Simulation setup
I0, R0, E0, D0 = result.x[4], result.x[5], result.x[6], result.x[7]
S0 = pop - I0 - E0 - R0 - D0
y0 = S0, E0, I0, R0, D0
#t = np.linspace(0, 800, 800)




#MAX_T = t_fit_end - t_fit_start + 1
#t = np.linspace(0, MAX_T, MAX_T)
print("end t = ", t[-1])
results = simulate_seirs_events(model, y0, t)

S, E, I, R, D = results.T

# Calculate daily new exposures and infections
new_E = E#np.max(np.diff(E, prepend=E[0]), np.zeros(len(E)))
new_I = I#np.max(np.diff(I, prepend=I[0]), np.zeros(len(E)))

plt.figure(figsize=(12, 10))
plt.subplot(3, 1, 1)
plt.plot(t, S, 'b', alpha=0.7, linewidth=2, label='Susceptible')
plt.plot(t, E, 'y', alpha=0.7, linewidth=2, label='Exposed')
plt.plot(t, I, 'r', alpha=0.7, linewidth=2, label='Infectious')
plt.plot(t, R, 'g', alpha=0.7, linewidth=2, label='Recovered')
plt.plot(t, D, 'k', alpha=0.7, linewidth=2, label='Deceased')
plt.axhline(y=hospital_capacity, color='c', linestyle='--', label='Hospital Capacity')
plt.title("Enhanced SEIRS Model with Hospital Capacity and Mortality")
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
plt.plot(t, actual_data, 'b--', label='Real Data')
#plt.plot(t, new_E, 'y--', label='New Exposures')
plt.plot(t, new_I, 'r--', label='Predicted Data')
plt.title("New Daily Cases (Exposures and Infections)")
plt.xlabel('Time (days)')
plt.ylabel('Number of New Cases')
plt.legend()

plt.tight_layout()
plt.show()