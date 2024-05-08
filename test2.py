import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint
import ipywidgets as widgets
from IPython.display import display

# Model and parameters
class SEIRSModel:
    def __init__(self, params, hospital_capacity):
        self.params = params
        self.hospital_capacity = hospital_capacity

    def deriv(self, y, t, beta, sigma, gamma, delta, mu_base):
        S, E, I, R, D = y
        N = S + E + I + R
        mu = mu_base + max(0, (I - self.hospital_capacity) / N * self.params['mu_factor'])
        dSdt = -beta * S * I / N + delta * R
        dEdt = beta * S * I / N - sigma * E
        dIdt = sigma * E - gamma * I - mu * I
        dRdt = gamma * I - delta * R
        dDdt = mu * I
        return dSdt, dEdt, dIdt, dRdt, dDdt

    def simulate(self, y0, t, beta, sigma, gamma, delta, mu_base):
        return odeint(self.deriv, y0, t, args=(beta, sigma, gamma, delta, mu_base))

params = {
    'mu_factor': 0.01,  # Mortality rate factor
    'N': 331002647  # Population
}
hospital_capacity = 2000
model = SEIRSModel(params, hospital_capacity)

# Initial conditions
I0, R0, E0, D0 = 1, 0, 0, 0
S0 = params['N'] - I0 - E0 - R0 - D0
y0 = S0, E0, I0, R0, D0
t = np.linspace(0, 160, 160)
slider = widgets.FloatSlider(value=7.5, min=0, max=10.0, step=0.1)
display(slider)
@widgets.interact
def update(beta=(0.01, 1.0, 0.01), sigma=(0.1, 1.0, 0.01), gamma=(0.01, 0.1, 0.01), delta=(0.0, 1.0, 0.01), mu_base=(0.0, 0.01, 0.0001)):
    results = model.simulate(y0, t, beta, sigma, gamma, delta, mu_base)
    S, E, I, R, D = results.T

    plt.figure(figsize=(10, 6))
    plt.plot(t, S, label='Susceptible')
    plt.plot(t, E, label='Exposed')
    plt.plot(t, I, label='Infectious')
    plt.plot(t, R, label='Recovered')
    plt.plot(t, D, label='Deceased')
    plt.title('SEIRS Model Simulation')
    plt.xlabel('Time (days)')
    plt.ylabel('Number of people')
    plt.legend()
    plt.grid(True)
    plt.show()
