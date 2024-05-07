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
        mu = self.params['mu_base']

        # Event adjustments
        current_params = self.params.copy()
        for event in self.events:
            if event.is_active(t):
                current_params.update(event.param_changes)
        beta, gamma = current_params['beta'], current_params['gamma']

        # Mortality adjustment based on capacity
        mu = mu + max(0, (I - self.hospital_capacity) / N * self.params['mu_factor'])

        dSdt = -beta * S * I / N + delta * R
        dEdt = beta * S * I / N - sigma * E
        dIdt = sigma * E - gamma * I - mu * I
        dRdt = gamma * I - delta * R
        dDdt = mu * I
        return dSdt, dEdt, dIdt, dRdt, dDdt

    def simulate(self, y0, t_intervals):
        results = []
        y = y0
        for t_start, t_end in zip(t_intervals[:-1], t_intervals[1:]):
            t_span = np.linspace(t_start, t_end, t_end - t_start)
            y = odeint(self.deriv, y, t_span)
            results.append(y)
            y = y[-1]  # Set the last result as the new initial condition
        return np.concatenate(results)

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

# Create a grid of models
grid_size = (5, 5)  # 5x5 grid of communities
models = np.empty(grid_size, dtype=object)
for i in range(grid_size[0]):
    for j in range(grid_size[1]):
        # Each model can have different parameters or events
        models[i, j] = SEIRSModel(params, events, hospital_capacity)

# Simulation time setup
t_intervals = np.arange(0, 800, 10)  # Simulate in 10-day intervals

# Initialize state for each community
state_grid = np.empty(grid_size, dtype=object)
for i in range(grid_size[0]):
    for j in range(grid_size[1]):
        S0, E0, I0, R0, D0 = params['N'] - 1 - 10, 10, 1, 0, 0
        y0 = S0, E0, I0, R0, D0
        state_grid[i, j] = y0

# Define interaction
def interact(state_grid, i, j):
    directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]  # Up, Down, Left, Right
    I_exchange_rate = 0.01  # 1% of the infectious population can move between counties
    for di, dj in directions:
        ni, nj = i + di, j + dj
        if 0 <= ni < grid_size[0] and 0 <= nj < grid_size[1]:
            I_to_move = int(I_exchange_rate * state_grid[i, j][2])
            state_grid[i, j] = (state_grid[i, j][0], state_grid[i, j][1], state_grid[i, j][2] - I_to_move, state_grid[i, j][3], state_grid[i, j][4])
            state_grid[ni, nj] = (state_grid[ni, nj][0], state_grid[ni, nj][1], state_grid[ni, nj][2] + I_to_move, state_grid[ni, nj][3], state_grid[ni, nj][4])

# Simulate each community step-by-step and apply interactions
for t_start, t_end in zip(t_intervals[:-1], t_intervals[1:]):
    print("start = ", t_start, " end = ", t_end)
    for i in range(grid_size[0]):
        for j in range(grid_size[1]):
            current_state = state_grid[i, j]
            new_state = models[i, j].simulate(current_state, [t_start, t_end])
            state_grid[i, j] = new_state[-1]  # Update state
            if i == j == 0:
                print(state_grid[i, j])

    # Apply interactions after each step
    for i in range(grid_size[0]):
        for j in range(grid_size[1]):
            interact(state_grid, i, j)