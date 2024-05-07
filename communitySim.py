import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint
from matplotlib.colors import Normalize
from matplotlib.animation import FuncAnimation

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
        results = odeint(self.deriv, y0, t_intervals)
        return results[-1]

class Community:
    def __init__(self, model, initial_state):
        self.model = model
        self.state = initial_state

    def update_state(self, new_state):
        self.state = new_state

    def simulate(self, t_start, t_end):
        t_intervals = np.linspace(t_start, t_end, t_end - t_start)
        self.state = self.model.simulate(self.state, t_intervals)

    def get_infected_ratio(self):
        total_population = np.sum(self.state) - self.state[4]  # Exclude deceased from the population
        if total_population > 0:
            return self.state[2] / total_population  # I/N
        return 0

class CommunityGrid:
    def __init__(self, size, model):
        self.size = size
        self.grid = np.array([[Community(model, model.params['initial_state']) for _ in range(size[1])] for _ in range(size[0])])

    def simulate_step(self, t_start, t_end):
        for i in range(self.size[0]):
            for j in range(self.size[1]):
                self.grid[i][j].simulate(t_start, t_end)
        self.interact()

    def interact(self):
        # Movement rate of infectious individuals between communities
        I_exchange_rate = 0.01  # 1% of the infectious population

        # Temporary array to store changes without immediate effect on the grid
        changes = np.zeros((self.size[0], self.size[1], 5))  # Same shape as community state: S, E, I, R, D

        # Calculate changes for each community based on neighboring exchanges
        for i in range(self.size[0]):
            for j in range(self.size[1]):
                current_community = self.grid[i][j]
                current_I = current_community.state[2]

                # Calculate the number of individuals to be moved to each neighbor
                I_to_move = I_exchange_rate * current_I

                # Neighboring positions
                neighbors = [
                    (i - 1, j), (i + 1, j),  # Up, Down
                    (i, j - 1), (i, j + 1)   # Left, Right
                ]

                # For each neighbor, adjust the states
                for ni, nj in neighbors:
                    if 0 <= ni < self.size[0] and 0 <= nj < self.size[1]:
                        # Decrease the infectious individuals from the current community
                        changes[i][j][2] -= I_to_move / 4  # Divide by 4 to distribute to four directions

                        # Increase the infectious individuals in the neighboring community
                        changes[ni][nj][2] += I_to_move / 4

        # Update the community states with the calculated changes
        for i in range(self.size[0]):
            for j in range(self.size[1]):
                self.grid[i][j].state += changes[i][j]

        # Ensure no negative numbers in state variables
        for i in range(self.size[0]):
            for j in range(self.size[1]):
                self.grid[i][j].state = np.maximum(self.grid[i][j].state, 0)

# Parameters and events setup
params = {
    'N': 100000, 'beta': 0.2, 'sigma': 0.1, 'gamma': 0.05, 'delta': 0.001, 'mu_base': 0.0001,
    'mu_factor': 0.05, 'initial_state': (99990, 10, 1, 0, 0)
}
events = [Event(100, 120, {'beta': 0.3})]
hospital_capacity = 1000
model = SEIRSModel(params, events, hospital_capacity)

# Grid and simulation setup
community_grid = CommunityGrid((5, 5), model)
t_intervals = np.arange(0, 800, 10)

# Initialize total_counts with one fewer element than t_intervals
total_counts = np.zeros((len(t_intervals) - 1, 5))  # Adjustment here

for idx, (t_start, t_end) in enumerate(zip(t_intervals[:-1], t_intervals[1:])):
    community_grid.simulate_step(t_start, t_end)
    for i in range(community_grid.size[0]):
        for j in range(community_grid.size[1]):
            total_counts[idx] += community_grid.grid[i][j].state

# Plotting
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 8))

# 100% Stacked Area Chart
ax1.stackplot(t_intervals[:-1], total_counts.transpose(),
              labels=['Susceptible', 'Exposed', 'Infected', 'Recovered', 'Deceased'])
ax1.legend(loc='upper left')
ax1.set_title('Population Over Time')
ax1.set_xlabel('Time')
ax1.set_ylabel('Number of Individuals')

# 2D Grid for Infected Ratio
infected_ratios = np.array([[community_grid.grid[i][j].get_infected_ratio() for j in range(community_grid.size[1])] for i in range(community_grid.size[0])])
norm = Normalize(vmin=0, vmax=1)
img = ax2.imshow(infected_ratios, interpolation='nearest', cmap='viridis', norm=norm)
fig.colorbar(img, ax=ax2)
ax2.set_title('Percentage of Infected Individuals')
ax2.set_xlabel('Community X')
ax2.set_ylabel('Community Y')

plt.tight_layout()
plt.show()


def update(frame, img, grid, ax):
    # Perform the simulation step
    t_start, t_end = t_intervals[frame], t_intervals[frame + 1]
    community_grid.simulate_step(t_start, t_end)

    # Update the data for imshow
    infected_ratios = np.array([[grid[i][j].get_infected_ratio() for j in range(grid.size[1])] for i in range(grid.size[0])])
    img.set_data(infected_ratios)
    ax.set_title(f'Time: {t_start}')
    return img,

# Setup initial state and figure
fig, ax = plt.subplots()
infected_ratios = np.array([[community_grid.grid[i][j].get_infected_ratio() for j in range(community_grid.size[1])] for i in range(community_grid.size[0])])
norm = Normalize(vmin=0, vmax=1)
img = ax.imshow(infected_ratios, interpolation='nearest', cmap='viridis', norm=norm)
fig.colorbar(img, ax=ax)

# Create the animation
ani = FuncAnimation(fig, update, frames=len(t_intervals)-1, fargs=(img, community_grid, ax), interval=50, blit=True)

plt.show()