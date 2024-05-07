import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation
from datetime import datetime, timedelta
from matplotlib.colors import Normalize

class County:
    def __init__(self, population, initial_infected, handwashing_freq, mandate_adherence, travel_level):
        self.population = population
        self.susceptible = population - initial_infected
        self.exposed = 0  # Initial exposed individuals
        self.infected = initial_infected
        self.recovered = 0
        self.handwashing_freq = handwashing_freq
        self.mandate_adherence = mandate_adherence
        self.travel_level = travel_level

    def update_status(self, adj_counties, current_date):
        # Apply event effects
        self.apply_events(current_date)
        
        # Calculate infections from travel
        travelers_infected = 0
        for county in adj_counties:
            travelers = county.travel_level * county.infected / 8
            travelers_infected += travelers * (1 - self.handwashing_freq)
        
        # Exposure rate
        new_exposures = 0.10 * self.susceptible * self.infected / self.population# + travelers_infected
        new_exposures = min(new_exposures, self.susceptible)
        
        # Transition from exposed to infected
        new_infections = 1/5 * self.exposed
        new_infections = min(new_infections, self.exposed)
        
        # Recovery rate
        new_recoveries = 1/14 * self.infected
        new_recoveries = min(new_recoveries, self.infected)
        
        # Loss of immunity rate
        new_susceptible = 1/365 * self.recovered
        new_susceptible = min(new_susceptible, self.recovered)

        # Update counts
        self.susceptible -= new_exposures
        self.susceptible += new_susceptible
        self.exposed += new_exposures - new_infections
        self.infected += new_infections - new_recoveries
        self.recovered += new_recoveries - new_susceptible

    def apply_events(self, current_date):
        self.travel_level = 0.01
        self.mandate_adherence = .8

    def get_infected_ratio(self):
        if self.population > 0:
            return self.infected / self.population
        return 0

# Simulation parameters
N = 10  # Grid size
steps = 200
start_date = datetime(2020, 12, 1)

# Initialize the grid
grid = [[County(10000, np.random.randint(1, 100), 0.0, 0.0, 0.1) for _ in range(N)] for _ in range(N)]

# Track total SEIRS
total_susceptible = []
total_exposed = []
total_infected = []
total_recovered = []

def update(frame_num, img, grid, N, current_date, ax2):
    new_grid = [[grid[i][j].get_infected_ratio() for j in range(N)] for i in range(N)]
    total_S, total_E, total_I, total_R = 0, 0, 0, 0
    print(current_date + timedelta(days=frame_num))
    for i in range(N):
        for j in range(N):
            adj_counties = []
            for di in (-1, 0, 1):
                for dj in (-1, 0, 1):
                    if di == 0 and dj == 0:
                        continue
                    ni, nj = (i + di) % N, (j + dj) % N
                    adj_counties.append(grid[ni][nj])
            grid[i][j].update_status(adj_counties, current_date + timedelta(days=frame_num))
            total_S += grid[i][j].susceptible
            total_E += grid[i][j].exposed
            total_I += grid[i][j].infected
            total_R += grid[i][j].recovered

    total_susceptible.append(total_S)
    total_exposed.append(total_E)
    total_infected.append(total_I)
    total_recovered.append(total_R)

    # Update the area chart
    ax2.clear()
    ax2.stackplot(range(len(total_susceptible)), total_susceptible, total_exposed, total_infected, total_recovered, labels=['Susceptible', 'Exposed', 'Infected', 'Recovered'], colors=['skyblue', 'yellow', 'salmon', 'lightgreen'])
    ax2.legend(loc='upper right')
    ax2.set_xlim(0, steps)
    ax2.set_ylim(0, N * N * 10000)
    ax2.set_title('Total SEIRS Dynamics Over Time')

    img.set_data(new_grid)
    return img,

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 8))
norm = Normalize(vmin=0, vmax=1)
img = ax1.imshow([[grid[i][j].get_infected_ratio() for j in range(N)] for i in range(N)],
                 interpolation='nearest', cmap='viridis', norm=norm)
ani = animation.FuncAnimation(fig, update, fargs=(img, grid, N, start_date, ax2),
                              frames=steps, interval=1, save_count=1)

plt.tight_layout()
plt.show()
