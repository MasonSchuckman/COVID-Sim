import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation

INFECTION_RATE = 0.0001
RECOVERY_RATE = 0.1

class County:
    def __init__(self, population, initial_infected, handwashing_freq, mandate_adherence, travel_level):
        self.susceptible = population - initial_infected
        self.infected = initial_infected
        self.recovered = 0
        self.handwashing_freq = handwashing_freq
        self.mandate_adherence = mandate_adherence
        self.travel_level = travel_level

    def update_status(self):
        # Model infection process
        new_infections = 0
        if self.susceptible != 0:
            new_infections = (self.infected * self.susceptible * (1 - self.handwashing_freq) *
                            (1 - self.mandate_adherence) * INFECTION_RATE)
        new_recoveries = self.infected * RECOVERY_RATE  # Recovery rate
        
        # Update counts
        new_infections = min(new_infections, self.susceptible)
        new_recoveries = min(new_recoveries, self.infected)

        self.susceptible -= new_infections
        self.infected += new_infections - new_recoveries
        self.recovered += new_recoveries

    def get_infected_ratio(self):
        if self.susceptible + self.infected + self.recovered > 0:
            return self.infected / (self.susceptible + self.infected + self.recovered)
        else:
            return 0

# Simulation parameters
N = 25  # Grid size
steps = 200

# Initialize the grid
grid = [[County(10000, np.random.randint(1, 10), 0.5, 0.5, 0.01) for _ in range(N)] for _ in range(N)]

def update(frame_num, img, grid, N):
    new_grid = [[grid[i][j].get_infected_ratio() for j in range(N)] for i in range(N)]
    
    # Update each county
    for i in range(N):
        for j in range(N):
            grid[i][j].update_status()

    img.set_data(new_grid)
    return img,

fig, ax = plt.subplots()
img = ax.imshow([[grid[i][j].get_infected_ratio() for j in range(N)] for i in range(N)],
                interpolation='nearest', cmap='viridis')
ani = animation.FuncAnimation(fig, update, fargs=(img, grid, N),
                              frames=steps, interval=50, save_count=50)

plt.show()
