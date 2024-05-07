import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation
from datetime import datetime, timedelta
from matplotlib.colors import Normalize
from scipy.integrate import odeint
import math

INTERVAL_LENGTH = 2



def sigmoid(x):
  return 1 / (1 + math.exp(-x))

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
        N = S+E+I+R
        #N = self.params['N'] - D
        # if(t%10 == 0):
        #     print(self.params['N'] - D)
        mu = self.params['mu_base']

        current_params = self.params.copy()
        for event in self.events:
            if event.is_active(t):
                current_params.update(event.param_changes)
        beta, gamma, mu_factor = current_params['beta'], current_params['gamma'], current_params['mu_factor']

        mu_multiplier = max(0, min(1, sigmoid( (I / (self.hospital_capacity)) - 4)))
        # if(int(t) % 10 == 0 and mu_multiplier != 0):
        #     print(mu_multiplier)
        mu = mu_factor * mu_multiplier
        #mu += max(0, (I - self.hospital_capacity) / N * mu_factor)        
        dSdt = -beta * S * I / N + self.params['delta'] * R
        dEdt = beta * S * I / N - self.params['sigma'] * E
        dIdt = self.params['sigma'] * E - gamma * I - mu * I
        dRdt = gamma * I - self.params['delta'] * R
        dDdt = mu * I
        return dSdt, dEdt, dIdt, dRdt, dDdt

    def simulate(self, y0, t):
        results = odeint(self.deriv, y0, t)
        return t, results


hospital_capacity = 2000  # Hospital capacity for serious cases

events = [
    # Holidays
    Event(100, 120, {'beta': 10 / 14}),
    Event(300, 320, {'beta': 10 / 14}),
    Event(500, 520, {'beta': 10 / 14}),
    Event(700, 720, {'beta': 10 / 14}),
    

    Event(250, 500, {'gamma': 1 / 7})
]

class County:
    def __init__(self, y0, params, travel):
        self.y0 = list(y0)
        self.s, self.e, self.i, self.r, self.d = y0
        self.population = np.sum(self.y0) - self.y0[4]
        self.params = params
        self.travel = 0.01
        self.model = SEIRSModel(params, events, hospital_capacity)

    def update_status(self, t, adj_counties):
         # # Calculate infections from travel
        # travelers_infected = 0

        #if(int(t[0]) % 2 == 0):
        tot = 0
        for county in adj_counties:
            
            travelers = county.i * county.travel
            tot += travelers

        #if 0 < tot <= 100:                
        self.y0[1] += tot
        self.y0[0] -= tot

        # Solve for the next step
        t_end, result = self.model.simulate(self.y0, t)        
        self.y0 = result[-1]
        self.s, self.e, self.i, self.r, self.d = self.y0
        self.population = np.sum(self.y0) - self.y0[4]
        

       
        

        self.s, self.e, self.i, self.r, self.d = self.y0
        self.population = np.sum(self.y0) - self.y0[4]


        if self.population < 100:
            print("WARNING!! {}\n\n".format(self.population))

        return result



    def get_infected_ratio(self):
        if self.population > 0:
            return (self.i * 1) / self.population
        return 0

# Simulation parameters
N = 3  # Grid size
steps = 800

def random_initial_conditions(params):
    I0 = np.random.randint(0, 5)
    R0 = 0
    E0 = np.random.randint(0, 5)
    D0 = 0
    S0 = params['N'] - I0 - E0 - R0 - D0
    return [S0, E0, I0, R0, D0]

import math
def random_params(i,j):
    mid = int(N/2)
    dist = math.hypot(i-mid, j-mid)
    params = {
        'N': np.random.randint(1000, 10000),
        'beta': np.random.uniform(1, dist**0.5 + 1) / 14,
        'sigma': 1 / 5,     # Incubation Period
        'gamma': 1 / 14,    # Sick Period
        'delta': 1 / 120,   # Reinfection period
        'mu_base': 0.001 / 14,  # Base mortality rate per day                            NOTE: non zero mu breaks code!
        'mu_factor': 0.05  # Additional mortality factor when over capacity
    } 
    
    return params

random_params_starts = [[random_params(i,j) for i in range(N)] for j in range(N)]
random_initial_cond_starts = [[random_initial_conditions(random_params_starts[i][j]) for i in range(N)] for j in range(N)]

# Middle start
# random_initial_cond_starts[int(N/2)][int(N/2)][2] = 5
# random_initial_cond_starts[int(N/2)][int(N/2)][0] -= 5 

# Top middle
#random_initial_cond_starts[int(N/2)][0][2] = 5
#random_initial_cond_starts[int(N/2)][0][0] -= 5 

#print(random_params_starts)
# Initialize the grid
grid = [[County(random_initial_cond_starts[i][j], random_params_starts[i][j], 0) for i in range(N)] for j in range(N)]

# Track total SIR
total_susceptible = []
total_exposed = []
total_infected = []
total_recovered = []
total_deceased = []

INITIAL_POPULATION = sum(params['N'] for sublist in random_params_starts for params in sublist)


def update(frame_num, img, grid, N, ax2):
    new_grid = [[grid[i][j].get_infected_ratio() for j in range(N)] for i in range(N)]
    total_S, total_E, total_I, total_R, total_D = 0, 0, 0, 0, 0
    rows = INTERVAL_LENGTH+1
    cols = 5
    totals = [[0 for _ in range(cols)] for _ in range(rows)]
    # Get adjacent counties for each county and update statuses
    for i in range(N):
        for j in range(N):
            adj_counties = []
            #if(frame_num % 4 == 0):
            for di in (-1, 0, 1):
                for dj in (-1, 0, 1):
                    if di == 0 and dj == 0:
                        continue
                    ni, nj = (i + di) % N, (j + dj) % N
                    adj_counties.append(grid[ni][nj])

            timeframe = np.linspace(INTERVAL_LENGTH*frame_num, INTERVAL_LENGTH*frame_num+INTERVAL_LENGTH, INTERVAL_LENGTH+1)

            results = grid[i][j].update_status(timeframe, adj_counties)

            if i == j == 0:
                for l in range(len(results)):
                    for k in range(5):
                        totals[l][k] = results[l][k]
            else:
                for l in range(len(results)):
                    for k in range(5):
                        totals[l][k] += results[l][k]

            # total_S += grid[i][j].s
            # total_E += grid[i][j].e
            # total_I += grid[i][j].i
            # total_R += grid[i][j].r
            # total_D += grid[i][j].d
    for i in range(INTERVAL_LENGTH + 1):
        total_susceptible.append(totals[i][0])
        total_exposed.append(totals[i][1])
        total_infected.append(totals[i][2])
        total_recovered.append(totals[i][3])
        total_deceased.append(totals[i][4])

    # total_susceptible.append(total_S)
    # total_exposed.append(total_E)
    # total_infected.append(total_I)
    # total_recovered.append(total_R)
    # total_deceased.append(total_D)
    # Update the area chart
    ax2.clear()
    ax2.stackplot(range(len(total_susceptible)), total_susceptible, total_exposed, total_infected, total_recovered, total_deceased, colors=['b', 'y', 'r', 'g', 'k'], labels=['Susceptible', 'Exposed', 'Infectious', 'Recovered', 'Deceased'], alpha=0.5)
    ax2.legend(loc='upper right')
    ax2.set_xlim(0, steps)
    #ax2.set_xlim(0, framenum)

    ax2.set_ylim(0, INITIAL_POPULATION)
    ax2.set_title('Total SIR Dynamics Over Time')

    img.set_data(new_grid)
    return img,

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 8))
norm = Normalize(vmin=0, vmax=1)  # Normalize from 0 to 1
img = ax1.imshow([[grid[i][j].get_infected_ratio() for j in range(N)] for i in range(N)],
                 interpolation='nearest', cmap='viridis', norm=norm)  # Use fixed normalization
ani = animation.FuncAnimation(fig, update, fargs=(img, grid, N, ax2),
                              frames=steps, interval=1, save_count=10)

# for i in range(200):
#     update(i, img, grid, N, ax2)
#     print(i)

plt.tight_layout()
plt.show()
