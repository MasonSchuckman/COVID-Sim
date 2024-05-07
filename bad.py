import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation
from datetime import datetime, timedelta
from matplotlib.colors import Normalize
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
        # Create an expanded array of time points with sub-steps
        # finer_t = []
        # for i in range(len(t) - 1):
        #     start = t[i]
        #     end = t[i + 1]
        #     step = (end - start) / 10  # Divide each main interval into 10 smaller steps
        #     finer_t.extend(np.arange(start, end, step))
        # finer_t.append(t[-1])  # Make sure to include the last point

        # Solve the system at these finer time points
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
        
        
        # Solve for the next step
        t_end, result = self.model.simulate(self.y0, t)        
        self.y0 = result[-1]
        self.s, self.e, self.i, self.r, self.d = self.y0
        self.population = np.sum(self.y0) - self.y0[4]
        

        # # Calculate infections from travel
        # travelers_infected = 0

        if(int(t[0]) % 2 == 0):
            tot = 0
            for county in adj_counties:
                
                travelers = county.i * county.travel / 8
                tot += travelers

            #if 0 < tot <= 100:                
            self.y0[1] += tot
            self.y0[0] -= tot

            # elif tot < -0.01:
            #     print("bad travelers = {}".format(tot))
        
        self.s, self.e, self.i, self.r, self.d = self.y0
        self.population = np.sum(self.y0) - self.y0[4]

        if self.population < 100:
            print("WARNING!! {}\n\n".format(self.population))

    def get_infected_ratio(self):
        if self.population > 0:
            return self.i / self.population
        return 0

# Simulation parameters
N = 20  # Grid size
steps = 800

def random_initial_conditions(params):
    I0 = 0 #np.random.randint(0, 1)
    R0 = 0
    E0 = 0 #np.random.randint(0, 1)
    D0 = 0
    S0 = params['N'] - I0 - E0 - R0 - D0
    return [S0, E0, I0, R0, D0]

def random_params():
    params = {
        'N': np.random.randint(1000, 10000),
        'beta': np.random.uniform(0.5, 5.0) / 14,
        'sigma': 1 / 5,
        'gamma': 1 / 14,
        'delta': 1 / 365,
        'mu_base': 0.000 / 14,  # Base mortality rate per day
        'mu_factor': 0.00  # Additional mortality factor when over capacity
    }
    return params

random_params_starts = [[random_params() for _ in range(N)] for _ in range(N)]
random_initial_cond_starts = [[random_initial_conditions(random_params_starts[i][j]) for i in range(N)] for j in range(N)]
random_initial_cond_starts[int(N/2)][int(N/2)][2] = 5
random_initial_cond_starts[int(N/2)][int(N/2)][0] -= 5 

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
    
    # Get adjacent counties for each county and update statuses
    for i in range(N):
        for j in range(N):
            adj_counties = []
            for di in (-1, 0, 1):
                for dj in (-1, 0, 1):
                    if di == 0 and dj == 0:
                        continue
                    ni, nj = (i + di) % N, (j + dj) % N
                    adj_counties.append(grid[ni][nj])
            grid[i][j].update_status([frame_num, frame_num + 1], adj_counties)
            total_S += grid[i][j].s
            total_E += grid[i][j].e
            total_I += grid[i][j].i
            total_R += grid[i][j].r
            total_D += grid[i][j].d
    
    total_susceptible.append(total_S)
    total_exposed.append(total_E)
    total_infected.append(total_I)
    total_recovered.append(total_R)
    total_deceased.append(total_D)
    # Update the area chart
    ax2.clear()
    ax2.stackplot(range(len(total_susceptible)), total_susceptible, total_exposed, total_infected, total_recovered, total_deceased, colors=['b', 'y', 'r', 'g', 'k'], labels=['Susceptible', 'Exposed', 'Infectious', 'Recovered', 'Deceased'], alpha=0.5)
    ax2.legend(loc='upper right')
    ax2.set_xlim(0, steps)
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

plt.tight_layout()
plt.show()
