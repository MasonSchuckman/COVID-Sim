import numpy as np
import matplotlib.pyplot as plt

# Constants
population_size = 8000   # Total population size
initial_infected = 3    # Initially infected people
R = .5                 # Infection radius
P = 0.2                 # Probability of infection (30%)
world_size = 20         # Size of the 2D world
steps = 100             # Number of simulation steps

# Status codes
SUSCEPTIBLE = 0
INFECTED = 1
EXPOSED = 2
RECOVERED = 3

# Constants for the SEIRS model to simulate COVID-19-like behavior
exposure_period = 5     # Days until exposed individuals become infectious
infection_period = 10   # Days infectious individuals remain so before recovery
immunity_period = 90    # Days recovered individuals remain immune before possibly becoming susceptible again



# Initialize the positions randomly
positions = np.random.uniform(0, world_size, (population_size, 2))

# Initialize the status (all susceptible except initial infected)
status = np.array([SUSCEPTIBLE] * population_size)
initially_infected_indices = np.random.choice(population_size, initial_infected, replace=False)
status[initially_infected_indices] = INFECTED

# Plotting function
def plot_population(positions, status, step):
    plt.figure(figsize=(8, 8))
    plt.title(f"Step {step}")
    for i in range(population_size):
        if status[i] == INFECTED:
            plt.scatter(positions[i, 0], positions[i, 1], color='red')
        else:
            plt.scatter(positions[i, 0], positions[i, 1], color='blue')
    plt.xlim(0, world_size)
    plt.ylim(0, world_size)
    plt.grid(True)
    plt.show()



from matplotlib.animation import FuncAnimation

# Create figure for the smoother movement plot
fig, ax = plt.subplots(figsize=(8, 8))
scat = ax.scatter([], [], c=[])

# Set plot limits and labels
ax.set_xlim(0, world_size)
ax.set_ylim(0, world_size)
ax.grid(True)

# Initialize positions and status again
positions = np.random.uniform(0, world_size, (population_size, 2))
velocities = np.random.uniform(-0.05, 0.05, (population_size, 2))  # Initial velocities
status = np.array([SUSCEPTIBLE] * population_size)
initially_infected_indices = np.random.choice(population_size, initial_infected, replace=False)
status[initially_infected_indices] = INFECTED


# Update function for the SEIRS animation
def update_seirs(frame):
    global positions, status, velocities
    # Update positions with smoother movements
    velocities += np.random.uniform(-0.02, 0.02, positions.shape)
    velocities = np.clip(velocities, -0.05, 0.05)  # Limit max velocity
    positions += velocities
    positions = np.clip(positions, 0, world_size)
    
    # Infection spread
    new_exposed = []
    new_infected = []
    new_recovered = []
    new_susceptible = []
    
    rows = 20
    cols = 20
    matrix = [[[0] for _ in range(cols)] for _ in range(rows)]

    for i in range(population_size):
        x,y = np.clip(positions[i], 0, rows - 1)
        x = int(x)
        y = int(y)
        
        matrix[y][x].append(i)

        if status[i] == INFECTED:
            if np.random.random() < 1/infection_period:
                new_recovered.append(i)
        elif status[i] == EXPOSED:
            if np.random.random() < 1/exposure_period:
                new_infected.append(i)
        elif status[i] == RECOVERED:
            if np.random.random() < 1/immunity_period:
                new_susceptible.append(i)

    for i in range(population_size):
        x,y = np.clip(positions[i], 0, rows - 1)
        x = int(x)
        y = int(y)

        # Susceptibles becoming exposed
        if status[i] == SUSCEPTIBLE:
            for j in matrix[y][x]:
                if status[j] == INFECTED:
                    distance = np.linalg.norm(positions[i] - positions[j])
                    if distance < R and np.random.random() < P:
                        new_exposed.append(i)
                        break

    # Update statuses
    for i in new_exposed:
        status[i] = EXPOSED
    for i in new_infected:
        status[i] = INFECTED
    for i in new_recovered:
        status[i] = RECOVERED
    for i in new_susceptible:
        status[i] = SUSCEPTIBLE

    # Update plot
    colors = ['blue' if s == SUSCEPTIBLE else 'orange' if s == EXPOSED else 'red' if s == INFECTED else 'green' for s in status]
    scat.set_offsets(positions)
    scat.set_color(colors)
    return scat,

# Create animation with SEIRS model
ani_seirs = FuncAnimation(fig, update_seirs, frames=steps, interval=10, blit=True)
plt.show()
