import numpy as np
import matplotlib.pyplot as plt

# Constants
population_size = 200   # Total population size
initial_infected = 3    # Initially infected people
R = 0.5                 # Infection radius
P = 0.3                 # Probability of infection (30%)
world_size = 10         # Size of the 2D world
steps = 100             # Number of simulation steps

# Status codes
SUSCEPTIBLE = 0
INFECTED = 1

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

# Simulation function
def simulate(positions, status):
    for step in range(steps):
        # Random walk
        positions += np.random.uniform(-0.1, 0.1, positions.shape)
        # Ensure positions stay within bounds
        positions = np.clip(positions, 0, world_size)

        # Check for new infections
        for i in range(population_size):
            if status[i] == INFECTED:
                for j in range(population_size):
                    if i != j and status[j] == SUSCEPTIBLE:
                        distance = np.linalg.norm(positions[i] - positions[j])
                        if distance < R and np.random.random() < P:
                            status[j] = INFECTED

        if step % 10 == 0:
            plot_population(positions, status, step)

# Run simulation
#simulate(positions, status)


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

# Update function for the smoother animation
def update(frame):
    global positions, status, velocities
    # Update positions with a smoother step, influenced by inertia
    velocities += np.random.uniform(-0.02, 0.02, positions.shape)
    velocities = np.clip(velocities, -0.05, 0.05)  # Limit max velocity
    positions += velocities
    positions = np.clip(positions, 0, world_size)
    
    # Infection spread
    for i in range(population_size):
        if status[i] == INFECTED:
            for j in range(population_size):
                if i != j and status[j] == SUSCEPTIBLE:
                    distance = np.linalg.norm(positions[i] - positions[j])
                    if distance < R and np.random.random() < P:
                        status[j] = INFECTED

    colors = ['red' if s == INFECTED else 'blue' for s in status]
    scat.set_offsets(positions)
    scat.set_color(colors)
    return scat,

# Create animation with smoother movement
ani = FuncAnimation(fig, update, frames=steps, interval=10, blit=True)
plt.show()
# Save or display the animation
# plt.close(fig)  # Avoid static plot output
# ani.save('/mnt/data/SIR_simulation_smoothed.mp4', writer='ffmpeg', dpi=80)