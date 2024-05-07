import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from matplotlib.animation import FuncAnimation

# Constants
population_size = 800   # Total population size
initial_infected = 3    # Initially infected people
R = 1                   # Infection radius
P = 0.5                 # Probability of infection

world_size = 20         # Size of the 2D world
steps = 1000            # Number of simulation steps

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
velocities = np.random.uniform(-0.05, 0.05, (population_size, 2))  # Initial velocities
status = np.array([SUSCEPTIBLE] * population_size)
initially_infected_indices = np.random.choice(population_size, initial_infected, replace=False)
status[initially_infected_indices] = INFECTED

fig, ax = plt.subplots(figsize=(8, 8))
scat = ax.scatter([], [], s=20)
title_text = ax.set_title("Initializing Simulation...", fontsize=15)

# Set plot limits and labels
ax.set_xlim(0, world_size)
ax.set_ylim(0, world_size)
ax.grid(True)

infection_circles = []

def update(frame):
    global positions, velocities, status, infection_circles

    # Move individuals
    positions += velocities
    positions = np.clip(positions, 0, world_size)

    # Update velocities slightly for random movement
    velocities += np.random.uniform(-0.01, 0.01, velocities.shape)
    velocities = np.clip(velocities, -0.05, 0.05)

    new_infections = []
    # Check for infections
    for i in range(population_size):
        if status[i] == INFECTED:
            for j in range(population_size):
                if i != j and status[j] == SUSCEPTIBLE:
                    distance = np.linalg.norm(positions[i] - positions[j])
                    if distance <= R and np.random.rand() < P:
                        new_infections.append(j)
                        circle = Circle(positions[i], 2, color='yellow', fill=False, linestyle='--')
                        ax.add_patch(circle)
                        infection_circles.append(circle)

    # Update statuses
    for i in new_infections:
        status[i] = INFECTED

    # Remove old circles
    if frame % 10 == 0:
        for circle in infection_circles:
            circle.remove()
        infection_circles = []

    colors = ['blue' if x == SUSCEPTIBLE else 'red' for x in status]
    scat.set_offsets(positions)
    scat.set_color(colors)

    title_text.set_text(f"Step {frame}")

    return scat,

ani = FuncAnimation(fig, update, frames=steps, interval=50, blit=False)

plt.show()