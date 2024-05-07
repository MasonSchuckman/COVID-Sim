import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.markers import MarkerStyle
from matplotlib.transforms import Affine2D
from matplotlib.animation import FuncAnimation
from matplotlib.patches import Circle


# Constants
population_size = 500   # Total population size
initial_infected = 3    # Initially infected people
R = 1                 # Infection radius
P = 0.5                 # Probability of infection (30%)

# Define parameters for social distancing
social_distancing_radius = 1.0  # Radius within which individuals try to maintain distance
repulsive_force_strength = 0.0  # Strength of the repulsive force


world_size = 20         # Size of the 2D world


# Grid size for spatial partitioning (each cell will be roughly the size of the distancing radius)
cell_size = social_distancing_radius * 2
grid_width = int(world_size / cell_size) + 1
grid_height = int(world_size / cell_size) + 1


steps = 1000             # Number of simulation steps

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


#plt.style.use('dark_background')
fig, ax = plt.subplots(figsize=(8, 8))

scat = ax.scatter([], [], s=20, edgecolors='none', facecolors='none')
title_text = ax.set_title("Initializing Simulation...", fontsize=15)

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


infection_events = {}
infectious_period_counter = {}

circle_lifetime = 2
infection_circles = [[] for _ in range(circle_lifetime)]

R_span = 15 # Average over 5 frames
prev_R = []

# Update function including social distancing and spatial partitioning
def update_seirs_social_distancing(frame):
    global positions, status, velocities, infection_circles
    

    for circle in infection_circles[frame % circle_lifetime]:
        circle.remove()
    infection_circles[frame % circle_lifetime] = []
    
    # Infection spread
    new_exposed = []
    new_infected = []
    new_recovered = []
    new_susceptible = []


    # Initialize grid
    grid = [[] for _ in range(grid_width * grid_height)]

    # Place individuals into grid
    for i in range(population_size):
        cell_x = int(positions[i][0] / cell_size)
        cell_y = int(positions[i][1] / cell_size)
        grid[cell_y * grid_width + cell_x].append(i)


    # Update positions with smoother movements and social distancing
    new_velocities = np.zeros_like(velocities)
    for i in range(population_size):
        cell_x = int(positions[i][0] / cell_size)
        cell_y = int(positions[i][1] / cell_size)
        
        # Social distancing forces
        nx, ny = cell_x, cell_y 
        if 0 <= nx < grid_width and 0 <= ny < grid_height:
            for j in grid[ny * grid_width + nx]:
                if i != j:
                    distance = np.linalg.norm(positions[i] - positions[j])
                    if distance < social_distancing_radius and distance > 0:
                        repulsion = (positions[i] - positions[j]) / distance * repulsive_force_strength
                        new_velocities[i] += repulsion


    # Apply velocities and limit them
    velocities += new_velocities + np.random.uniform(-0.02, 0.02, positions.shape)
    velocities = np.clip(velocities, -0.05, 0.05)
    positions += velocities
    #positions = np.clip(positions, 0, world_size)
    positions = np.mod(positions, world_size)
    #do the model
    for i in range(population_size):
        if status[i] == INFECTED:
            infectious_period_counter[i] = infectious_period_counter.get(i, 0) + 1
            if np.random.random() < 1/infection_period:
                new_recovered.append(i)
        elif status[i] == EXPOSED:
            if np.random.random() < 1/exposure_period:
                new_infected.append(i)
        elif status[i] == RECOVERED:
            if np.random.random() < 1/immunity_period:
                new_susceptible.append(i)

    for i in range(population_size):
        cell_x = int(positions[i][0] / cell_size)
        cell_y = int(positions[i][1] / cell_size)

        # Susceptibles becoming exposed
        if status[i] == INFECTED:
            for j in grid[cell_y * grid_width + cell_x]:
                if status[j] == SUSCEPTIBLE and j not in new_exposed:
                    distance = np.linalg.norm(positions[i] - positions[j])
                    if distance < R and np.random.random() < P:
                        if i not in infection_events.keys():
                            infection_events[i] = 1
                        else:
                            infection_events[i] = infection_events.get(i, 0) + 1
                        
                        new_exposed.append(j)
                        circle = Circle(positions[i], R, color='black', fill=False, linestyle='--')
                        ax.add_patch(circle)
                        
                        infection_circles[frame % circle_lifetime].append(circle)
                        

    

    # Update statuses
    for i in new_exposed:
        status[i] = EXPOSED
    for i in new_infected:
        status[i] = INFECTED
    for i in new_recovered:
        status[i] = RECOVERED
        del infectious_period_counter[i]  # Remove from counter once recovered
        #print(infection_events, " ", i)
        if(i in infection_events.keys()):
            del infection_events[i]
    for i in new_susceptible:
        status[i] = SUSCEPTIBLE

    #Calculate R value
    SUM = sum(infection_events.values())
    if len(infection_events) > 0 and SUM > 0:
        R_value = SUM / len(infection_events)
    else:
        R_value = 0

    AVG_R = R_value
    
    if(frame < R_span - 1):
        prev_R.append(R_value)
    else:
        prev_R[frame % R_span] = R_value
        AVG_R = sum(prev_R) / R_span
    
    

    title_text.set_text(f"Step {frame}: R value = {AVG_R:.2f}")


    # Update plot
    colors = ['blue' if s == SUSCEPTIBLE else 'orange' if s == EXPOSED else 'red' if s == INFECTED else 'green' for s in status]
    scat.set_offsets(positions)
    scat.set_color(colors)
    scat.set_facecolor(colors)
    scat.set_edgecolor(colors)
    

    return scat,


# Create animation with SEIRS model
ani_seirs = FuncAnimation(fig, update_seirs_social_distancing, frames=steps, interval=50, blit=False)

plt.show()
