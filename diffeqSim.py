import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# Parameters
beta = 2    # Transmission rate
sigma = 1/5.5 # Rate of becoming infectious
gamma = 1/2   # Recovery rate
alpha = 1/900  # Waning immunity rate
mu = 1/365    # Natural death rate
D_S = 1.0     # Diffusion coefficient for S
D_E = 0.08     # Diffusion coefficient for E
D_I = 0.05     # Diffusion coefficient for I
D_R = 0.03     # Diffusion coefficient for R

# Spatial domain
Lx, Ly = 5, 5        # Dimensions of the domain
dx = dy = 1.0           # Spatial step
Nx, Ny = int(Lx/dx), int(Ly/dy)  # Number of spatial points
x, y = np.linspace(0, Lx, Nx), np.linspace(0, Ly, Ny)
X, Y = np.meshgrid(x, y)

# Initial conditions
S = np.ones((Nx, Ny)) * 9900
E = np.zeros((Nx, Ny))
I = np.zeros((Nx, Ny))
I[Nx//2, Ny//2] = 100  # Initial infection in the center
R = np.zeros((Nx, Ny))
population = S + E + I + R

# Temporal domain
T = 200                # Total time
dt = .1               # Time step
Nt = int(T/dt)         # Number of time points

# Helper function for the diffusion term
def diffusion(u, D):
    return D * (np.roll(u, -1, axis=0) + np.roll(u, 1, axis=0) +
                np.roll(u, -1, axis=1) + np.roll(u, 1, axis=1) - 4 * u) / dx**2

# Simulation function
def update(frame):
    global S, E, I, R
    dSdt = diffusion(S, D_S) - beta * S * I / population + alpha * R - mu * S
    dEdt = diffusion(E, D_E) + beta * S * I / population - (sigma + mu) * E
    dIdt = diffusion(I, D_I) + sigma * E - (gamma + mu) * I
    dRdt = diffusion(R, D_R) + gamma * I - (alpha + mu) * R
    
    S += dSdt * dt
    E += dEdt * dt
    I += dIdt * dt
    R += dRdt * dt
    
    # Ensure all populations remain non-negative
    S[S < 0] = 0
    E[E < 0] = 0
    I[I < 0] = 0
    R[R < 0] = 0
    
    # Update the plots
    im.set_array(R)
    return im,

# Set up the figure and axis
fig, ax = plt.subplots(figsize=(6, 6))
im = ax.imshow(I, interpolation='bilinear', cmap='viridis',
               origin='lower', extent=[0, Lx, 0, Ly],
               vmax=100, vmin=0)
ax.set_title('Infectious Population Over Time')
ax.set_xlabel('X')
ax.set_ylabel('Y')
plt.colorbar(im, ax=ax)

# Create the animation
ani = FuncAnimation(fig, update, frames=Nt, repeat=False, interval=1)
plt.show()
