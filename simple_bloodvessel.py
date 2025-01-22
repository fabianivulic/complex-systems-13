import numpy as np
import matplotlib.pyplot as plt
import random

# Parameters
grid_size = 200  # Size of the 2D lattice
num_walkers = 5000  # Number of endothelial cells
max_steps = 1000  # Maximum steps for each walker

# Initialize grid
vessel_grid = np.zeros((grid_size, grid_size), dtype=int)

# Set initial seed (existing vessel segment)
vessel_grid[grid_size // 2, grid_size // 2] = 1

# Function to perform random walk
def random_walk():
    x, y = random.randint(0, grid_size-1), random.randint(0, grid_size-1)
    
    for _ in range(max_steps):
        # Random movement (isotropic diffusion)
        dx, dy = random.choice([(1,0), (-1,0), (0,1), (0,-1)])
        x, y = min(max(x + dx, 0), grid_size-1), min(max(y + dy, 0), grid_size-1)
        
        # Check if the walker is near a vessel segment
        if (vessel_grid[max(x-1,0):min(x+2,grid_size), max(y-1,0):min(y+2,grid_size)] > 0).any():
            vessel_grid[x, y] = 1  # Attach to the vessel network
            return

# Perform DLA Growth
for _ in range(num_walkers):
    random_walk()

# Visualization
plt.figure(figsize=(8, 8))
plt.imshow(vessel_grid, cmap='gray')
plt.title('DLA-Based Angiogenesis Simulation')
plt.axis('off')
plt.show()
