import numpy as np
import matplotlib.pyplot as plt
import random

def initialize_lattice(size):
    """Initialize a lattice with random values."""
    return np.random.rand(size, size)

def get_neighbors(x, y, size):
    """Return the valid neighboring positions of a given site."""
    neighbors = []
    if x > 0:
        neighbors.append((x-1, y))
    if x < size - 1:
        neighbors.append((x+1, y))
    if y > 0:
        neighbors.append((x, y-1))
    if y < size - 1:
        neighbors.append((x, y+1))
    return neighbors

def invasion_percolation(size=400, num_seeds=20, steps=5000, elongation=0.0):
    """Perform invasion percolation with multiple seeds and elongation factor."""
    lattice = initialize_lattice(size)
    
    # Initialize seeds randomly
    occupied = set()
    frontier = []
    for _ in range(num_seeds):
        x, y = random.randint(0, size-1), random.randint(0, size-1)
        occupied.add((x, y))
        for nx, ny in get_neighbors(x, y, size):
            frontier.append((lattice[nx, ny], nx, ny))
    
    frontier.sort()  # Maintain priority queue behavior
    
    for _ in range(steps):
        if not frontier:
            break
        
        # Select the lowest value site
        _, x, y = frontier.pop(0)
        occupied.add((x, y))
        
        # Add new frontier sites, modifying values near endpoints if elongation is applied
        for nx, ny in get_neighbors(x, y, size):
            if (nx, ny) not in occupied:
                new_value = lattice[nx, ny] * (1 - elongation) if any(n in occupied for n in get_neighbors(nx, ny, size)) else lattice[nx, ny]
                frontier.append((new_value, nx, ny))
        frontier.sort()
    
    return occupied

def plot_percolation(occupied, size):
    """Plot the percolated structure."""
    grid = np.zeros((size, size))
    for x, y in occupied:
        grid[x, y] = 1
    plt.imshow(grid, cmap='gray')
    plt.title("Invasion Percolation Model")
    plt.show()

# Run the invasion percolation simulation
size = 400
num_seeds = 30
steps = 5000
elongation = 0.2
occupied_sites = invasion_percolation(size, num_seeds, steps, elongation)
plot_percolation(occupied_sites, size)
