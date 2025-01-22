import numpy as np
import matplotlib.pyplot as plt
import random
import imageio
import os

def initialize_lattice(size):
    """Initialize a lattice with random values."""
    return np.random.rand(size, size)

def initialize_background(size):
    """Initialize a background grid with values set to 1 (good to move)."""
    return np.ones((size, size))

def update_background(background, x, y, decay_amount, neighborhood_radius):
    """Decrease the background values in the Moore neighborhood around occupied sites with periodic boundary conditions."""
    size = background.shape[0]
    for dx in range(-neighborhood_radius, neighborhood_radius + 1):
        for dy in range(-neighborhood_radius, neighborhood_radius + 1):
            nx, ny = (x + dx) % size, (y + dy) % size  # Wrap around edges
            background[nx, ny] = max(0, background[nx, ny] - decay_amount)

def get_neighbors(x, y, size):
    """Return the valid neighboring positions of a given site with periodic boundary conditions."""
    neighbors = [
        ((x - 1) % size, y),
        ((x + 1) % size, y),
        (x, (y - 1) % size),
        (x, (y + 1) % size)
    ]
    return neighbors

def calculate_bias(frontier, background, bias_factor):
    """Calculate selection probability based on background influence."""
    if bias_factor > 0:
        frontier = [((1 - bias_factor) * random_value - bias_factor * background[x, y], x, y) for random_value, x, y in frontier]
        frontier.sort()
        return frontier[0]  # Returns tuple (value, x, y)
    else:
        return frontier[0]  # Returns tuple (value, x, y)

def invasion_percolation(size=400, num_seeds=20, steps=5000, bias_factor=0.9, decay_amount=0.1, neighborhood_radius=3, make_gif=False, gif_name="percolation.gif"):
    """Perform invasion percolation with multiple seeds and periodic boundary conditions."""
    lattice = initialize_lattice(size)
    background = initialize_background(size)
    images = [] if make_gif else None

    # Initialize seeds randomly and update background accordingly
    occupied = set()
    frontier = []
    for _ in range(num_seeds):
        x, y = random.randint(0, size-1), random.randint(0, size-1)
        occupied.add((x, y))
        update_background(background, x, y, decay_amount, neighborhood_radius)
        for nx, ny in get_neighbors(x, y, size):
            frontier.append((lattice[nx, ny], nx, ny))
    
    frontier.sort()
    
    # Simulation steps
    for step in range(steps):
        if not frontier:
            break

        selected_site = calculate_bias(frontier, background, bias_factor)
        
        # Frontier update could be done more efficiently (works for now)
        val, x, y = selected_site
        frontier = [(v, fx, fy) for v, fx, fy in frontier if (fx, fy) != (x, y)]
        occupied.add((x, y))

        update_background(background, x, y, decay_amount, neighborhood_radius)
        
        # Update frontier list now that theres a new occupant in the lattice
        for nx, ny in get_neighbors(x, y, size):
            if (nx, ny) not in occupied:
                new_value = lattice[nx, ny]
                frontier.append((new_value, nx, ny))

        frontier.sort()

        # Optional, only to make a gif if needed
        if make_gif and step % 100 == 0:
            fig, axes = plt.subplots(1, 2, figsize=(12, 6))
            grid = np.zeros((size, size))
            for ox, oy in occupied:
                grid[ox, oy] = 1
            axes[0].imshow(grid, cmap='gray')
            axes[0].set_title("Invasion Percolation Model")
            
            cax = axes[1].imshow(background, cmap='hot', interpolation='nearest')
            axes[1].set_title("Background Intensity")
            fig.colorbar(cax, ax=axes[1], label="Background Intensity")
            
            plt.tight_layout()
            plt.savefig("frame.png")
            images.append(imageio.imread("frame.png"))
            plt.close()
    
    if make_gif and images:
        imageio.mimsave(gif_name, images, duration=0.1)
        os.remove("frame.png")
    
    return occupied, background

def plot_percolation(occupied, background, size):
    """Plot the percolated structure and background values."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    grid = np.zeros((size, size))
    for x, y in occupied:
        grid[x, y] = 1
    axes[0].imshow(grid, cmap='gray')
    axes[0].set_title("Invasion Percolation Model")
    
    cax = axes[1].imshow(background, cmap='hot', interpolation='nearest')
    axes[1].set_title("Background Intensity")
    fig.colorbar(cax, ax=axes[1], label="Background Intensity")
    plt.show()


# Just a function to visually compare some parameter settings quickly
def plot_percolation_variations(size, num_seeds, steps):
    """Plot percolation structures for different parameter values."""
    decay_values = [0.001, 0.01]
    bias_values = [0.1, 0.5, 0.9]
    neighborhood_radii = [5, 10, 20]
    make_gif = False
    fig, axes = plt.subplots(len(decay_values), len(bias_values) * len(neighborhood_radii), figsize=(15, 10))
    
    for i, decay_amount in enumerate(decay_values):
        for j, bias_factor in enumerate(bias_values):
            for k, neighborhood_radius in enumerate(neighborhood_radii):
                print(i,j,k)
                occupied, _ = invasion_percolation(size, num_seeds, steps, bias_factor, decay_amount, neighborhood_radius, make_gif, "percolation.gif")
                
                ax = axes[i, j * len(neighborhood_radii) + k]
                grid = np.zeros((size, size))
                for x, y in occupied:
                    grid[x, y] = 1
                ax.imshow(grid, cmap='gray')
                ax.set_title(f"D: {decay_amount}, B: {bias_factor}, R: {neighborhood_radius}")
                ax.axis('off')
    
    plt.tight_layout()
    plt.show()


# Run the percolation simulation
size = 400
num_seeds = 10
steps = 3000
bias_factor = 0.6
neighborhood_radius = 20
decay_amount = 0.01
make_gif = True  # Set to False to disable GIF creation

occupied_sites, background_grid = invasion_percolation(size, num_seeds, steps, bias_factor, decay_amount, neighborhood_radius, make_gif, "percolation.gif")
plot_percolation(occupied_sites, background_grid, size)

# For visually inspecting some varying parameter settings
size = 400
num_seeds = 10
steps = 3000
# plot_percolation_variations(size, num_seeds, steps)

