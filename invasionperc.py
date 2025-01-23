import numpy as np
import matplotlib.pyplot as plt
import random
import imageio
import os

def initialize_lattice(size):
    """Initialize a lattice with random values."""
    return np.random.rand(size, size)

def initialize_background_old(size):
    """Initialize a background grid with values set to 1 (good to move)."""
    return np.ones((size, size))

def initialize_background(size):
    """Initialize a background grid with Gaussian decay centered at the middle, leaving an empty center."""
    background = np.zeros((size, size))
    center = size // 2
    sigma = size / 3  # Controls the spread of the Gaussian decay
    empty_radius = size / 10  # Defines the empty center region
    for x in range(size):
        for y in range(size):
            distance = np.sqrt((x - center) ** 2 + (y - center) ** 2)
            if distance > empty_radius:
                background[x, y] = np.exp(-distance**2 / (2 * sigma**2))
    return background

# This function is not used right now, but could be another implementation of background decrease
# This might be better since it uses circular decrease (instead of von neumann, which still causes effects of being too "square")
def update_background(background, x, y, decay_amount, neighborhood_radius, wrap_around=True):
    """Decrease the background values using a Gaussian weight in a circular neighborhood with maximal decay at the center."""
    size = background.shape[0]
    sigma = neighborhood_radius / 2  # Controls smoothness
    for dx in range(-neighborhood_radius, neighborhood_radius + 1):
        for dy in range(-neighborhood_radius, neighborhood_radius + 1):
            distance = np.sqrt(dx**2 + dy**2)
            if distance <= neighborhood_radius:
                weight = np.exp(-distance**2 / (2 * sigma**2))  # Gaussian function
                nx, ny = x + dx, y + dy
                if wrap_around:
                    nx %= size
                    ny %= size
                if 0 <= nx < size and 0 <= ny < size:
                    background[nx, ny] = max(0, background[nx, ny] * (1 - weight * (1 - decay_amount)))

def update_background_neumann(background, x, y, decay_amount, neighborhood_radius, wrap_around=True):
    """Decrease the background values in the Von Neumann neighborhood around occupied sites with optional periodic boundary conditions."""
    size = background.shape[0]
    for dx in range(-neighborhood_radius, neighborhood_radius + 1):
        for dy in range(-neighborhood_radius, neighborhood_radius + 1):
            if abs(dx) + abs(dy) <= neighborhood_radius:  # Von Neumann neighborhood condition
                nx, ny = x + dx, y + dy
                if wrap_around:
                    nx %= size
                    ny %= size
                if 0 <= nx < size and 0 <= ny < size:
                    background[nx, ny] = max(0, background[nx, ny] * decay_amount)

def get_neighbors(x, y, size, wrap_around=True):
    """Return the valid neighboring positions of a given site with optional periodic boundary conditions."""
    neighbors = []
    shifts = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    for dx, dy in shifts:
        nx, ny = x + dx, y + dy
        if wrap_around:
            nx %= size
            ny %= size
        if 0 <= nx < size and 0 <= ny < size:
            neighbors.append((nx, ny))
    return neighbors

def calculate_bias(frontier, background, bias_factor):
    """Calculate selection probability based on background influence."""
    if bias_factor > 0:
        frontier = [((1 - bias_factor) * random_value - bias_factor * background[x, y], x, y) for random_value, x, y in frontier]
        frontier.sort()
        return frontier[0]  # Returns tuple (value, x, y)
    else:
        return frontier[0]  # Returns tuple (value, x, y)

def invasion_percolation(size=400, num_seeds=20, steps=5000, bias_factor=0.9, decay_amount=0.1, neighborhood_radius=3, make_gif=False, gif_name="percolation.gif", wrap_around=False):
    """Perform invasion percolation with multiple seeds and optional periodic boundary conditions."""
    lattice = initialize_lattice(size)
    background = initialize_background(size)
    images = [] if make_gif else None

    # Initialize seeds randomly and update background accordingly
    occupied = set()
    frontier = []
    for _ in range(num_seeds):
        x, y = random.randint(0, size-1), random.randint(0, size-1)
        occupied.add((x, y))
        update_background(background, x, y, decay_amount, neighborhood_radius, wrap_around)
        for nx, ny in get_neighbors(x, y, size, wrap_around):
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

        update_background(background, x, y, decay_amount, neighborhood_radius, wrap_around)
        
        # Update frontier list now that there's a new occupant in the lattice
        for nx, ny in get_neighbors(x, y, size, wrap_around):
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


def plot_percolation_variations(size, num_seeds, steps):
    """Plot percolation structures for different parameter values."""
    
    bias_values = [0.1, 0.5, 0.9]
    decay_values = [0.999, 0.99]
    neighborhood_radii = [5, 10, 20]
    make_gif = False
    fig, axes = plt.subplots(len(bias_values), len(decay_values) * len(neighborhood_radii), figsize=(15, 15))
    
    for i, bias_factor in enumerate(bias_values):
        for j, decay_amount in enumerate(decay_values):
            for k, neighborhood_radius in enumerate(neighborhood_radii):
                print(i,j,k)
                occupied, _ = invasion_percolation(size, num_seeds, steps, bias_factor, decay_amount, neighborhood_radius, make_gif, "percolation.gif")
                
                ax = axes[i % 3, j * len(neighborhood_radii) + k]
                grid = np.zeros((size, size))
                for x, y in occupied:
                    grid[x, y] = 1
                ax.imshow(grid, cmap='gray')
                ax.set_title(f"b: {bias_factor}, d: {decay_amount}, r: {neighborhood_radius}")
                ax.axis('off')
    
    plt.tight_layout()
    plt.show()

# Run the percolation simulation
size = 200
num_seeds = 10
steps = 2000
bias_factor = 0.9
neighborhood_radius = 10
decay_amount = 0.99 # this is a decay factor for the background update, based on the walk of the cells
make_gif = True 

occupied_sites, background_grid = invasion_percolation(size, num_seeds, steps, bias_factor, decay_amount, neighborhood_radius, make_gif, "percolation.gif")
plot_percolation(occupied_sites, background_grid, size)

# For visually inspecting some varying parameter settings
size = 400
num_seeds = 10
steps = 3000
#plot_percolation_variations(size, num_seeds, steps)

