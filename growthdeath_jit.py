"""
This file simulates the CA model with growth and death of tumor cells based on the number of 
blood vessels surrounding it.
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import random
from numba import njit
import time

def initialize_seeds(size, seeds_per_edge=5):
    """
    Generates a list of seeds evenly distributed along the four edges.
    Order is left, right, top, bottom.
    Input:
    - size: The size/dimension of the grid
    - seeds_per_edge: The number of seeds to generate along each edge
    Output:
    - A list of seed coordinates
    """
    seeds = []
    
    seeds.extend([(0, random.randint(0, size - 1)) for _ in range(seeds_per_edge)])
    seeds.extend([(size - 1, random.randint(0, size - 1)) for _ in range(seeds_per_edge)])
    seeds.extend([(random.randint(0, size - 1), size - 1) for _ in range(seeds_per_edge)])
    seeds.extend([(random.randint(0, size - 1), 0) for _ in range(seeds_per_edge)])
    
    return seeds

def initialize_lattice(size):
    """
    Initialize a lattice with random values. Separate grid eases the analysis later.
    Input:
    - size: The size/dimension of the grid
    Output:
    - A grid initialized with random values for stochastic choice.
    """
    return np.random.rand(size, size)

@njit
def initialize_background(size):
    """
    Initialize a background grid with VEGF concentrated at the center.
    Input:
    - size: The size/dimension of the grid
    Output:
    - A grid initialized with Gaussian decay of VEGF centered at the middle.
    """
    background = np.zeros((size, size))
    center = size // 2
    sigma = size / 3
    empty_radius = size / 10
    for x in range(size):
        for y in range(size):
            distance = np.sqrt((x - center) ** 2 + (y - center) ** 2)
            if distance > empty_radius:
                background[x, y] = 0.9 * np.exp(-distance**2 / (2 * sigma**2))
    return background

@njit
def create_tumor(size, background, tumor_prob, tumor_factor):
    """
    Creates the list of coordinates for the tumor.
    Input:
    - size: The size/dimension of the grid
    - background: The grid with VEGF values
    - tumor_prob: The probability of a cell becoming a tumor cell
    - tumor_factor: The factor by which the VEGF value of a tumor cell is multiplied
    Output:
    - A set of coordinates for the tumor cells
    """
    tumor = np.zeros((size, size), dtype=np.bool_)
    center = size // 2
    empty_radius = size / 10
    tumor_radius = size / 5
    for x in range(size):
        for y in range(size):
            distance = np.sqrt((x - center) ** 2 + (y - center) ** 2)
            if tumor_radius > distance > empty_radius and random.random() > tumor_prob:
                tumor[x, y] = True
                background[x, y] += tumor_factor
    return tumor

@njit
def check_blood(x, y, occupied, radius):
    """Check the number of blood vessels surrounding a tumor cell.
    Input:
    - x, y: The coordinates of the tumor cell
    - occupied: The set of occupied sites
    - radius: The radius of the neighborhood to check
    Output:
    - A list of coordinates of blood vessels surrounding the tumor cell
    """
    blood = []
    for dx in range(-radius, radius + 1):
        for dy in range(-radius, radius + 1):
            if abs(dx) + abs(dy) <= radius and occupied[x, y]:
                nx, ny = x + dx, y + dy
                blood.append((nx, ny))
    return blood

@njit
def growth_death(background, size, tumor, tumor_factor, radius, occupied, p):
    """Determines growth/death of tumor cells based on how many blood vessels cells surround it.
    Input:
    - background: The grid with VEGF values
    - size: The size/dimension of the grid
    - tumor: The set of coordinates for the tumor cells
    - tumor_factor: The factor by which the VEGF value of a tumor cell is multiplied
    - radius: The radius of the neighborhood to check
    - occupied: The set of occupied sites with blood vessel cells
    - p: The probability of growth/death
    Output:
    No output, but updates the tumor set and the background grid.
    """
    center = size // 2
    empty_radius = size / 10
    tumor_radius = size / 5
    total_neighborhood = 2 * radius * (radius + 1)
    for x in range(size):
        for y in range(size):
            distance = np.sqrt((x - center) ** 2 + (y - center) ** 2)
            if tumor_radius > distance > empty_radius and tumor[x, y]:
                blood = check_blood(x, y, occupied, radius)
                blood_bias = 1 / (1 + np.exp(-1 * (len(blood)-1)))
                death = p * (1 - blood_bias)
                growth = p * (blood_bias)
                if random.random() <= growth:
                    neighbors = get_neighbors(x, y, size, wrap_around=False)
                    for nx, ny in neighbors:
                        if not occupied[nx, ny] and not tumor[nx, ny]:
                            tumor[nx, ny] = True
                            background[nx, ny] += tumor_factor
                elif random.random() <= death:
                    tumor[x, y] = False
                    background[x, y] -= tumor_factor

@njit
def get_neighbors(x, y, size, wrap_around=True):
    """
    Return neighboring positions with periodic boundaries if enabled.
    Input:
    - x, y: The coordinates of the current site
    - size: The size/dimension of the grid
    - wrap_around: A boolean to enable periodic boundaries
    Output:
    - A list of neighboring coordinates
    """
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

@njit
def update_background(background, x, y, decay_factor, neighborhood_radius, wrap_around=True):
    """
    Decrease the background values using a Gaussian weight in a circular neighborhood 
    with maximal decay at the center.
    Input:
    - background: The grid with VEGF values
    - x, y: The coordinates of the current site
    - decay_factor: The factor by which the VEGF value is multiplied
    - neighborhood_radius: The radius of the neighborhood to consider
    - wrap_around: A boolean to enable periodic boundaries
    Output:
    No output, but updates the background grid.
    """
    size = background.shape[0]
    sigma = neighborhood_radius / 2
    for dx in range(-neighborhood_radius, neighborhood_radius + 1):
        for dy in range(-neighborhood_radius, neighborhood_radius + 1):
            distance = np.sqrt(dx**2 + dy**2)
            if distance <= neighborhood_radius:
                weight = np.exp(-distance**2 / (2 * sigma**2))
                nx, ny = x + dx, y + dy
                if wrap_around:
                    nx %= size
                    ny %= size
                if 0 <= nx < size and 0 <= ny < size:
                    background[nx, ny] = max(0, background[nx, ny] * (1 - weight * (1 - decay_factor)))

@njit
def move_seed(x, y, background, size, wrap_around, bias_factor):
    """
    Move seed based on stochastic angiogenesis rules.
    Input:
    - x, y: The coordinates of the current site
    - background: The grid with VEGF values
    - size: The size/dimension of the grid
    - wrap_around: A boolean to enable periodic boundaries
    - bias_factor: The factor by which the VEGF value is multiplied
    Output:
    - The new coordinates for the seed
    """
    neighbors = get_neighbors(x, y, size, wrap_around)
    move_probabilities = [(random.random() * (1 - bias_factor) + background[nx, ny] * bias_factor, nx, ny) for nx, ny in neighbors]
    move_probabilities.sort(reverse=True) # Favor higher VEGF concentration with stochasticity
    return move_probabilities[0][1], move_probabilities[0][2]

@njit
def shannon_entropy(grid):
    """
    Compute Shannon entropy for a grid considering 3 states and limiting the region to tumor radius.
    Input:
    - grid: The grid with tumor cells and blood vessels
    Output:
    - The Shannon entropy value
    """    
    size = grid.shape[0]
    center = size // 2
    empty_radius = size / 10
    tumor_radius = size / 6

    flattened = []
    for x in range(size):
        for y in range(size):
            distance = np.sqrt((x - center)**2 + (y - center)**2)
            if tumor_radius > distance > empty_radius:
                flattened.append(grid[x, y])

    flattened = np.array(flattened)
    total_cells = len(flattened)
    if total_cells == 0:
        return 0

    unique_values = []
    counts = []
    for value in flattened:
        if value not in unique_values:
            unique_values.append(value)
            counts.append(1)
        else:
            counts[unique_values.index(value)] += 1

    probabilities = np.array(counts) / total_cells
    entropy = -np.sum(probabilities * np.log2(probabilities))
    return entropy

def simulate_CA(size=200, num_seeds=20, steps=500, bias_factor=0.93, decay_factor=0.99, neighborhood_radius=10, wrap_around=False):
    """
    Run a cellular automata-based angiogenesis model and compute Shannon entropy.
    Input:
    - size: The size/dimension of the grid
    - num_seeds: The number of seeds to initialize
    - steps: The number of time steps to simulate
    - bias_factor: The factor by which the VEGF value is multiplied
    - decay_factor: The factor by which the VEGF value is multiplied
    - neighborhood_radius: The radius of the neighborhood to consider
    - wrap_around: A boolean to enable periodic boundaries
    Output:
    - The Shannon entropy value in the last time step
    """
    background = initialize_background(size)
    occupied = np.zeros((size, size), dtype=np.bool_)
    tumor_factor = 0.1
    tumor_prob = 0.5
    
    # Initialize seeds for blood vessels at random positions
    seeds = initialize_seeds(size)
    for x, y in seeds:
        occupied[x, y] = True
        update_background(background, x, y, decay_factor, neighborhood_radius, wrap_around=False)
    tumor = create_tumor(size, background, tumor_prob, tumor_factor)
    entropies = []
    for i in range(steps):
        new_seeds = []
        for x, y in seeds:
            nx, ny = move_seed(x, y, background, size, wrap_around, bias_factor)
            new_seeds.append((nx, ny))
            occupied[nx, ny] = True
            update_background(background, x, y, decay_factor, neighborhood_radius, wrap_around=False)
        seeds = new_seeds
        
        # State 2 represents tumor cells, state 1 represents blood vessels
        grid = np.zeros((size, size))
        for x in range(size):
            for y in range(size):
                if tumor[x, y]:
                    grid[x, y] = 2
                elif occupied[x, y]:
                    grid[x, y] = 1
                    
        # Introduce growth and death of tumor cells after a certain time step
        if i > 400:
            growth_death(background, size, tumor, tumor_factor, 2, occupied, p=0.01)
            
        # Currently, we are only calculating the entropy for the tumor region
        tumor_grid = np.zeros((size, size))
        for x in range(size):
            for y in range(size):
                if tumor[x, y]:
                    tumor_grid[x, y] = 2
        entropy = shannon_entropy(tumor_grid)
        entropies.append(entropy)
    
    # Plotting the visualization and tumor entropy over time
    plt.figure(figsize=(10, 5))
    cmap = ListedColormap(["white", "red", "green"])
    print(f"Number of blood vessel pixels: {np.sum(occupied)}")
    print(f"Number of tumor pixels: {np.sum(tumor)}")
    plt.subplot(1, 2, 1)
    plt.imshow(grid, cmap=cmap)
    plt.title("Angiogenesis-Based CA Growth with Stochasticity")
    plt.subplot(1, 2, 2)
    plt.plot(entropies, label="Shannon Entropy")
    plt.title("Tumor Shannon Entropy Over Time")
    plt.xlabel("Time Step")
    plt.ylabel("Entropy")
    plt.legend()
    plt.tight_layout()
    plt.show()
    
    return grid, entropies[-1]

def vessel_image(grid, filename):
    """
    Create a vessel image from the grid.
    Input:
    - grid: The grid with blood vessels and tumor cells
    Output:
    - The vessel image
    """
    image = np.zeros_like(grid, dtype=np.uint8)
    image[grid == 1] = 255  # Set blood vessel cells to white

    # Save the black-and-white image
    bw_fig, bw_ax = plt.subplots(figsize=(6, 6))
    bw_ax.imshow(image, cmap='gray', interpolation='nearest')
    bw_ax.axis('off')
    plt.subplots_adjust(left=0, right=1, top=1, bottom=0)  # Remove padding
    bw_fig.savefig(f'images/{filename}', dpi=300, bbox_inches='tight', pad_inches=0)
    plt.close(bw_fig)

def main():
    """
    Main function to execute the simulation.
    """
    size = 200
    num_seeds = 20
    steps = 500
    bias_factor = 0.93
    decay_factor = 0.99
    neighborhood_radius = 10
    wrap_around = False

    grid, final_entropy = simulate_CA(
        size=size,
        num_seeds=num_seeds,
        steps=steps,
        bias_factor=bias_factor,
        decay_factor=decay_factor,
        neighborhood_radius=neighborhood_radius,
        wrap_around=wrap_around,
    )
    vessel_image(grid, 'final_grid.png')
    
if __name__ == "__main__":
    start_time = time.time()
    main()
    elapsed_time = time.time() - start_time
    print(f"Total execution time: {elapsed_time:.6f} seconds.")