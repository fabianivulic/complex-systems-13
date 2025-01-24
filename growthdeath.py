import numpy as np
import matplotlib.pyplot as plt
import random
import imageio
import os

def initialize_lattice(size):
    """Initialize a lattice with random values."""
    return np.random.rand(size, size)

def initialize_background(size):
    """Initialize a background grid with Gaussian decay centered at the middle."""
    background = np.zeros((size, size))
    center = size // 2
    sigma = size / 3  # Controls the spread of the Gaussian decay
    empty_radius = size / 10  # Defines the empty center region
    for x in range(size):
        for y in range(size):
            distance = np.sqrt((x - center) ** 2 + (y - center) ** 2)
            if distance > empty_radius:
                background[x, y] = 0.9 * np.exp(-distance**2 / (2 * sigma**2))
    return background

def create_tumor(size, background, tumor_prob, tumor_factor):
    """Creates the list of coordinates for the tumor."""
    tumor = set()
    center = size // 2
    empty_radius = size / 10
    tumor_radius = size / 6

    for x in range(size):
        for y in range(size):
            distance = np.sqrt((x - center) ** 2 + (y - center) ** 2)
            if tumor_radius > distance > empty_radius and random.random() > tumor_prob:
                tumor.add((x, y))
                background[x, y] += tumor_factor

    return tumor

def check_blood(x, y, occupied, radius):
    blood = []

    for dx in range(-radius, radius + 1):
        for dy in range(-radius, radius + 1):
            if abs(dx) + abs(dy) <= radius and (x, y) in occupied:
                nx, ny = x + dx, y + dy
                blood.append((nx, ny))
    
    return blood

def growth_death(background, size, tumor, tumor_factor, radius, occupied, p):
    """Determines growth/death of tumor cells based on how many blood vessels cells surround it."""
    center = size // 2
    empty_radius = size / 10
    tumor_radius = size / 6
    total_neighborhood = 2 * radius * (radius + 1)

    for x in range(size):
        for y in range(size):
            distance = np.sqrt((x - center) ** 2 + (y - center) ** 2)
            if tumor_radius > distance > empty_radius and (x, y) in tumor:
                blood = check_blood(x, y, occupied, radius)
                blood_bias = 1 / (1 + np.exp(-2 * (len(blood)-2)))
                death = p * (1 - blood_bias)
                growth = p * (blood_bias)
                if random.random() <= growth:
                    print("Growth")
                    neighbors = get_neighbors(x, y, size, wrap_around=False)
                    for x, y in neighbors:
                        if (x, y) not in occupied and (x, y) not in tumor:
                            occupied.add((x, y))
                            background[x, y] += tumor_factor
                elif random.random() <= death:
                    print("Death")
                    tumor.remove((x, y))
                else:
                    continue

def get_neighbors(x, y, size, wrap_around=True):
    """Return neighboring positions with periodic boundaries if enabled."""
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

def update_background(background, x, y, decay_factor, neighborhood_radius, wrap_around=True):
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
                    background[nx, ny] = max(0, background[nx, ny] * (1 - weight * (1 - decay_factor)))

def move_seed(x, y, background, size, wrap_around, bias_factor):
    """Move seed based on stochastic angiogenesis rules."""
    neighbors = get_neighbors(x, y, size, wrap_around)
    move_probabilities = [(random.random() * (1 - bias_factor) + background[nx, ny] * bias_factor, nx, ny) for nx, ny in neighbors]
    move_probabilities.sort(reverse=True)  # Favor higher VEGF concentration with stochasticity
    return move_probabilities[0][1], move_probabilities[0][2]

def shannon_entropy(grid):
    """Compute Shannon entropy for a grid considering 3 states and limiting the region to tumor radius."""    
    size = grid.shape[0]
    center = size // 2
    empty_radius = size / 10
    tumor_radius = size / 6

    flattened = []

    for x in range(size):
        for y in range(size):
            distance = np.sqrt((x - center)**2 + (y - center)**2)
            if tumor_radius > distance > empty_radius :
                flattened.append(grid[x, y])
    
    # Compute probability of each state
    flattened = np.array(flattened)
    total_cells = len(flattened)
    if total_cells == 0:
        return 0

    unique_states, counts = np.unique(flattened, return_counts=True)
    probabilities = counts/total_cells

    # Compute Shannon entropy
    entropy = - np.sum(probabilities * np.log2(probabilities))

    return entropy

def simulate_CA_with_entropy(size=200, num_seeds=10, steps=500, bias_factor=0.93, decay_factor=0.99, neighborhood_radius=10, wrap_around=False):
    """Run a cellular automata-based angiogenesis model and compute Shannon entropy."""
    background = initialize_background(size)
    occupied = set()
    tumor_factor = 0.1
    tumor_prob = 0.5
    
    # Initialize seeds at random positions
    seeds = [(random.choice([0, size-1]), random.randint(0, size-1)) if random.random() < 0.5 else (random.randint(0, size-1), random.choice([0, size-1])) for _ in range(num_seeds)]
    occupied.update(seeds)
    for x, y in seeds:
        update_background(background, x, y, decay_factor, neighborhood_radius, wrap_around=False)
    
    tumor = create_tumor(size, background, tumor_prob, tumor_factor)

    entropies = []  # To store entropy values over time
    
    for i in range(steps):
        print(f"Step {i}")
        new_seeds = []
        for x, y in seeds:
            nx, ny = move_seed(x, y, background, size, wrap_around, bias_factor)
            new_seeds.append((nx, ny))
            occupied.add((nx, ny))
            update_background(background, x, y, decay_factor, neighborhood_radius, wrap_around=False)
        seeds = new_seeds
        
        # Create a binary grid (1 for occupied, 0 for unoccupied)
        grid = np.zeros((size, size))
        for x, y in tumor:
            grid[x, y] = 2
        for x, y in occupied:
            grid[x, y] = 1

        if i > 400:
            growth_death(background, size, tumor, tumor_factor, 2, occupied, p=0.05)
            
        # Compute entropy and store
        entropy = shannon_entropy(grid)
        entropies.append(entropy)
    
    # Plot the final state
    plt.figure(figsize=(10, 5))
    
    from matplotlib.colors import ListedColormap
    cmap = ListedColormap(["white", "red", "green"])

    print(f"Occupied: {len(occupied)}")
    print(f"Tumor: {len(tumor)}")

    # Plot the growth
    plt.subplot(1, 2, 1)
    plt.imshow(grid, cmap=cmap)
    plt.title("Angiogenesis-Based CA Growth with Stochasticity")
    
    # Plot the entropy over time
    plt.subplot(1, 2, 2)
    plt.plot(entropies, label="Shannon Entropy")
    plt.title("Shannon Entropy Over Time")
    plt.xlabel("Time Step")
    plt.ylabel("Entropy")
    plt.legend()
    plt.tight_layout()
    plt.show()

    return entropies[-1]

last_entropy = simulate_CA_with_entropy()
print("Entropy in last time step: ", last_entropy)