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
from matplotlib.animation import FuncAnimation
from scipy.interpolate import make_interp_spline

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
            if tumor_radius > distance > empty_radius and random.random() < tumor_prob:
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
def growth_death(background, size, tumor, tumor_factor, radius, occupied, p, midpoint_sigmoid, steepness):
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

    # Extra line for calculating Von Neumann neighbor count if necessary
    # total_neighborhood = 2 * radius * (radius + 1)

    for x in range(size):
        for y in range(size):
            distance = np.sqrt((x - center) ** 2 + (y - center) ** 2)

            # if not(tumor_radius > distance > empty_radius) or not tumor[x, y]:
            #     continue

            if not(distance > empty_radius) or not tumor[x, y]:
                continue

            blood_count = len(check_blood(x, y, occupied, radius))
            blood_bias = 1 / (1 + np.exp(-steepness * (blood_count-midpoint_sigmoid)))
            growth, death = p * (blood_bias), p * (1-blood_bias)
            
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
def update_background(background, x, y, decay_factor, radius, wrap_around=True):
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
    sigma = radius / 2

    for dx in range(-radius, radius + 1):
        for dy in range(-radius, radius + 1):
            distance = np.sqrt(dx**2 + dy**2)

            if distance > radius:
                continue

            weight = np.exp(-distance**2 / (2 * sigma**2))
            nx, ny = x + dx, y + dy

            if wrap_around:
                nx %= size
                ny %= size

            if 0 <= nx < size and 0 <= ny < size:
                background[nx, ny] = max(0, background[nx, ny] * (1 - weight * (1 - decay_factor)))

@njit
def move_seed(x, y, background, size, wrap_around, bias_factor, tumor_grid):
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
    # neighbors = [(nx, ny) for (nx, ny) in neighbors if not tumor_grid[nx, ny]]

    move_probabilities = [(random.random() * (1 - bias_factor) + background[nx, ny] * bias_factor, nx, ny) for nx, ny in neighbors]
    move_probabilities.sort(reverse=True) # Favor higher VEGF concentration with stochasticity

    return move_probabilities[0][1], move_probabilities[0][2]

@njit
def shannon_entropy(grid, tumor_grid):
    """
    Compute Shannon entropy for a grid considering 3 states and limiting the region to tumor radius.
    Input:
    - grid: The grid with tumor cells and blood vessels
    Output:
    - The Shannon entropy value
    """    
    tumor_cells = np.sum(tumor_grid)
    size = grid.shape[0]
    total_cells = size * size
    tumor_density = tumor_cells / total_cells
    
    return tumor_density

def simulate_CA(size=200, seeds_per_edge=5, steps=500, bias_factor=0.93, decay_factor=0.99, neighborhood_radius=10, tumor_prob=0.5, wrap_around=False, plot=True, breakpoint=350, p=0.1, plot_steps = 5, midpoint_sigmoid=1, steepness=1):
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
    - plot_steps: The number of evenly spaced time steps to plot
    Output:
    - The Shannon entropy value in the last time step
    """
    background = initialize_background(size)
    vessel_grid = np.zeros((size, size), dtype=np.bool_) # Need to be separately delineated 
    tumor_grid = np.zeros((size, size), dtype=np.bool_)  # so they can occupy the same space
    tumor_factor = 0.1
    
    # Initialize seeds for blood vessels at random positions
    seeds = initialize_seeds(size, seeds_per_edge)
    for x, y in seeds:
        vessel_grid[x, y] = True
        update_background(background, x, y, decay_factor, neighborhood_radius, wrap_around=False)
    
    # Initialize tumor cells
    tumor_grid = create_tumor(size, background, tumor_prob, tumor_factor)
    entropies = []
    cluster_sizes_over_time = []
    cluster_counts = []
    
    for i in range(steps):
        new_seeds = []
        if i < breakpoint:
            for x, y in seeds:
                nx, ny = move_seed(x, y, background, size, wrap_around, bias_factor, tumor_grid)
                new_seeds.append((nx, ny))
                vessel_grid[nx, ny] = True 
                update_background(background, x, y, decay_factor, neighborhood_radius, wrap_around=False)
            seeds = new_seeds
        
        # Introduce growth and death of tumor cells after a certain time step
        #if i > breakpoint:
        growth_death(background, size, tumor_grid, tumor_factor, 2, vessel_grid, p, midpoint_sigmoid, steepness)
        
        # Combine grids for visualization
        grid = np.zeros((size, size))
        grid[vessel_grid] = 1  # Blood vessels
        grid[tumor_grid] = 2   # Tumor cells
        
        # Calculate entropy for tumor cells
        entropy = shannon_entropy(grid, tumor_grid.astype(np.float64))
        entropies.append(entropy)

        if i % plot_steps == 0:
            # Calculate number of tumor clusters
            tumor_coordinates = set(zip(*np.where(tumor_grid)))
            cluster_count, cluster_sizes = tumor_clusters(size, tumor_coordinates, wrap_around, plot=False)
            cluster_sizes_over_time.append(cluster_sizes)
            cluster_counts.append(cluster_count)
    
    # Plotting the visualization and tumor entropy, and cluster count over time
    if plot:
        plt.figure(figsize=(10, 5))
        cmap = ListedColormap(["white", "red", "green"])
        print(f"Number of blood vessel pixels: {np.sum(vessel_grid)}")
        print(f"Number of tumor pixels: {np.sum(tumor_grid)}")
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

        plt.plot(cluster_counts, label="Number of Tumor Clusters", color = "orange")
        plt.title("Tumor Clustering Over Time")
        plt.xlabel("Time Step")
        plt.ylabel("Number of Clusters")
        plt.legend()
        plt.tight_layout()
        plt.show()
    
    return vessel_grid, tumor_grid, entropies[-1], cluster_sizes_over_time

def animate_histogram(cluster_sizes_over_time, plot_steps, plot=False):
    """
    Create an animated histogram showing the distribution of cluster sizes over time,
    with a fitted curve overlayed on the bars and a fixed y-axis.
    """
    # Find the maximum frequency across all time steps
    max_frequency = 0
    for cluster_sizes in cluster_sizes_over_time:
        frequencies, _ = np.histogram(cluster_sizes, bins=range(1, max(cluster_sizes) + 2))
        max_frequency = max(max_frequency, max(frequencies))

    # Set up the figure and axis
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.set_title("Cluster Size Distribution Over Time")
    ax.set_xlabel("Cluster Size")
    ax.set_ylabel("Frequency")
    ax.set_ylim(0, max_frequency + 1)  # Fix the y-axis

    def update(frame):
        ax.clear()  # Clear the previous histogram and curve
        cluster_sizes = cluster_sizes_over_time[frame]
        
        # Histogram data
        frequencies, bin_edges = np.histogram(cluster_sizes, bins=range(1, max(cluster_sizes) + 2))
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2  # Compute bin centers
        
        # Plot histogram
        ax.hist(cluster_sizes, bins=bin_edges, color="blue", edgecolor="black", alpha=0.7, label="Histogram")
        
        # Fit a smooth curve to the histogram data
        if len(bin_centers) > 3:  # Fit only if enough data points exist
            spline = make_interp_spline(bin_centers, frequencies, k=3)  # Cubic spline for smooth curve
            x_smooth = np.linspace(bin_centers[0], bin_centers[-1], 200)  # Dense x-values for smooth curve
            y_smooth = spline(x_smooth)
            ax.plot(x_smooth, y_smooth, color="red", linewidth=2, label="Fitted Curve")
        
        # Set titles and labels
        ax.set_title(f"Cluster Size Distribution at Time Step {frame * plot_steps}")
        ax.set_xlabel("Cluster Size")
        ax.set_ylabel("Frequency")
        ax.set_ylim(0, max_frequency + 1)  # Keep the y-axis fixed
        ax.legend()

    # Create the animation
    anim = FuncAnimation(fig, update, frames=len(cluster_sizes_over_time), repeat=False)
    plt.show()

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

def tumor_clusters(size, tumor_grid, wrap_around = False, plot = True):
    """
    Analyze tumor clustering over time.
    Input:
    - size: The size of the grid
    - tumor_grid: The grid with tumor cells
    - wrap_around: A boolean to enable periodic boundaries
    - plot: A boolean to enable plotting

    Output:
    - cluster counts: The number of tumor clusters over time
    """
    visited = set()
    clusters = 0
    cluster_sizes = []

    for cell in tumor_grid:
        if cell not in visited:
            # Depth-first search to find all connected tumor cells
            stack = [cell]
            cluster_nodes = []
            while stack:
                cx, cy = stack.pop()
                if (cx, cy) not in visited:
                    visited.add((cx, cy))
                    cluster_nodes.append((cx, cy))
                    neighbors = get_neighbors(cx, cy, size, wrap_around)
                    stack.extend(n for n in neighbors if n in tumor_grid)

            if len(cluster_nodes) >= 1: # Minimum cluster size
                clusters += 1
                cluster_sizes.append(len(cluster_nodes))

    return clusters, cluster_sizes

def main():
    """
    Main function to execute the simulation.
    """
    size = 200
    seeds_per_edge = 5
    steps = 500
    bias_factor = 0.93
    decay_factor = 0.99
    neighborhood_radius = 10
    tumor_prob = 0.3
    wrap_around = False
    breakpoint = 350
    p = 0.1
    midpoint_sigmoid = 1
    steepness = 1

    vessel_grid, _, _, cluster_sizes_over_time = simulate_CA(
        size=size,
        seeds_per_edge=seeds_per_edge,
        steps=steps,
        bias_factor=bias_factor,
        decay_factor=decay_factor,
        neighborhood_radius=neighborhood_radius,
        tumor_prob=tumor_prob,
        wrap_around=wrap_around,
        plot=False,
        breakpoint=breakpoint, 
        p=p,
        plot_steps=10,
        midpoint_sigmoid=midpoint_sigmoid,
        steepness=steepness
    )
    # vessel_image(vessel_grid, 'final_grid.png')
    # animate_histogram(cluster_sizes_over_time, 10, plot=False)

if __name__ == "__main__":
    start_time = time.time()
    main()
    elapsed_time = time.time() - start_time
    print(f"Total execution time: {elapsed_time:.6f} seconds.")
