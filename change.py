import numpy as np
import matplotlib.pyplot as plt
import random
from matplotlib.animation import FuncAnimation, PillowWriter

def create_grid(size, uniform_grid=True):
    """
    Creates a grid with the highest concentration of VEGF at the center.

    Parameters
    ----------
    size : int
        The size (dimension) of the grid.

    Returns
    -------
    numpy.ndarray
        A grid initialized with a concentration of VEGF decreasing from the center, 
        along with random numbers for growth sites.
    """
    grid = np.zeros((size, size))
    random_grid = np.random.random((size, size)) # Each site gets a random number assignment
    
    if uniform_grid == False:
        for i in range(size):
            for j in range(size):
                distance = np.sqrt((i - (size // 2))**2 + (j - (size // 2))**2)
                grid[i, j] = (100 / (distance + 1)) # Concentration is inverse of distance from the center
    else:
        grid = np.full((size, size), 100)
    return grid, random_grid

def get_neighbors(grid, random_grid, x, y):
    """
    Finds the neighbors of an occupied site and returns their coordinates,
    along with the random values for those sites.
    """
    neighbors = []
    neighborhood = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    for horizontal_radius, vertical_radius in neighborhood:
        neighbor_x = x + horizontal_radius
        neighbor_y = y + vertical_radius
        if 0 <= neighbor_x < grid.shape[0] and 0 <= neighbor_y < grid.shape[1]:
            neighbors.append((neighbor_x, neighbor_y, random_grid[neighbor_x, neighbor_y]))
    return neighbors

def decrease_vegf(grid, x, y, r, n):
    """
    Decrease VEGF in a circular area of radius "r" around (x,y) by "n".
    """
    size = grid.shape[0]
    for i in range(-r, r + 1):
        for j in range(-r, r + 1):
            if i**2 + j**2 <= r**2: # Check if (i,j) is in the circle
                nx, ny = x + i, y + j
                if 0 <= nx < size and 0 <= ny <= size:
                    grid[nx, ny] = max(grid[nx, ny] - n, 0) # Decrease VEGF, but not negative number
    
def vessel_diameter(grid, random_grid, occupied_sites, vegf_radius = 2):
    """
    Calculates the diameter of vessels based on VEGF concentration in the neighborhood.

    Parameters
    grid : numpy.ndarray
        The simulation grid representing vessel growth.
    random_grid : numpy.ndarray
        The grid of random VEGF values.
    occupied_sites : list of tuples
        Coordinates of occupied sites (vessel positions).
    vegf_radius : int
        Radius around the vessel to consider for VEGF averaging.

    Returns
    -------
    dict
        A dictionary with vessel coordinates as keys and their calculated diameters as values.
    """
    diameters = {}
    for x, y in occupied_sites:
        # Get VEGF values in a circular neighborhood
        local_vegf = []
        for i in range(-vegf_radius, vegf_radius + 1):
            for j in range(-vegf_radius, vegf_radius + 1):
                if i**2 + j**2 <= vegf_radius**2:
                    nx, ny = x + i, y + j
                    if 0 <= nx < grid.shape[0] and 0 <= ny < grid.shape[1]:
                        local_vegf.append(random_grid[nx, ny])
        
        # Calculate mean VEGF in the neighborhood
        mean_vegf = np.mean(local_vegf) if local_vegf else 0
        # Use a power-law-like relationship to assign diameter
        diameter = 1 + mean_vegf ** 0.5  # Example: diameter scales with sqrt of VEGF
        diameters[(x, y)] = diameter
    
    return diameters

def ip_model_with_diameters(grid, random_grid, init, steps, vegf_threshold, r, n, interval=5):
    """
    Growth according to the IP model, calculating vessel diameters at regular intervals.
    """
    occupied_sites = init.copy()
    viable_sites = init.copy()
    lattice_snapshots = []
    vegf_snapshots = []
    diameter_snapshots = []

    for step in range(steps):
        new_growth_sites = []

        for x, y in viable_sites:
            neighbors = get_neighbors(grid, random_grid, x, y)

            viable_neighbors = []
            for hor, ver, random_value in neighbors:
                if grid[hor, ver] != -1 and grid[hor, ver] > vegf_threshold:
                    viable_neighbors.append((hor, ver, random_value))

            new_growth_sites.append(viable_neighbors)

        if new_growth_sites:               
            for i, elem in enumerate(new_growth_sites):
                if not elem:
                    continue

                min_random_site = min(elem, key=lambda site: site[2])
                x, y, _ = min_random_site
                grid[x, y] = -1
                decrease_vegf(grid, x, y, r, n)
                occupied_sites.append((x, y))
                viable_sites[i] = (x, y)
        else:
            break

        # Capture snapshots at specified intervals
        if step % interval == 0:
            lattice_snapshots.append(np.copy(grid))
            vegf_snapshots.append(np.copy(random_grid))
            diameters = vessel_diameter(grid, random_grid, occupied_sites)
            diameter_snapshots.append(diameters)

    return grid, occupied_sites, lattice_snapshots, vegf_snapshots, diameter_snapshots

def init_walkers(grid, size, num_walkers):
    init = []

    for _ in range(num_walkers):
        while True:
            x, y = np.random.randint(0, size, size=2)
            if grid[x, y] != -1: # Ensure the walker spawns in an empty location
                grid[x, y] = -1 # Mark the starting position
                init.append((x, y))
                break

    return init

def create_vegf_growth_animation(lattice_snapshots, vegf_snapshots, output_file="vegf_growth.gif"):
    fig, ax = plt.subplots(1, 2, figsize=(12, 6))
    im1 = ax[0].imshow(lattice_snapshots[0], cmap="viridis", origin="lower")
    im2 = ax[1].imshow(vegf_snapshots[0], cmap="hot", origin="lower")
    ax[0].set_title("Vessel Growth")
    ax[1].set_title("VEGF Concentration")

    def update(frame):
        im1.set_array(lattice_snapshots[frame])
        im2.set_array(vegf_snapshots[frame])
        return [im1, im2]

    ani = FuncAnimation(fig, update, frames=len(lattice_snapshots), interval=200, blit=True)
    ani.save(output_file, writer=PillowWriter(fps=5))
    plt.close()

def create_diameter_histogram_animation(diameter_snapshots, output_file="diameter_histogram.gif"):
    fig, ax = plt.subplots(figsize=(8, 6))
    bins = np.linspace(1, 4, 100)  # Adjust bins as needed
    ax.set_title("Diameter Distribution")
    ax.set_xlabel("Diameter")
    ax.set_ylabel("Frequency")
    bar_container = ax.hist([], bins=bins, color="blue", alpha=0.7)[2]

    def update(frame):
        diameters = list(diameter_snapshots[frame].values())
        for rect, height in zip(bar_container, np.histogram(diameters, bins=bins)[0]):
            rect.set_height(height)
        ax.set_ylim(0, max(np.histogram(diameters, bins=bins)[0]) + 3)
        return bar_container

    ani = FuncAnimation(fig, update, frames=len(diameter_snapshots), interval=200, blit=True)
    ani.save(output_file, writer=PillowWriter(fps=5))
    plt.close()

def main():
    size = 500
    num_walkers = 10
    steps = 5000
    interval = 10

    grid, random_grid = create_grid(size, uniform_grid=True)
    init = init_walkers(grid, size, num_walkers)

    grid, occupied_sites, lattice_snapshots, vegf_snapshots, diameter_snapshots = ip_model_with_diameters(
        grid, random_grid, init, steps, vegf_threshold=0, r=2, n=0.00001, interval=interval
    )

    create_vegf_growth_animation(lattice_snapshots, vegf_snapshots, output_file="vegf_growth.gif")
    create_diameter_histogram_animation(diameter_snapshots, output_file="diameter_histogram.gif")

if __name__ == "__main__":
    main()