import numpy as np
import matplotlib.pyplot as plt
import random
from matplotlib.animation import FuncAnimation

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

def ip_model(grid, random_grid, init, steps, vegf_threshold, r, n):
    """
    Growth according to the IP model, with added randomness in site selection and VEGF depletion.
    """
    occupied_sites = init.copy()  # Starting at the center
    viable_sites = init.copy()  # Initial growth from the center
    lattice_snapshots = []
    vegf_snapshots = []
    
    for _ in range(steps):
        new_growth_sites = []

        for x, y in viable_sites:
            neighbors = get_neighbors(grid, random_grid, x, y)

            viable_neighbors = []
            for hor, ver, random_value in neighbors:
                # If there's enough VEGF around, add that to possible new sites for growth
                if grid[hor, ver] != -1 and grid[hor, ver] > vegf_threshold:
                    viable_neighbors.append((hor, ver, random_value))

            new_growth_sites.append(viable_neighbors)

        if new_growth_sites:               
            for i, elem in enumerate(new_growth_sites):
                # Check if elem exists
                if not elem:
                    continue

                # Find the possible site with the lowest random number to be occupied
                min_random_site = min(elem, key=lambda site: site[2])
                x, y, _ = min_random_site
                
                # Change state of selected neighbor
                grid[x, y] = -1
                decrease_vegf(grid, x, y, r, n) # Decrease VEGF in a circular area
                occupied_sites.append((x, y))
                viable_sites[i] = (x, y)
        else:
            break

        # Store snapshots only at intervals (e.g., every 10 steps)
        if steps % 10 == 0:  # Adjust the interval to control how many frames are saved
            lattice_snapshots.append(np.copy(grid))
            vegf_snapshots.append(np.copy(random_grid))

    return grid, occupied_sites, lattice_snapshots, vegf_snapshots

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

def create_animation(lattice_snapshots, vegf_snapshots, video_name):
    """
    Creates an animation of the IP model growth and saves it as a video.
    """
    fig, ax = plt.subplots(1, 2, figsize=(14, 7))

    # Initialize the plot elements
    lattice_plot = ax[0].imshow(lattice_snapshots[0], cmap="binary", interpolation="nearest")
    vegf_plot = ax[1].imshow(vegf_snapshots[0], cmap="hot", interpolation="nearest")
    fig.colorbar(lattice_plot, ax=ax[0], label="State (0: Empty, -1: Occupied)")
    fig.colorbar(vegf_plot, ax=ax[1], label="VEGF Level")

    ax[0].set_title("Vessel Growth")
    ax[0].set_xlabel("X-axis")
    ax[0].set_ylabel("Y-axis")
    ax[1].set_title("VEGF Gradient")
    ax[1].set_xlabel("X-axis")
    ax[1].set_ylabel("Y-axis")

    def update(frame):
        """Updates the plot for each frame."""
        lattice_plot.set_data(lattice_snapshots[frame])
        vegf_plot.set_data(vegf_snapshots[frame])
        ax[0].set_title(f"Vessel Growth (Step {frame})")
        return lattice_plot, vegf_plot

    # Create the animation
    anim = FuncAnimation(fig, update, frames=len(lattice_snapshots), interval=100, blit=False)

    # Save the animation as a video
    anim.save(video_name, writer='ffmpeg', fps=10)
    plt.close(fig)
    print(f"Animation saved as {video_name}")


def main():
    # Parameters
    size = 400
    num_walkers = 10
    only_center = False
    video_name = "vegf_growth_animation.mp4"

    grid, random_grid = create_grid(size, uniform_grid=True)
    center = grid.shape[0] // 2

    if only_center == True:
        init = [(center, center)]
        grid[center, center] = -1
    else:
        init = init_walkers(grid, size, num_walkers)

    grid, occupied_sites, lattice_snapshots, vegf_snapshots = ip_model(grid, random_grid, init, steps=5000, vegf_threshold=0, r = 3, n = 0.01)
    plt.imshow(grid, cmap='hot', interpolation='nearest')
    # plt.colorbar()
    plt.show()
    print(f"Total number of occupied sites: {len(occupied_sites)}")
    create_animation(lattice_snapshots, vegf_snapshots, video_name)

if __name__ == "__main__":
    main()
