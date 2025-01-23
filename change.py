import numpy as np
import matplotlib.pyplot as plt
import random

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

def ip_model(grid, random_grid, init, branching_prob, steps, vegf_threshold):
    """
    Growth according to the IP model, with added randomness in site selection.
    """
    occupied_sites = init.copy()  # Starting at the center
    viable_sites = init.copy()  # Initial growth from the center
    
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
                if random.random() < branching_prob and len(elem) >= 2:
                    two_min_sites = tuple(sorted(elem, key=lambda site: site[2])[:2])
                    (x1, y1, _), (x2, y2, _) = two_min_sites
                    
                    # Change the state of the selected neighbors
                    grid[x1, y1] = -1
                    grid[x2, y2] = -1
                    
                    occupied_sites.append((x1, y1))
                    occupied_sites.append((x2, y2))
                    
                    viable_sites[i] = (x1, y1)
                    viable_sites.append((x2, y2))
                    # print(f"Branched: {viable_neighbors}")
                else:
                    min_random_site = min(elem, key=lambda site: site[2])
                    x1, y1, _ = min_random_site

                    # Change state of selected neighbor
                    grid[x1, y1] = -1
                    occupied_sites.append((x1, y1))
                    viable_sites[i] = (x1, y1)
                    # print(f"No branch: {viable_neighbors}")
        else:
            break
    
    return grid, occupied_sites

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

def main():
    # Parameters
    size = 300
    num_walkers = 5
    only_center = False

    grid, random_grid = create_grid(size, uniform_grid=True)
    center = grid.shape[0] // 2

    if only_center == True:
        init = [(center, center)]
        grid[center, center] = -1
    else:
        init = init_walkers(grid, size, num_walkers)

    grid, occupied_sites = ip_model(grid, random_grid, init, branching_prob=0.1, steps=100, vegf_threshold=0)
    plt.imshow(grid, cmap='hot', interpolation='nearest')
    # plt.colorbar()
    plt.show()
    print(f"Total number of occupied sites: {len(occupied_sites)}")

if __name__ == "__main__":
    main()
