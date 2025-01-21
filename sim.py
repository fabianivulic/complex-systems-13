import numpy as np
import matplotlib.pyplot as plt

def create_grid(size, uniform_grid=True):
    """
    Creates a grid with the highest concentration of VEGF at the center.
    Input:
    - size: The size/dimension of the grid
    Output:
    - A grid initialized with decreasing VEGF from the center and random numbers for growth sites
    """
    grid = np.zeros((size, size))
    random_grid = np.random.random((size, size))  #Each site gets a random number assignment
    
    if uniform_grid == False:
        for i in range(size):
            for j in range(size):
                distance = np.sqrt((i - (size // 2))**2 + (j - (size // 2))**2)
                grid[i, j] = (100 / (distance + 1))  # concentration is inverse of distance from the center
    else:
        grid = np.full((size, size), 100)
    return grid, random_grid

def get_neighbors(grid, random_grid, x, y):
    """
    Finds the neighbors of a site and returns their coordinates and their random number.
    Input:
    - grid: The environemnt grid
    - random_grid: The grid with random number assignment
    - x, y: The x and y coordinates of the site
    Output:
    - neighbors: The list of neighbors with their coordinates and random number
    """
    neighbors = []
    neighborhood = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    for horizontal_radius, vertical_radius in neighborhood:
        neighbor_x = x + horizontal_radius
        neighbor_y = y + vertical_radius
        if 0 <= neighbor_x < grid.shape[0] and 0 <= neighbor_y < grid.shape[1]:
            neighbors.append((neighbor_x, neighbor_y, random_grid[neighbor_x, neighbor_y]))
    return neighbors

def ip_model(grid, random_grid, steps, vegf_threshold):
    """
    Sets up the invasion percolation model for angiogenesis according to VEGF concentration.
    Input:
    - grid: The grid with VEGF concentration
    - random_grid: The grid with random number assignment according to the IP model
    - steps: simulation length
    - vegf_threshold: The VEGF concentration necessary to occupy a site
    Output:
    - grid: The updated grid
    - occupied_sites: The list of sites with blood vessel
    """
    occupied_sites = [(grid.shape[0] // 2, grid.shape[1] // 2)]  # Starting at the center
    viable_sites = [(grid.shape[0] // 2, grid.shape[1] // 2)]  # Initial growth from the center
    
    for _ in range(steps):
        new_growth_sites = []
        for x, y in viable_sites:
            neighbors = get_neighbors(grid, random_grid, x, y)
            for horizontal_neighbor, vertical_neighbor, random_value in neighbors:
                #Check if both the site is unoccupied and the VEGF concentration is above the threshold
                if grid[horizontal_neighbor, vertical_neighbor] != -1 and grid[horizontal_neighbor, vertical_neighbor] > vegf_threshold:
                        new_growth_sites.append((horizontal_neighbor, vertical_neighbor, random_value))
                                
        if new_growth_sites:
            # Find the possible site with the lowest random number to be occupied
            min_random_site = min(new_growth_sites, key=lambda site: site[2])
            x, y, _ = min_random_site
            
            grid[x, y] = -1 
            occupied_sites.append((x, y)) 
            viable_sites = [(x, y)]
        else:
            break
    
    return grid, occupied_sites

def main():
    size = 75
    grid, random_grid = create_grid(size, uniform_grid=False)
    center = grid.shape[0] // 2 
    grid[center, center] = -1 
    grid, occupied_sites = ip_model(grid, random_grid, steps=10000, vegf_threshold=0)
    plt.imshow(grid, cmap='hot', interpolation='nearest')
    #plt.colorbar()
    plt.show()
    print(f"Total number of occupied sites: {len(occupied_sites)}")

if __name__ == "__main__":
    main()
