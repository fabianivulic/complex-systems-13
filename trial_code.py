import numpy as np
import matplotlib.pyplot as plt
import random


def create_lattice(size, nutrient_source=True):
    """
    Creates a 2D lattice with nutrient gradient.
    - size: Dimension of the lattice
    - nutrient_source: If True, creates a gradient towards the center.
    
    Returns:
    - lattice: A 2D array representing the states of cells (0: empty, 1: vessel, 2: nutrient source).
    - nutrient_grid: A 2D array representing the nutrient gradient.
    """
    lattice = np.zeros((size, size), dtype=int)  # 0: empty, 1: vessel, 2: nutrient source
    nutrient_grid = np.zeros((size, size), dtype=float)

    center = size // 2
    if nutrient_source:
        for i in range(size):
            for j in range(size):
                distance = np.sqrt((i - center) ** 2 + (j - center) ** 2)
                nutrient_grid[i, j] = max(0, 100 - distance * 2)  # Linear decay of nutrient

    lattice[center, center] = 1  # Seed particle at the center
    return lattice, nutrient_grid

def biased_random_walk(start_x, start_y, nutrient_grid):
    """
    Performs a biased random walk influenced by the nutrient gradient.
    
    - start_x, start_y: Starting position of the particle.
    - nutrient_grid: The nutrient concentration guiding the walk.
    
    Returns:
    - x, y: Final position of the particle.
    """

    x, y = start_x, start_y
    size = nutrient_grid.shape[0]

    while True:
        # choose a random direction
        direction = random.choice([(0,1), (0,-1), (1,0), (-1,0)])
        new_x, new_y = x + direction[0], y + direction[1] # random step

        if 0 <= new_x < size and 0 <= new_y < size: # ensure the particle is in the grid
            # perform a step with p = 1 if the new cell has higher nutrient level,
            # otherwise with p = 0.5
            if nutrient_grid[new_x, new_y] > nutrient_grid[x, y] or random.random() < 0.5:
                x, y = new_x, new_y

            return x, y

def dla_growth(lattice, nutrient_grid, steps):
    """
    Simulates DLA growth on the lattice.
    
    - lattice: 2D array representing the lattice (states of cells).
    - nutrient_grid: 2D array representing the nutrient gradient.
    - steps: Number of particles to simulate.
    
    Returns:
    - lattice: Updated lattice after growth.
    """
    size = lattice.shape[0]
    center = size // 2

    for _ in range(steps):
        # Launch a particle from a random position far from the center
        start_x, start_y = random.randint(0, size - 1), random.randint(0, size - 1)
        while np.sqrt((start_x - center) ** 2 + (start_y - center) ** 2) < size // 4:
            start_x, start_y = random.randint(0, size - 1), random.randint(0, size - 1)

        while True:
            x, y = biased_random_walk(start_x, start_y, nutrient_grid)

            # Check if the particle is adjacent to a vessel
            neighborhood = [(-1, 0), (1, 0), (0, -1), (0, 1)]
            for dx, dy in neighborhood:
                neighbor_x, neighbor_y = x + dx, y + dy
                if 0 <= neighbor_x < size and 0 <= neighbor_y < size:
                    if lattice[neighbor_x, neighbor_y] == 1:  # Adjacent to a vessel
                        lattice[x, y] = 1  # Occupy this site
                        nutrient_grid[x, y] += 10  # Reinforce nutrient at this location
                        break
            else:
                # Continue random walk if not adjacent to a vessel
                start_x, start_y = x, y
                continue

            break  # Exit once the particle has been absorbed

    return lattice


def main():
    size = 100
    steps = 1000

    # Create lattice and nutrient gradient
    lattice, nutrient_grid = create_lattice(size, nutrient_source=True)

    # Simulate DLA growth
    lattice = dla_growth(lattice, nutrient_grid, steps)

    # Visualize the results
    plt.imshow(lattice, cmap="viridis", interpolation="nearest")
    plt.title("DLA Blood Vessel Growth")
    plt.colorbar(label="Cell State (0: Empty, 1: Vessel, 2: Nutrient Source)")
    plt.show()

if __name__ == "__main__":
    main()
