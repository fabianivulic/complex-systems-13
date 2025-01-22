import numpy as np
import random
import matplotlib.pyplot as plt
import imageio
from tqdm import tqdm

class DLASimulation:
    """
    A class to simulate Diffusion-Limited Aggregation (DLA) with neighbor-dependent sticking probability.
    """
    
    def __init__(self, grid_size=300, num_particles=2000, launch_radius=50, kill_radius=60, 
                 stick_prob=0.8, neighbor_dependency=0.5, num_seeds=1, save_gif=False):
        """Initializes the DLA simulation with given parameters."""
        self.grid_size = grid_size
        self.num_particles = num_particles
        self.launch_radius = launch_radius
        self.kill_radius = kill_radius
        self.stick_prob = stick_prob
        self.neighbor_dependency = neighbor_dependency  # Controls dependence on number of neighbors
        self.num_seeds = num_seeds
        self.save_gif = save_gif
        self.grid = self.initialize_grid()
        self.frames = []

    def initialize_grid(self):
        """Initializes the simulation grid with seed positions."""
        grid = np.zeros((self.grid_size, self.grid_size), dtype=int)
        seed_positions = [(self.grid_size//2, self.grid_size//2) for _ in range(self.num_seeds)]
        
        for x, y in seed_positions:
            grid[x, y] = 1  # Mark seed positions
        return grid

    def count_neighbors(self, x, y):
        """Counts the number of occupied neighboring sites."""
        neighbors = [(1, 0), (-1, 0), (0, 1), (0, -1)]
        return sum(self.grid[x + dx, y + dy] == 1 for dx, dy in neighbors if 0 <= x + dx < self.grid_size and 0 <= y + dy < self.grid_size)

    def random_walk(self):
        """Simulates a single particle's random walk until it sticks or is removed."""
        angle = random.uniform(0, 2 * np.pi)
        x = int(self.grid_size // 2 + self.launch_radius * np.cos(angle))
        y = int(self.grid_size // 2 + self.launch_radius * np.sin(angle))

        while True:
            if np.linalg.norm([x - self.grid_size//2, y - self.grid_size//2]) > self.kill_radius:
                return None  # Remove particle if it reaches kill boundary
            
            # Choose a random movement
            dx, dy = random.choice([(1, 0), (-1, 0), (0, 1), (0, -1)])
            x_new, y_new = x + dx, y + dy
            
            if 0 <= x_new < self.grid_size and 0 <= y_new < self.grid_size:
                x, y = x_new, y_new
                
                # Check if adjacent to an occupied site
                if any(self.grid[x + dx, y + dy] == 1 for dx, dy in [(1, 0), (-1, 0), (0, 1), (0, -1)]):
                    num_neighbors = self.count_neighbors(x, y)
                    
                    # Adjust stick probability based on neighbors
                    adjusted_prob = self.stick_prob * (1 - self.neighbor_dependency + self.neighbor_dependency * (num_neighbors / 4))
                    
                    if random.random() < adjusted_prob:
                        return x, y  # Particle sticks

    def run(self):
        """Runs the DLA simulation by adding particles one at a time with a progress bar."""
        for _ in tqdm(range(self.num_particles), desc="Simulation Progress"):
            result = self.random_walk()
            if result:
                x, y = result
                self.grid[x, y] = 1
                
                if self.save_gif and _ % 50 == 0:  # Capture frames every 50 steps
                    self.frames.append(np.uint8(self.grid * 255))  # Convert grid to 8-bit image format
        
        if self.save_gif:
            imageio.mimsave('dla_simulation.gif', self.frames, duration=0.1, format='GIF', loop=0)

    def plot(self):
        """Plots the final state of the DLA simulation."""
        plt.figure(figsize=(10,10))
        plt.imshow(self.grid, cmap='gray')
        plt.axis('off')
        plt.show()

# Example run
dla = DLASimulation(save_gif=True, neighbor_dependency=0.9)  # Adjust neighbor_dependency between 0 and 1
dla.run()
dla.plot()
