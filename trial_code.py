import numpy as np
import matplotlib.pyplot as plt
import random
from matplotlib.animation import FuncAnimation


def create_lattice(size, nutrient_source=True, nutrient_decay_rate=2):
    """
    Creates a 2D lattice with nutrient gradient.
    - size: Dimension of the lattice
    - nutrient_source: If True, creates a gradient towards the center.
    - nutrient_decay_rate: Controls the rate at which nutrients decrease with distance.
    
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
                nutrient_grid[i, j] = max(0, 100 - distance * nutrient_decay_rate)  # Controlled decay

    lattice[center, center] = 1  # Seed particle at the center
    return lattice, nutrient_grid

def biased_random_walk(start_x, start_y, nutrient_grid, nutrient_threshold=5):
    """
    Performs a biased random walk influenced by the nutrient gradient.
    
    - start_x, start_y: Starting position of the particle.
    - nutrient_grid: The nutrient concentration guiding the walk.
    - nutrient_threshold: Minimum nutrient level for movement to continue.
    
    Returns:
    - x, y: Final position of the particle or None if it stops due to low nutrients.
    """
    x, y = start_x, start_y
    size = nutrient_grid.shape[0]
    
    while True:
        # Check if nutrient level is too low
        if nutrient_grid[x, y] < nutrient_threshold:
            return None  # Particle stops due to low nutrient concentration
        
        # Choose a random direction (up, down, left, right)
        direction = random.choice([(0, 1), (0, -1), (1, 0), (-1, 0)])
        new_x, new_y = x + direction[0], y + direction[1]

        # Ensure the particle stays within bounds
        if 0 <= new_x < size and 0 <= new_y < size:
            # Bias the movement towards higher nutrient concentration
            if nutrient_grid[new_x, new_y] > nutrient_grid[x, y] or random.random() < 0.5:
                x, y = new_x, new_y    
                return x, y



def dla_growth(lattice, nutrient_grid, steps, nutrient_threshold=5, branching_probability=0.1, video_writer = None):
    """
    Simulates DLA growth on the lattice with branching capability.
    
    - lattice: 2D array representing the lattice (states of cells).
    - nutrient_grid: 2D array representing the nutrient gradient.
    - steps: Number of particles to simulate.
    - nutrient_threshold: Minimum nutrient level for growth to continue.
    - branching_probability: Probability of creating a new branch at an occupied site.
    
    Returns:
    - lattice: Updated lattice after growth.
    - nutrient_grid: Updated nutrient grid after growth.
    """
    size = lattice.shape[0]
    center = size // 2

    for step in range(steps):
        # Launch a particle from a random position far from the center
        while True:
            start_x, start_y = random.randint(0, size - 1), random.randint(0, size - 1)
            if np.sqrt((start_x - center) ** 2 + (start_y - center) ** 2) >= size // 4:
                break

        # Debug: Print starting position
        # print(f"Launching particle {step+1} at position ({start_x}, {start_y})")

        # Perform the random walk
        x, y = start_x, start_y
        walk_done = False  # Flag to track if the walk finishes

        while True:
            walk_result = biased_random_walk(x, y, nutrient_grid, nutrient_threshold)
            
            if walk_result is None:
                # Walk ended due to low nutrient level
                #print(f"Particle {step+1} stopped due to low nutrients at position ({x}, {y})")
                break

            x, y = walk_result
            # print(f"Particle {step+1} moved to position ({x}, {y})")

            # Check if the particle is adjacent to a vessel
            neighborhood = [(-1, 0), (1, 0), (0, -1), (0, 1)]
            absorbed = False
            for dx, dy in neighborhood:
                neighbor_x, neighbor_y = x + dx, y + dy
                if 0 <= neighbor_x < size and 0 <= neighbor_y < size:
                    if lattice[neighbor_x, neighbor_y] == 1:  # Adjacent to a vessel
                        # Occupy this site
                        lattice[x, y] = 1
                        # Reinforce nutrient at this location
                        nutrient_grid[x, y] += 10
                        # print(f"Particle {step+1} occupied position ({x}, {y})")

                        # Introduce branching with a certain probability
                        if random.random() < branching_probability:
                            for branch_dx, branch_dy in neighborhood:
                                branch_x, branch_y = x + branch_dx, y + branch_dy
                                if (0 <= branch_x < size and 0 <= branch_y < size 
                                        and lattice[branch_x, branch_y] == 0):
                                    lattice[branch_x, branch_y] = 1
                                    nutrient_grid[branch_x, branch_y] += 10
                                    # print(f"Branch created at position ({branch_x}, {branch_y})")

                        absorbed = True
                        break
            if absorbed:
                break  # Exit the random walk as the particle has been absorbed
                    # Save frame for video

    return lattice, nutrient_grid

def create_animation(image_folder, lattice_snapshots, nutrient_snapshots, video_filename):
    """
    Creates an animation of the DLA growth and saves it as a video.
    
    Parameters:
    - image_folder: Folder containing frames (optional if using lattice_snapshots).
    - lattice_snapshots: List of 2D arrays representing lattice states at each step.
    - nutrient_snapshots: List of 2D arrays representing nutrient grids at each step.
    - video_filename: Output video file name.
    """
    fig, ax = plt.subplots(1, 2, figsize=(14, 7))

    # Initialize the plot elements
    lattice_plot = ax[0].imshow(lattice_snapshots[0], cmap="binary", interpolation="nearest")
    nutrient_plot = ax[1].imshow(nutrient_snapshots[0], cmap="hot", interpolation="nearest")
    fig.colorbar(lattice_plot, ax=ax[0], label="State (0: Empty, 1: Vessel)")
    fig.colorbar(nutrient_plot, ax=ax[1], label="Nutrient Level")

    ax[0].set_title("Vessel Growth")
    ax[0].set_xlabel("X-axis")
    ax[0].set_ylabel("Y-axis")
    ax[1].set_title("Nutrient Gradient")
    ax[1].set_xlabel("X-axis")
    ax[1].set_ylabel("Y-axis")

    def update(frame):
        """Updates the plot for each frame."""
        lattice_plot.set_data(lattice_snapshots[frame])
        nutrient_plot.set_data(nutrient_snapshots[frame])
        ax[0].set_title(f"Vessel Growth (Step {frame})")
        return lattice_plot, nutrient_plot

    # Create the animation
    anim = FuncAnimation(fig, update, frames=len(lattice_snapshots), interval=100, blit=False)

    # Save the animation as a video
    anim.save(video_filename, writer='ffmpeg', fps=10)
    plt.close(fig)
    print(f"Animation saved as {video_filename}")

# Example of how to collect snapshots during DLA growth
def main():
    size = 200
    steps = 200
    nutrient_decay_rate = 1
    nutrient_threshold = 1
    video_filename = "dla_growth_animation.mp4"

    lattice, nutrient_grid = create_lattice(size, nutrient_source=True, nutrient_decay_rate=nutrient_decay_rate)
    lattice_snapshots = []
    nutrient_snapshots = []

    for step in range(steps):
        lattice, nutrient_grid = dla_growth(
            lattice, nutrient_grid, 1, nutrient_threshold=nutrient_threshold, branching_probability=0.7
        )
        if step % 1 == 0:  # Store snapshots for every step
            lattice_snapshots.append(np.copy(lattice))
            nutrient_snapshots.append(np.copy(nutrient_grid))

    create_animation("", lattice_snapshots, nutrient_snapshots, video_filename)

if __name__ == "__main__":
    main()