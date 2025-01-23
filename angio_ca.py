import numpy as np
import matplotlib.pyplot as plt
import random

# Adjusted parameters for better accuracy
GRID_SIZE = 200  # Increase resolution
N_STEPS = 1  # More steps for complex vascular structures
D_n = 0.00035  # Endothelial cell diffusion coefficient
rho_0 = 0.34  # Haptotaxis coefficient
chi_0 = 0.38  # Chemotaxis coefficient
alpha = 0.6  # Receptor kinetic parameter
beta = 0.05  # Fibronectin production rate
k = 0.0725  # Fibronectin degradation rate
gamma = 0.1  # Nutrient decay rate
eta = 0.035  # Angiogenic factor uptake
h = 0.005  # Grid spacing
T_branch = 3  # Increased time before branching
k_branch = 0.85  # More modulation for branching
C_inc = 0.02  # TAF increment for stronger chemotaxis
C_max = 25  # Max TAF concentration
C_c = 0.002  # Slightly increased TAF decay rate
D_c = 0.02  # Increased angiogenic factor diffusion

# Initialize grids
grid = np.zeros((GRID_SIZE, GRID_SIZE))  # Blood vessel lattice
taf = np.zeros((GRID_SIZE, GRID_SIZE))  # Tumor angiogenic factor
fibronectin = np.ones((GRID_SIZE, GRID_SIZE))  # ECM support structure

tumor_center = (GRID_SIZE // 2, GRID_SIZE // 2)
taf[tumor_center] = C_max  # Set tumor center with max TAF

# Define proliferating tumor region
def initialize_proliferating_tumor():
    radius = 10  # Tumor growth radius
    for x in range(-radius, radius + 1):
        for y in range(-radius, radius + 1):
            if x**2 + y**2 <= radius**2:  # Ensure circular proliferation
                px, py = tumor_center[0] + x, tumor_center[1] + y
                if 0 <= px < GRID_SIZE and 0 <= py < GRID_SIZE:
                    taf[px, py] = C_max * 0.8  # Set high TAF near tumor

initialize_proliferating_tumor()

# Define initial tip cells at the tumor periphery
tip_cells = [(tumor_center[0] - 15, tumor_center[1]), (tumor_center[0] + 15, tumor_center[1]),
             (tumor_center[0], tumor_center[1] - 15), (tumor_center[0], tumor_center[1] + 15)]

# Function to diffuse TAF
def diffuse_taf():
    global taf
    new_taf = taf.copy()
    for x in range(1, GRID_SIZE - 1):
        for y in range(1, GRID_SIZE - 1):
            new_taf[x, y] += D_c * (
                taf[x-1, y] + taf[x+1, y] + taf[x, y-1] + taf[x, y+1] - 4 * taf[x, y]
            ) - C_c * taf[x, y]
    taf = np.clip(new_taf, 0, C_max)

# Function to move and branch tip cells
def move_tips():
    global tip_cells, grid
    new_tips = []
    for x, y in tip_cells:
        grid[x, y] = 1  # Mark vessel presence

        neighbors = [(x+1, y), (x-1, y), (x, y+1), (x, y-1)]
        random.shuffle(neighbors)
        best_move = max(neighbors, key=lambda pos: taf[pos] if 0 <= pos[0] < GRID_SIZE and 0 <= pos[1] < GRID_SIZE else 0)

        if 0 <= best_move[0] < GRID_SIZE and 0 <= best_move[1] < GRID_SIZE:
            new_tips.append(best_move)

        # Improved branching model
        if random.random() < k_branch and taf[x, y] > C_max * 0.4:
            branch_move = random.choice(neighbors)
            if 0 <= branch_move[0] < GRID_SIZE and 0 <= branch_move[1] < GRID_SIZE:
                new_tips.append(branch_move)

    tip_cells = list(set(new_tips))  # Remove duplicates for anastomosis

# Run simulation
for step in range(N_STEPS):
    diffuse_taf()
    move_tips()

# Plot final vascular network
plt.figure(figsize=(8, 8))
plt.imshow(grid, cmap="Reds", interpolation='nearest')
plt.scatter(tumor_center[1], tumor_center[0], color='blue', label="Tumor Center")
plt.title("Simulated Blood Vessel Network Around Tumor")
plt.legend()
plt.show()


