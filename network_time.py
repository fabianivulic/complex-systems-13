import numpy as np
import pandas as pd
from skimage import io, color, filters, morphology, transform
import matplotlib.pyplot as plt
from skan import draw, Skeleton
from skan.csr import skeleton_to_nx
import networkx as nx
import os
from growthdeath_jit import simulate_CA, vessel_image
# from network_analysis import network_analysis

def network_analysis(image, tumor_grid, show_skeleton=True, show_graph=True, print_results=True):
    # Convert the image to grayscale and binarize
    image = image[:, :, :3]
    image = color.rgb2gray(image)
    threshold = filters.threshold_otsu(image)
    binary_image = image > threshold
    
    expected_size = (200, 200)
    binary_image = transform.resize(binary_image, expected_size, anti_aliasing=False)
    if binary_image.shape[:2] != expected_size:
        raise ValueError(f"Skeletonized image must be {expected_size}, but got {binary_image.shape[:2]}")
    
    # Skeletonize the image
    skeleton = morphology.skeletonize(binary_image)
    if show_skeleton:
        _, ax = plt.subplots()
        draw.overlay_skeleton_2d(binary_image, skeleton, dilate=1, axes=ax)
        plt.title("Skeletonized Image")
        plt.show()
    
    skan_skeleton = Skeleton(skeleton)
    multigraph = skeleton_to_nx(skan_skeleton)

    # Convert the multigraph to a simple graph for later analysis
    graph = nx.Graph()
    for u, v, data in multigraph.edges(data=True):
        if u != v:
            if not graph.has_edge(u, v):
                graph.add_edge(u, v, **data)
    valid_nodes = list(graph.nodes)
    pos = {node: skan_skeleton.coordinates[node] for node in valid_nodes}
    pos = {node: (coord[1], coord[0]) for node, coord in pos.items()}  # Convert (y, x) to (x, y)

    center_node = max(graph.nodes) + 1
    graph.add_node(center_node)
    size = min(binary_image.shape[0], binary_image.shape[1])
    center_position = np.array([binary_image.shape[0] / 2, binary_image.shape[1] / 2])

    # Add the center node's position to the pos dictionary
    pos[center_node] = (center_position[1], center_position[0])

    # Connect the center node to valid graph nodes within the radius if they have a neighbor from the tumor grid
    for node in valid_nodes:
        node_position = np.array([pos[node][1], pos[node][0]])
        neighbors = list(graph.neighbors(node))
        if any(tumor_grid[int(pos[neighbor][1]), int(pos[neighbor][0])] for neighbor in neighbors):
            graph.add_edge(center_node, node)

    if show_graph:
        nx.draw(graph, pos, node_size=10, font_size=8, edge_color='blue', node_color='red', with_labels=False)
        plt.title("Graph Representation with Center Node")
        plt.show()

    # Calculate network analysis measures
    average_degree = np.mean(list(dict(graph.degree()).values()))
    average_betweenness = np.mean(list(nx.betweenness_centrality(graph).values()))
    average_page_rank = np.mean(list(nx.pagerank(graph).values()))
    average_clustering_coefficient = nx.average_clustering(graph)
    degree_distribution = np.array(list(dict(graph.degree()).values()))

    if print_results:
        print(f"Average degree: {average_degree}")
        print(f"Average betweenness: {average_betweenness}")
        print(f"Average page rank: {average_page_rank}")
        print(f"Average clustering coefficient: {average_clustering_coefficient}")

    return average_degree, average_betweenness, average_page_rank, average_clustering_coefficient, degree_distribution


num_runs = 1
bias_factor = 0.93
decay_factor = 0.99
breakpoint=350
results = []

# run code, saving the images
def run_sim():
    for run in range(num_runs):
        print(f'Running simulation for constant factors, run {run + 1}...')
        vessel_grid, tumor_grid, final_density, cluster_sizes_over_time, tumor_grids, timesteps = simulate_CA(
        size=200, 
        seeds_per_edge=5, 
        steps=500, 
        bias_factor=bias_factor, 
        decay_factor=decay_factor, 
        neighborhood_radius=5,
        tumor_prob=0.5,
        wrap_around=False,
        plot=False,
        breakpoint=breakpoint,
        save_networks=True)
    return tumor_grids, timesteps

tumor_grids, timesteps = run_sim()

results = []
timesteps_str = ['0', '50', '100', '150', '200', '250', '300', '350', '400', '450']
# timesteps = []
images_folder = "images_time"  # Folder containing images
print(os.listdir(images_folder))
# for i, filename in enumerate(os.listdir(images_folder)):  # Iterate over files in the folder
for i in range(len(os.listdir(images_folder))):
    # timesteps.append(int(filename.split("_")[1].split(".")[0]))
    
    filename = f'grid_{timesteps_str[i]}.png'
    print(filename)
    print(timesteps)
    # timestep = timesteps[i]
    image_path = os.path.join(images_folder, filename)  # Full path to the image
    
    image = io.imread(image_path)  # Read the image
    average_degree, average_betweenness, average_page_rank, average_cc, _ = network_analysis(
        image, tumor_grids[i], show_skeleton=False, show_graph=False, print_results=False
    )

    results.append({
        #'run': run + 1,
        #'final_density': final_density,
        'average_degree': average_degree,
        'average_betweenness': average_betweenness,
        'average_page_rank': average_page_rank,
        'average_clustering_coefficient': average_cc
    })

def plot(results, timesteps): 
    # Extracting network statistics from results
    average_degrees = [res['average_degree'] for res in results]
    average_betweennesses = [res['average_betweenness'] for res in results]
    average_page_ranks = [res['average_page_rank'] for res in results]
    average_clustering_coefficients = [res['average_clustering_coefficient'] for res in results]

    # Creating a 4-subplot figure
    fig, axes = plt.subplots(1,3, figsize=(12, 5))

    # Plot each network statistic
    axes[0].scatter(timesteps, average_degrees, marker='o', label="Average Degree")
    axes[0].set_xlabel("Timesteps")
    axes[0].set_ylabel("Average Degree")
    axes[0].set_title("Average Degree over Time")
    axes[0].legend()

    axes[1].scatter(timesteps, average_betweennesses, marker='o',label="Average Betweenness")
    axes[1].set_xlabel("Timesteps")
    axes[1].set_ylabel("Average Betweenness")
    axes[1].set_title("Average Betweenness over Time")
    axes[1].legend()

    axes[2].scatter(timesteps, average_clustering_coefficients, marker='o', label="Avg Clustering Coefficient")
    axes[2].set_xlabel("Timesteps")
    axes[2].set_ylabel("Average Clustering Coefficient")
    axes[2].set_title("Average Clustering Coefficient over Time")
    axes[2].legend()

    # Adjust layout for better readability
    plt.tight_layout()
    plt.show()

plot(results, timesteps)