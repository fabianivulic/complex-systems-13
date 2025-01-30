import numpy as np
import pandas as pd
from skimage import io, color, filters, morphology, transform
import matplotlib.pyplot as plt
from skan import draw, Skeleton
from skan.csr import skeleton_to_nx
import networkx as nx
import os
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats
from growthdeath_jit import simulate_CA, vessel_image

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

def run_sim(network_steps = 20):
    """Run a single simulation
    Inputs:
    network_steps: save network every network_steps timesteps
    
    Returns:
    tumor_grids: list of tumor grids over time
    timesteps: list of timesteps
    """
    print('run')
    bias_factor = 0.93
    decay_factor = 0.99
    breakpoint=350

    vessel_grid, tumor_grid, final_density, cluster_sizes_over_time_tumor, cluster_sizes_over_time_vessel, tumor_grids, timesteps = simulate_CA(
    size=200, 
    seeds_per_edge=5, 
    steps=500, 
    bias_factor=bias_factor, 
    decay_factor=decay_factor, 
    neighborhood_radius=5,
    tumor_prob=0.3,
    wrap_around=False,
    plot=False,
    breakpoint=breakpoint,
    save_networks=True,
    network_steps=network_steps)

    return tumor_grids, timesteps

def run_and_statistics(network_steps = 20):
    results = []
    tumor_grids, timesteps = run_sim(network_steps)
    images_folder = "images_time"  # Folder containing images

    for i, timestep in enumerate(timesteps):
        filename = f'grid_{timestep}.png'
        image_path = os.path.join(images_folder, filename)  # Full path to the image
        
        image = io.imread(image_path)  # Read the image
        average_degree, average_betweenness, average_page_rank, average_cc, _ = network_analysis(
            image, tumor_grids[i], show_skeleton=False, show_graph=False, print_results=False
        )

        results.append({
            'average_degree': average_degree,
            'average_betweenness': average_betweenness,
            'average_page_rank': average_page_rank,
            'average_clustering_coefficient': average_cc
        })

    return results, timesteps


def compute_mean_and_ci(data, confidence=0.95):
    """
    Computes the mean and confidence interval for a given list of lists (data over multiple runs).
    """
    data = np.array(data)
    mean = np.mean(data, axis=0)
    sem = stats.sem(data, axis=0, nan_policy='omit')  # Standard error of the mean
    ci = sem * stats.t.ppf((1 + confidence) / 2., len(data) - 1)  # Confidence interval
    return mean, ci

def run_mulitple(repeat=10, network_steps=20):
    """Run multiple simulations
    Inputs:
    repeat: number of times to repeat the simulation
    network_steps: save network every network_steps timesteps
    
    Returns:
    big_results: list of lists of network statistics
    timesteps: list of timesteps
    """
    big_results = []

    for i in range(repeat):
        results, timesteps = run_and_statistics(network_steps)
        big_results.append(results)

    return big_results, timesteps

def plot(big_results,timesteps): 
    # Extracting network statistics from big_results (list of lists)
    avg_degrees = [[res['average_degree'] for res in run] for run in big_results]
    avg_betweennesses = [[res['average_betweenness'] for res in run] for run in big_results]
    avg_clustering_coefficients = [[res['average_clustering_coefficient'] for res in run] for run in big_results]
    
    # Compute mean and confidence intervals
    mean_degrees, ci_degrees = compute_mean_and_ci(avg_degrees)
    mean_betweennesses, ci_betweennesses = compute_mean_and_ci(avg_betweennesses)
    mean_clustering, ci_clustering = compute_mean_and_ci(avg_clustering_coefficients)
    
    # Creating a 3-subplot figure
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Plot with confidence interval as shaded region
    axes[0].plot(timesteps, mean_degrees, marker='o', label="Average Degree")
    axes[0].fill_between(timesteps, mean_degrees - ci_degrees, mean_degrees + ci_degrees, alpha=0.2)
    axes[0].set_xlabel("Timesteps")
    axes[0].set_ylabel("Average Degree")
    axes[0].set_title("Average Degree over Time")
    axes[0].legend()
    
    axes[1].plot(timesteps, mean_betweennesses, marker='o', label="Average Betweenness")
    axes[1].fill_between(timesteps, mean_betweennesses - ci_betweennesses, mean_betweennesses + ci_betweennesses, alpha=0.2)
    axes[1].set_xlabel("Timesteps")
    axes[1].set_ylabel("Average Betweenness")
    axes[1].set_title("Average Betweenness over Time")
    axes[1].legend()
    
    axes[2].plot(timesteps, mean_clustering, marker='o', label="Average Clustering Coefficient")
    axes[2].fill_between(timesteps, mean_clustering - ci_clustering, mean_clustering + ci_clustering, alpha=0.2)
    axes[2].set_xlabel("Timesteps")
    axes[2].set_ylabel("Average Clustering Coefficient")
    axes[2].set_title("Average Clustering Coefficient over Time")
    axes[2].legend()
    
    plt.tight_layout()
    plt.show()

big_results,timesteps = run_mulitple(repeat=10,network_steps=20)
plot(big_results,timesteps)