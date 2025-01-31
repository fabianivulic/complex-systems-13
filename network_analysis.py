"""
This file contains the code for the network analysis of the simulated images.  
The network analysis measures that are calculated are the average degree, average degree betweenness, 
average page rank, and average clustering coefficient. 
"""
import numpy as np
import pandas as pd
from skimage import io, color, filters, morphology, transform
import matplotlib.pyplot as plt
from skan import draw, Skeleton
from skan.csr import skeleton_to_nx
import networkx as nx
from growthdeath_jit import simulate_CA, vessel_image

def network_analysis(image, tumor_grid, show_skeleton=True, show_graph=True, print_results=True):
    """
    Perform network analysis on the skeletonized image.
    Input:
    - image: The original blood vessel image
    - tumor_grid: The tumor grid
    - show_skeleton: Whether to show the skeletonized image
    - show_graph: Whether to show the graph representation
    - print_results: Whether to print the network analysis results
    Output:
    - average_degree: The average degree of the graph
    - average_betweenness: The average betweenness centrality of the graph
    - average_page_rank: The average page rank of the graph
    - average_clustering_coefficient: The average clustering coefficient of the graph
    - degree_distribution: The degree distribution of the graph
    """
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

def run_experiments():
    """
    Run the experiments for the different control parameters. 
    Data from the experiments is saved in the data folder.
    No input or output arguments.
    """
    experiment_type = input('Enter the experiment type (bias_factor, prolif_prob, midpoint_sigmoid): ')
    num_runs = int(input('Enter the number of runs for each experimental value: '))
    
    if experiment_type == 'bias_factor':
        bias_factors = np.linspace(0, 0.99, 20)
        decay_factor = 0.99
        breakpoint = 350
        results = []
        for bias_factor in bias_factors:
            for run in range(num_runs):
                print(f'Running simulation for bias factor {bias_factor}, run {run + 1}...')
                vessel_grid, tumor_grid, final_density, _, _ = simulate_CA(
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
                    p=0.1, 
                    plot_steps = 5, 
                    midpoint_sigmoid=1, 
                    steepness=1)
                
                vessel_image(vessel_grid, 'final_grid.png')
                image = io.imread('images/final_grid.png')
                average_degree, average_betweenness, average_page_rank, average_cc, _ = network_analysis(image, tumor_grid, show_skeleton=False, show_graph=False, print_results=False)
                results.append({
                    'run': run + 1,
                    'bias_factor': bias_factor,
                    'final_density': final_density,
                    'average_degree': average_degree,
                    'average_betweenness': average_betweenness,
                    'average_page_rank': average_page_rank,
                    'average_clustering_coefficient': average_cc
                })
                print()
        
        df = pd.DataFrame(results)
        df.to_csv(f'data/{experiment_type}_results_1.csv', index=False)
    
    elif experiment_type == 'prolif_prob':
        bias_factor = 0.93
        decay_factor = 0.99
        breakpoint =350
        p_factors = np.linspace(0, 0.99, 20)
        results = []
        for prolif_prob in p_factors:
            for run in range(num_runs):
                print(f'Running simulation for prolif_prob {prolif_prob}, run {run + 1}...')
                vessel_grid, tumor_grid, final_density, _, _ = simulate_CA(
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
                    p=prolif_prob,
                    plot_steps = 5, 
                    midpoint_sigmoid=1, 
                    steepness=1)
                
                vessel_image(vessel_grid, 'final_grid.png')
                image = io.imread('images/final_grid.png')
                average_degree, average_betweenness, average_page_rank, average_cc, _ = network_analysis(image, tumor_grid, show_skeleton=False, show_graph=False, print_results=False)
                results.append({
                    'run': run + 1,
                    'prolif_prob': prolif_prob,
                    'final_density': final_density,
                    'average_degree': average_degree,
                    'average_betweenness': average_betweenness,
                    'average_page_rank': average_page_rank,
                    'average_clustering_coefficient': average_cc
                })
                print()
        
        df = pd.DataFrame(results)
        df.to_csv(f'data/{experiment_type}_results_1.csv', index=False)
    
    elif experiment_type == 'midpoint_sigmoid':
        bias_factor = 0.93
        decay_factor = 0.99
        breakpoint =350
        midpoints = np.linspace(0.5, 5, 20)
        results = []
        for midpoint in midpoints:
            for run in range(num_runs):
                print(f'Running simulation for midpoint {midpoint}, run {run + 1}...')
                vessel_grid, tumor_grid, final_density, _, _ = simulate_CA(
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
                    p=0.1,
                    plot_steps=1,
                    midpoint_sigmoid=midpoint,
                    steepness=1)
                
                vessel_image(vessel_grid, 'final_grid.png')
                image = io.imread('images/final_grid.png')
                average_degree, average_betweenness, average_page_rank, average_cc, _ = network_analysis(image, tumor_grid, show_skeleton=False, show_graph=False, print_results=False)
                results.append({
                    'run': run + 1,
                    'midpoint_sigmoid': midpoint,
                    'final_density': final_density,
                    'average_degree': average_degree,
                    'average_betweenness': average_betweenness,
                    'average_page_rank': average_page_rank,
                    'average_clustering_coefficient': average_cc
                })
                print()
        
        df = pd.DataFrame(results)
        df.to_csv(f'data/{experiment_type}_results_1.csv', index=False)

run_experiments()