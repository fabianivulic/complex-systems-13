"""
This file contains the code for the network analysis of the simulated images.  
The network analysis measures that are calculated are the average degree, average degree betweenness, 
average page rank, and average clustering coefficient. 
"""

import numpy as np
import pandas as pd
from skimage import io, color, filters, morphology
import matplotlib.pyplot as plt
from skan import draw, Skeleton
from skan.csr import skeleton_to_nx
import networkx as nx
from growthdeath_jit import simulate_CA, vessel_image

def network_analysis(image, show_skeleton=True, show_graph=True, print_results=True):
    # For some reason, the image has 4 channels, so we only keep the first 3 (RGB)
    image = image[:, :, :3]
    image = color.rgb2gray(image)
    threshold = filters.threshold_otsu(image)
    binary_image = image > threshold # Image binarization. Not sure why the original image isn't already binary...

    skeleton = morphology.skeletonize(binary_image)

    if show_skeleton:
        fig, ax = plt.subplots()
        draw.overlay_skeleton_2d(image, skeleton, dilate=1, axes=ax)
        plt.title("Skeletonized Image")
        plt.show()

    #Create SKAN skeleton object
    skan_skeleton = Skeleton(skeleton)
    multi_graph = skeleton_to_nx(skan_skeleton)

    # The multigraph has to be converted to a normal graph for networkx to work
    graph = nx.Graph()
    for u, v, data in multi_graph.edges(data=True):
        if u != v:  # Remove self-loops from the graph
            if not graph.has_edge(u, v):
                graph.add_edge(u, v, **data)
    pos = {node: (coord[1], coord[0]) for node, coord in enumerate(skan_skeleton.coordinates)}
    if show_graph:
        nx.draw(graph, pos, node_size=10, font_size=8, edge_color='blue', node_color='red', with_labels=False)
        plt.title("Graph Representation of Skeleton")
        plt.show()

    #Calculate network analysis measures
    average_degree = np.mean(list(dict(graph.degree()).values()))
    average_betweenness = np.mean(list(nx.betweenness_centrality(graph).values()))
    average_page_rank = np.mean(list(nx.pagerank(graph).values()))
    average_clustering_coefficient = nx.average_clustering(graph)
    degree_distribution = np.array(list(dict(graph.degree()).values()))
    
    if print_results:
        print(f"Average degree: {average_degree}")
        print(f"Average degree betweenness: {average_betweenness}")
        print(f"Average page rank: {average_page_rank}")
        print(f"Average clustering coefficient: {average_clustering_coefficient}")
    
    return average_degree, average_betweenness, average_page_rank, average_clustering_coefficient, degree_distribution

def run_experiments():
    experiment_type = input('Enter the experiment type (bias_factor, decay_factor, constant): ')
    num_runs = int(input('Enter the number of runs for each experimental value: '))
    
    if experiment_type == 'bias_factor':
        bias_factors = np.linspace(0, 1.0, 5)
        decay_factor = 0.99
        results = []
        for bias_factor in bias_factors:
            for run in range(num_runs):
                print(f'Running simulation for bias factor {bias_factor}, run {run + 1}...')
                grid, min_entropy = simulate_CA(
                    size=200, 
                    num_seeds=20, 
                    steps=500, 
                    bias_factor=bias_factor, 
                    decay_factor=decay_factor, 
                    neighborhood_radius=10, 
                    wrap_around=False,
                    plot=False)
                
                vessel_image(grid, 'final_grid.png')
                image = io.imread('images/final_grid.png')
                average_degree, average_betweenness, average_page_rank, average_cc, degree_dist = network_analysis(image, show_skeleton=False, show_graph=False, print_results=False)
                results.append({
                    'bias_factor': bias_factor,
                    'run': run + 1,
                    'min_entropy': min_entropy,
                    'average_degree': average_degree,
                    'average_betweenness': average_betweenness,
                    'average_page_rank': average_page_rank,
                    'average_clustering_coefficient': average_cc
                })
                print()
        
        df = pd.DataFrame(results)
        df.to_csv(f'data/{experiment_type}_results.csv', index=False)
    
    elif experiment_type == 'decay_factor':
        bias_factor = 0.9
        decay_factors = np.linspace(0.8, 1.0, 5)
        results = []
        for decay_factor in decay_factors:
            for run in range(num_runs):
                print(f'Running simulation for decay factor {decay_factor}, run {run + 1}...')
                grid, min_entropy = simulate_CA(
                    size=200, 
                    num_seeds=20, 
                    steps=500, 
                    bias_factor=bias_factor, 
                    decay_factor=decay_factor, 
                    neighborhood_radius=10, 
                    wrap_around=False, 
                    plot=False)
                
                vessel_image(grid, 'final_grid.png')
                image = io.imread('images/final_grid.png')
                average_degree, average_betweenness, average_page_rank, average_cc, degree_dist = network_analysis(image, show_skeleton=False, show_graph=False, print_results=False)
                results.append({
                    'decay_factor': decay_factor,
                    'run': run + 1,
                    'min_entropy': min_entropy,
                    'average_degree': average_degree,
                    'average_betweenness': average_betweenness,
                    'average_page_rank': average_page_rank,
                    'average_clustering_coefficient': average_cc
                })
                print()
        
        df = pd.DataFrame(results)
        df.to_csv(f'data/{experiment_type}_results.csv', index=False)
    
    elif experiment_type == 'constant':
        bias_factor = 0.9
        decay_factor = 0.99
        results = []
        for run in range(num_runs):
            print(f'Running simulation for constant factors, run {run + 1}...')
            grid, min_entropy = simulate_CA(
                size=200, 
                num_seeds=20, 
                steps=500, 
                bias_factor=bias_factor, 
                decay_factor=decay_factor, 
                neighborhood_radius=10, 
                wrap_around=False,
                plot=False)
            
            vessel_image(grid, 'final_grid.png')
            image = io.imread('images/final_grid.png')
            average_degree, average_betweenness, average_page_rank, average_cc, degree_dist = network_analysis(image, show_skeleton=False, show_graph=False, print_results=False)
            results.append({
                'run': run + 1,
                'min_entropy': min_entropy,
                'average_degree': average_degree,
                'average_betweenness': average_betweenness,
                'average_page_rank': average_page_rank,
                'average_clustering_coefficient': average_cc
            })
            print()
        
        df = pd.DataFrame(results)
        df.to_csv(f'data/{experiment_type}_results.csv', index=False)
        
    else:
        print('Invalid experiment type. Please enter either "bias_factor" or "decay_factor".')

run_experiments()