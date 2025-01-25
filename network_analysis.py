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
    
    if print_results:
        print(f"Average degree: {average_degree}")
        print(f"Average degree betweenness: {average_betweenness}")
        print(f"Average page rank: {average_page_rank}")
        print(f"Average clustering coefficient: {average_clustering_coefficient}")
    
    return average_degree, average_betweenness, average_page_rank, average_clustering_coefficient

def run_experiments():
    size = 200
    num_seeds = 20
    steps = 500
    bias_factor = 0.93
    decay_factor = 0.99
    neighborhood_radius = 10
    wrap_around = False
    num_simulations = 3

    results = []
    for i in range(num_simulations):
        print(f'Running simulation {i+1}...')
        grid, final_entropy = simulate_CA(
            size=size, 
            num_seeds=num_seeds, 
            steps=steps, 
            bias_factor=bias_factor, 
            decay_factor=decay_factor, 
            neighborhood_radius=neighborhood_radius, 
            wrap_around=wrap_around)
        
        vessel_image(grid, 'final_grid.png')
        image = io.imread('images/final_grid.png')
        average_degree, average_betweenness, average_page_rank, average_cc = network_analysis(image, show_skeleton=False, show_graph=False, print_results=False)
        results.append({
            'final_entropy': final_entropy,
            'average_degree': average_degree,
            'average_betweenness': average_betweenness,
            'average_page_rank': average_page_rank,
            'average_clustering_coefficient': average_cc
        })
        print()

    df = pd.DataFrame(results)
    print(df)
    df.to_csv('data/results.csv', index=False)

run_experiments()