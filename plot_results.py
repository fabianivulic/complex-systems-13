"""
This file contains functions to plot the results of the experiments, the data for which can be found
in the data folder.
The data was generated using the run_experiments function in the network_analysis.py file.
Simply run the script and input the experiment type. The resulting plots will be saved in the plots folder.
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.stats as stats
import json

def plot_network_results(experiment_type):
    """
    ### Description:
    Plots the average of the network statistics with a 95% confidence interval for the given experiment type.
    The control parameter is on the x-axis and the network statistic is on the y-axis.
    ### Input:
    experiment_type: str, the type of experiment to plot (bias_factor, prolif_prob, midpoint_sigmoid)
    ### Output:
    No returned value, the plot is displayed.
    """
    metrics = ["average_degree", "average_clustering_coefficient"]
    df = pd.read_csv(f"data/{experiment_type}_results.csv")
    grouped_mean = df.groupby(experiment_type).mean().reset_index()
    grouped_sem = df.groupby(experiment_type).sem().reset_index()
    confidence = 0.95
    ci_multiplier = stats.t.ppf((1 + confidence) / 2., df.groupby(experiment_type).count().iloc[:, 0] - 1)
    grouped_ci = grouped_sem.mul(ci_multiplier, axis=0)

    _, axes = plt.subplots(1, 3, figsize=(15, 5))
    axes = axes.flatten()

    for i, metric in enumerate(metrics):
        ax = axes[i]
        x = grouped_mean[experiment_type]
        y = grouped_mean[metric]
        yerr = grouped_ci[metric]
        ax.plot(x, y, marker='o', label=f"Mean {metric.replace('_', ' ').title()}")
        ax.fill_between(x, y - yerr, y + yerr, alpha=0.3, label="Standard Deviation")
        ax.set_xlabel(f"{experiment_type.replace('_', ' ').title()}")
        ax.grid(False)
        ax.set_ylabel(metric.replace('_', ' ').title())

    axes[2].plot(x, grouped_mean["final_density"], marker='o', label="Mean Final Density")
    axes[2].fill_between(x, grouped_mean["final_density"] - grouped_ci["final_density"], grouped_mean["final_density"] + grouped_ci["final_density"], alpha=0.3)
    axes[2].set_ylabel("Final Tumor Density")

    if experiment_type == "bias_factor":
        plt.suptitle("Bias Factor vs. Network Metrics \n (95% Confidence Interval)")
        axes[2].set_xlabel("Bias Factor")
    elif experiment_type == "prolif_prob":
        plt.suptitle("Proliferation Probability vs. Network Metrics \n (95% Confidence Interval)")
        axes[0].set_xlabel("Proliferation Probability")
        axes[1].set_xlabel("Proliferation Probability")
        axes[2].set_xlabel("Proliferation Probability")
    elif experiment_type == "midpoint_sigmoid":
        plt.suptitle("Sigmoid Midpoint vs. Network Metrics \n (95% Confidence Interval)")
        axes[0].set_xlabel("Sigmoid Midpoint")
        axes[1].set_xlabel("Sigmoid Midpoint")
        axes[2].set_xlabel("Sigmoid Midpoint")
    plt.tight_layout()
    plt.savefig(f"plots/{experiment_type}_network_results.png")

def plot_scatter(experiment_type):
    """
    ### Description:
    Plots the network metrics against the final tumor density for the given experiment type.
    The data points are colored by the control parameter.
    ### Input:
    experiment_type: str, the type of experiment to plot (bias_factor, prolif_prob, midpoint_sigmoid)
    ### Output:
    No returned value, the plot is displayed.
    """
    df = pd.read_csv(f"data/{experiment_type}_results.csv")
    metrics = [
        "average_degree",
        "average_clustering_coefficient"]
    _, axes = plt.subplots(1, 2, figsize=(15, 5))
    axes = axes.flatten()
    for i, metric in enumerate(metrics):
        ax = axes[i]
        scatter = ax.scatter(
            df[metric],  # x-axis
            df["final_density"],  # y-axis
            c=df[experiment_type],  # color by factor
            label=metric.replace('_', ' ').title(),
            s=20
        )
        ax.set_xlabel(metric.replace('_', ' ').title())
        ax.grid(False)
    handles, labels = scatter.legend_elements(prop="colors", alpha=0.6)
    axes[0].set_ylabel("Final Tumor Density")
    for ax in axes[1:]:
        ax.set_yticks([])
    if experiment_type == "bias_factor":
        axes[1].legend(handles, labels, title="Bias Factor", bbox_to_anchor=(1.02, 1.02), loc='upper left')
    elif experiment_type == "prolif_prob":
        axes[1].legend(handles, labels, title="Proliferation P", bbox_to_anchor=(1.02, 1.02), loc='upper left')
    elif experiment_type == "midpoint_sigmoid":
        axes[1].legend(handles, labels, title="Sigmoid Midpoint", bbox_to_anchor=(1.02, 1.02), loc='upper left')
    plt.suptitle("Final Tumor Density vs. Network Metrics")
    plt.tight_layout()
    plt.savefig(f"plots/{experiment_type}_scatter_results.png")

def compute_mean_and_ci(data, confidence=0.95):
    """
    ### Description:
    Computes the mean and confidence interval for a given list of lists (data over multiple runs).
    ### Input:
    data: list of lists of data
    confidence: confidence level for the confidence interval
    ### Output:
    mean: mean of the data
    ci: confidence interval of the mean
    """
    data = np.array(data)
    mean = np.mean(data, axis=0)
    sem = stats.sem(data, axis=0, nan_policy='omit')  # Standard error of the mean
    ci = sem * stats.t.ppf((1 + confidence) / 2., len(data) - 1)  # Confidence interval
    return mean, ci

def plot_results_over_time():
    """
    ### Description:
    Extracts network statistics from the big_results list and plots the average over time with 
    confidence intervals.
    ### Input:
    big_results: list of lists of network statistics
    timesteps: list of timesteps
    ### Output:
    Returns nothing, but displays the plot.
    """
    # Load data from JSON
    with open("data/results_data_time.json", "r") as f:
        data = json.load(f)

    # Extract data
    results_over_time = data["results_multiple_runs"]
    timesteps = data["timesteps"]

    # Extracting network statistics from big_results (list of lists)
    avg_degrees = [[res['average_degree'] for res in run] for run in results_over_time]
    avg_betweennesses = [[res['average_betweenness'] for res in run] for run in results_over_time]
    avg_clustering_coefficients = [[res['average_clustering_coefficient'] for res in run] for run in results_over_time]

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
    axes[0].set_title("Average Degree over Time \n (95% Confidence Interval)")

    axes[1].plot(timesteps, mean_betweennesses, marker='o', label="Average Betweenness")
    axes[1].fill_between(timesteps, mean_betweennesses - ci_betweennesses, mean_betweennesses + ci_betweennesses, alpha=0.2)
    axes[1].set_xlabel("Timesteps")
    axes[1].set_ylabel("Average Betweenness")
    axes[1].set_title("Average Betweenness over Time \n (95% Confidence Interval)")

    axes[2].plot(timesteps, mean_clustering, marker='o', label="Average Clustering Coefficient")
    axes[2].fill_between(timesteps, mean_clustering - ci_clustering, mean_clustering + ci_clustering, alpha=0.2)
    axes[2].set_xlabel("Timesteps")
    axes[2].set_ylabel("Average Clustering Coefficient")
    axes[2].set_title("Average Clustering Coefficient over Time \n (95% Confidence Interval)")

    plt.tight_layout()
    plt.savefig("plots/network_metrics_over_time.png")

experiment_type = input("Enter the experiment type (bias_factor, prolif_prob, midpoint_sigmoid, networks_over_time): ")

if experiment_type == "bias_factor":
    plot_network_results("bias_factor")
    plot_scatter("bias_factor")
elif experiment_type == "prolif_prob":
    plot_network_results("prolif_prob")
    plot_scatter("prolif_prob")
elif experiment_type == "midpoint_sigmoid":
    plot_network_results("midpoint_sigmoid")
    plot_scatter("midpoint_sigmoid")
elif experiment_type == "networks_over_time":
    plot_results_over_time()
