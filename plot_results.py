"""
This file contains functions to plot the results of the experiments, the data for which can be found
in the data folder. 
The data was generated using the run_experiments function in the network_analysis.py file.
Simply run the script and input the experiment type to see the plots.
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.stats as stats

def plot_network_results(experiment_type):
    """
    Plots the average of the network statistics with a 95% confidence interval for the given experiment type.
    The control parameter is on the x-axis and the network statistic is on the y-axis.
    Input:
    experiment_type: str, the type of experiment to plot (bias_factor, prolif_prob, midpoint_sigmoid)
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
        plt.suptitle("Bias Factor vs. Network Metrics")
        axes[2].set_xlabel("Bias Factor")
    elif experiment_type == "prolif_prob":
        plt.suptitle("Proliferation Probability vs. Network Metrics")
        axes[0].set_xlabel("Proliferation Probability")
        axes[1].set_xlabel("Proliferation Probability")
        axes[2].set_xlabel("Proliferation Probability")
    elif experiment_type == "midpoint_sigmoid":
        plt.suptitle("Sigmoid Midpoint vs. Network Metrics")
        axes[0].set_xlabel("Sigmoid Midpoint")
        axes[1].set_xlabel("Sigmoid Midpoint")
        axes[2].set_xlabel("Sigmoid Midpoint")
    plt.tight_layout()
    plt.show()

def plot_scatter(experiment_type):
    """
    Plots the network metrics against the final tumor density for the given experiment type.
    The data points are colored by the control parameter.
    Input:
    experiment_type: str, the type of experiment to plot (bias_factor, prolif_prob, midpoint_sigmoid)
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
    plt.show()

experiment_type = input("Enter the experiment type (bias_factor, prolif_prob, midpoint_sigmoid, steepness): ")
if experiment_type == "bias_factor":
    plot_network_results("bias_factor")
    plot_scatter("bias_factor")
elif experiment_type == "prolif_prob":
    plot_network_results("prolif_prob")
    plot_scatter("prolif_prob")
elif experiment_type == "midpoint_sigmoid":
    plot_network_results("midpoint_sigmoid")
    plot_scatter("midpoint_sigmoid")
