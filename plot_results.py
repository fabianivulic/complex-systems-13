import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import powerlaw

def plot_network_results(experiment_type):
    metrics = ["average_degree", "average_clustering_coefficient"]
    df = pd.read_csv(f"data/{experiment_type}_results.csv")
    grouped_mean = df.groupby(experiment_type).mean().reset_index()
    grouped_std = df.groupby(experiment_type).std().reset_index()
    _, axes = plt.subplots(1, 3, figsize=(15, 5))
    axes = axes.flatten()

    for i, metric in enumerate(metrics):
        ax = axes[i]
        x = grouped_mean[experiment_type]
        y = grouped_mean[metric]
        yerr = grouped_std[metric]
        ax.plot(x, y, marker='o', label=f"Mean {metric.replace('_', ' ').title()}")
        ax.fill_between(x, y - yerr, y + yerr, alpha=0.3, label="Standard Deviation")
        ax.set_xlabel(f"{experiment_type.replace('_', ' ').title()}")
        ax.grid(False)
        ax.set_ylabel(metric.replace('_', ' ').title())
    
    axes[2].plot(x, grouped_mean["final_density"], marker='o', label="Mean Final Density")
    axes[2].fill_between(x, grouped_mean["final_density"] - grouped_std["final_density"], grouped_mean["final_density"] + grouped_std["final_density"], alpha=0.3)
    axes[2].set_ylabel("Final Tumor Entropy")
    
    if experiment_type == "bias_factor":
        plt.suptitle("Bias Factor vs. Network Metrics")
        axes[2].set_xlabel("Bias Factor")
    elif experiment_type == "prolif_prob":
        plt.suptitle("Proliferation Probability vs. Network Metrics")
        axes[0].set_xlabel("Proliferation Probability")
        axes[1].set_xlabel("Proliferation Probability")
        axes[2].set_xlabel("Proliferation Probability")
    plt.tight_layout()
    plt.show()

def plot_distributions():
        df = pd.read_csv(f"data/constant_results.csv")
        metrics = ["average_degree", "average_clustering_coefficient"]
        _, axes = plt.subplots(2, 2, figsize=(12, 7.5))
        axes = axes.flatten()
        for i, metric in enumerate(metrics):
            ax = axes[i]
            ax.hist(
                df[metric],
                bins=30,
                color='blue',
                alpha=0.7
            )
            ax.set_xlabel(metric.replace('_', ' ').title())
            ax.set_ylabel("Frequency")
            ax.grid(False)

        plt.tight_layout()
        plt.show()
        
        for metric in metrics:
            fit = powerlaw.Fit(df[metric])
            plt.figure(figsize=(8, 6))
            fit.plot_pdf(color='blue', linewidth=0, marker='o')
            fit.power_law.plot_pdf(color='red', linestyle='--', label=f'Power-Law Fit ($\\alpha={fit.alpha:.2f}$)')
            print(f"Estimated scaling exponent (alpha): {fit.alpha:.2f}")
            plt.xlabel(metric.replace('_', ' ').title())
            plt.ylabel("Frequency")
            plt.title(f"Power-Law Fit to {metric.replace('_', ' ').title()} Distribution")
            plt.show()
            
            #Compare distributions
            R1, p1 = fit.distribution_compare('power_law', 'exponential')
            print(f"R1: {R1}, p1: {p1}")
            R2, p2 = fit.distribution_compare('power_law', 'lognormal')
            print(f"R2: {R2}, p2: {p2}")
            print()

def plot_scatter(experiment_type):
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
    plt.suptitle("Final Tumor Entropy vs. Network Metrics")
    plt.tight_layout()
    plt.show()

experiment_type = input("Enter the experiment type (bias_factor or prolif_prob): ")
if experiment_type == "bias_factor":
    plot_network_results("bias_factor")
    plot_scatter("bias_factor")
elif experiment_type == "prolif_prob":
    plot_network_results("prolif_prob")
    plot_scatter("prolif_prob")