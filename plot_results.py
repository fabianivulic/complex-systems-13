import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import powerlaw

# Load the data
def plot_results():
    experiment_type = input('Enter the experiment type (bias_factor, decay_factor, constant, distributions, scatter): ')
    metrics = [
        "average_degree",
        "average_betweenness",
        "average_page_rank",
        "average_clustering_coefficient"]

    if experiment_type not in ['constant', 'distributions', 'scatter']:
        df = pd.read_csv(f"data/{experiment_type}_results.csv")
        grouped_mean = df.groupby(experiment_type).mean().reset_index()
        grouped_std = df.groupby(experiment_type).std().reset_index()
        _, axes = plt.subplots(2, 2, figsize=(12, 7.5))
        axes = axes.flatten()

        for i, metric in enumerate(metrics):
            ax = axes[i]
            x = grouped_mean[experiment_type]
            y = grouped_mean[metric]
            yerr = grouped_std[metric]
            ax.plot(x, y, marker='o', label=f"Mean {metric.replace('_', ' ').title()}")
            ax.fill_between(x, y - yerr, y + yerr, alpha=0.3, label="Standard Deviation")
            ax.set_xlabel(f"{experiment_type.replace('_', ' ').title()}")
            ax.set_ylabel(metric.replace('_', ' ').title())
            ax.grid(False)
        
        plt.tight_layout()
        plt.show()

    elif experiment_type == 'constant':
        df = pd.read_csv(f"data/{experiment_type}_results.csv")
        _, axes = plt.subplots(2, 2, figsize=(12, 7.5))
        axes = axes.flatten()
        for i, metric in enumerate(metrics):
            ax = axes[i]
            ax.scatter(
                df[metric],  # x-axis
                df["min_entropy"],  # y-axis
                color='blue',
                label=metric.replace('_', ' ').title(),
                s=20
            )
            ax.set_xlabel(metric.replace('_', ' ').title())
            ax.set_ylabel("Min Tumor Entropy")
            ax.grid(False)

        #plt.suptitle("Min Tumor Entropy vs. Network Metrics")
        plt.tight_layout()
        plt.show()
    
    elif experiment_type == 'distributions':
        df = pd.read_csv(f"data/constant_results.csv")
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
            
    elif experiment_type == 'scatter':
        new_experiment_type = input('Enter the experiment type (bias_factor, decay_factor, tumor_prob): ')
        df = pd.read_csv(f"data/{new_experiment_type}_results.csv")
        # combined_df = pd.concat(
        #     [pd.read_csv(f"data/{factor}_results.csv").assign(factor=factor) for factor in ['bias_factor', 'decay_factor', 'tumor_prob']],
        #     ignore_index=True
        # )
        _, axes = plt.subplots(2, 2, figsize=(12, 7.5))
        axes = axes.flatten()
        for i, metric in enumerate(metrics):
            ax = axes[i]
            scatter = ax.scatter(
                df[metric],  # x-axis
                df["min_entropy"],  # y-axis
                c=df[new_experiment_type],  # color by factor
                label=metric.replace('_', ' ').title(),
                s=20
            )
            ax.set_xlabel(metric.replace('_', ' ').title())
            ax.set_ylabel("Min Tumor Entropy")
            ax.grid(False)
        handles, labels = scatter.legend_elements(prop="colors", alpha=0.6)
        axes[1].legend(handles, labels, title=f"{new_experiment_type}", bbox_to_anchor=(1.02, 1.025), loc='upper left')
        plt.suptitle("Min Tumor Entropy vs. Network Metrics")
        plt.tight_layout()
        plt.show()
    
plot_results()