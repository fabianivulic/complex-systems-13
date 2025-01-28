import powerlaw
import numpy as np
import matplotlib.pyplot as plt
from growthdeath_jit import *

def analyze_power_law(cluster_sizes, plot = True):
    """
    Analyze the power law distribution of cluster sizes.
    Input:
    - cluster_sizes: The sizes of the tumor clusters
    - plot: A boolean to enable plotting

    Output:
    - Dict, containing estimated parameters and goodness of fit metrics
    """
    results = {}

    # Fit a power law distribution to the data
    fit = powerlaw.Fit(cluster_sizes, discrete=True)
    results["alpha"] = fit.power_law.alpha
    results["xmin"] = fit.power_law.xmin

    # Compute goodness of fit metrics
    R, p_value = fit.distribution_compare('power_law', 'exponential', normalized_ratio=True)
    results["R"] = R
    results["p_value"] = p_value
    results["is_power_law"] = p_value > 0.05

    # Plot the power law distribution
    if plot:
        plt.figure(figsize=(8, 6))
        
        # Plot empirical PDF (data points)
        fit.plot_pdf(marker='o', label="Empirical Data (PDF)", linestyle='None')
        
        # Plot fitted power-law curve
        fit.power_law.plot_pdf(color='red', linestyle='--', linewidth=2, label=f"Power Law Fit (α={fit.power_law.alpha:.2f})")
        
        # Add labels, title, and legend
        plt.xlabel("Cluster Size (log scale)")
        plt.ylabel("Probability Density (log scale)")
        plt.title("Log-Log Plot of Tumor Cluster Sizes with Power Law Fit")
        plt.legend()
        plt.tight_layout()
        plt.show()

    return results

# Run simulations and analyze the power law distribution
def simulations(num_simulations, **params):
    """
    Run multiple simulations with different parameters.
    Input:
    - num_simulations: Number of simulations to run
    - params: Additional parameters for the simulation
    """
    powerlaw_count = 0
    alphas = []
    x_mins = []
    all_results = []

    for i in range(num_simulations):
        _, _, _, cluster_sizes_over_time = simulate_CA(plot = False, **params)
        print(f"Analyzing simulation {i+1}...")
        cluster_size = cluster_sizes_over_time[-1]
        results = analyze_power_law(cluster_size, plot=False)
        
        if results["is_power_law"]:
            powerlaw_count += 1
        
        alphas.append(results["alpha"])
        x_mins.append(results["xmin"])
        all_results.append(results)
        
    # Compute statistics
    proportion_power_law = powerlaw_count / num_simulations * 100
    mean_alpha = np.mean(alphas)
    std_alpha = np.std(alphas)
    mean_xmin = np.mean(x_mins)
    std_xmin = np.std(x_mins)

    # Return results as a dictionary
    return {
        "proportion_power_law": proportion_power_law,
        "mean_alpha": mean_alpha,
        "std_alpha": std_alpha,
        "mean_xmin": mean_xmin,
        "std_xmin": std_xmin,
        "all_results": all_results,
    }

factor_simulate = str(input("Which factor to run simulations on (bias_factor, tumor_prob): "))
if factor_simulate == "bias_factor":
    # Testing power-law for different bias values
    bias_factor = np.linspace(0.1, 1.0, 10)
    num_simulations = int(input("Enter number of simulations per parameter set: "))
    experiment_results = []
    for bias in bias_factor:
        print(f"Running simulation with bias factor: {bias}")
        results = simulations(num_simulations, bias_factor=bias)
        experiment_results.append({
            "bias": bias,
            "proportion_power_law": results["proportion_power_law"],
            "mean_alpha": results["mean_alpha"],
            "std_alpha": results["std_alpha"],
            "mean_xmin": results["mean_xmin"],
            "std_xmin": results["std_xmin"],
        })

    # Display results in a table format
    print("\nResults Table:")
    print("{:<10} {:<20} {:<20} {:<20} {:<20}".format(
        "Bias", "Proportion Power Law (%)", "Mean Alpha", "Std Alpha", "Mean x_min", "Std x_min"
    ))
    for res in experiment_results:
        print("{:<10.2f} {:<20.2f} {:<20.2f} {:<20.2f} {:<20.2f} {:<20.2f}".format(
            res["bias"], res["proportion_power_law"], res["mean_alpha"], res["std_alpha"], res["mean_xmin"], res["std_xmin"]
        ))

elif factor_simulate == "tumor_prob":
    # Testing power-law for different tumor prob
    tumor_prob = np.linspace(0.1, 1.0, 10)
    num_simulations = int(input("Enter number of simulations per parameter set: "))
    experiment_results = []
    for prob in tumor_prob:
        print(f"Running simulation with tumor probability: {prob}")
        results = simulations(num_simulations, tumor_prob=prob)
        experiment_results.append({
            "tumor_prob": prob,
            "proportion_power_law": results["proportion_power_law"],
            "mean_alpha": results["mean_alpha"],
            "std_alpha": results["std_alpha"],
            "mean_xmin": results["mean_xmin"],
            "std_xmin": results["std_xmin"],
        })

    # Display results in a table format
    print("\nResults Table:")
    print("{:<10} {:<20} {:<20} {:<20} {:<20}".format(
        "Tumor_prob", "Proportion Power Law (%)", "Mean Alpha", "Std Alpha", "Mean x_min", "Std x_min"
    ))
    for res in experiment_results:
        print("{:<10.2f} {:<20.2f} {:<20.2f} {:<20.2f} {:<20.2f} {:<20.2f}".format(
            res["tumor_prob"], res["proportion_power_law"], res["mean_alpha"], res["std_alpha"], res["mean_xmin"], res["std_xmin"]
        ))
