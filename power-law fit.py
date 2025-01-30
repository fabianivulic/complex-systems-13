import powerlaw
import numpy as np
import matplotlib.pyplot as plt
from growthdeath_jit import simulate_CA

def analyze_power_law(cluster_sizes, plot = False):
    """
    Analyze whether the cluster sizes follow a power law or a truncated power law.
    Input:
    - cluster_sizes: The sizes of the tumor/vessel clusters
    - plot: A boolean to enable plotting

    Output:
    - results: A dictionary containing the power-law and truncated power-law test results
    """
    results = {}

    # Fit a power law distribution to the data
    fit = powerlaw.Fit(cluster_sizes, discrete=True)
    results["alpha"] = fit.power_law.alpha
    results["xmin"] = fit.power_law.xmin

   # Compare power law to other distributions
    R_exp, p_exp = fit.distribution_compare('power_law', 'exponential', normalized_ratio=True)
    R_log, p_log = fit.distribution_compare('power_law', 'lognormal', normalized_ratio=True)
    R_tpl, p_tpl = fit.distribution_compare('power_law', 'truncated_power_law', normalized_ratio=True)
    
    results["power_law_comparisons"] = {
        "vs_exponential": (R_exp, p_exp),
        "vs_lognormal": (R_log, p_log),
        "vs_truncated_power_law": (R_tpl, p_tpl)
    }
    
    # Check if power law is a good fit
    results["is_power_law"] = all(p < 0.05 for p in [p_exp, p_log, p_tpl]) and all(R > 0 for R in [R_exp, R_log, R_tpl])
    
    # Compare truncated power law to other distributions
    R_tpl_exp, p_tpl_exp = fit.distribution_compare('truncated_power_law', 'exponential', normalized_ratio=True)
    R_tpl_log, p_tpl_log = fit.distribution_compare('truncated_power_law', 'lognormal', normalized_ratio=True)
    R_tpl_pl, p_tpl_pl = fit.distribution_compare('truncated_power_law', 'power_law', normalized_ratio=True)
    
    results["truncated_power_law_comparisons"] = {
        "vs_exponential": (R_tpl_exp, p_tpl_exp),
        "vs_lognormal": (R_tpl_log, p_tpl_log),
        "vs_power_law": (R_tpl_pl, p_tpl_pl)
    }
    
    # Check if truncated power law is a better fit
    results["is_truncated_power_law"] = all(p < 0.05 for p in [p_tpl_exp, p_tpl_log, p_tpl_pl]) and all(R > 0 for R in [R_tpl_exp, R_tpl_log, R_tpl_pl])
    
    # Print results
    print(f"Estimated alpha: {results['alpha']}")
    print(f"Xmin: {results['xmin']}")
    print(f"Power law comparisons: {results['power_law_comparisons']}")
    print(f"Is Power Law: {results['is_power_law']}")
    print(f"Truncated Power Law comparisons: {results['truncated_power_law_comparisons']}")
    print(f"Is Truncated Power Law: {results['is_truncated_power_law']}")

    if plot:
        plt.figure(figsize=(6, 4))
        ax = plt.gca()  # Get the current axis to plot everything on the same figure

        # Plot empirical CCDF as green dots
        fit.plot_ccdf(linewidth=0, marker='o', markersize=5, color='green', label='Data', ax=ax)

        # Plot fitted distributions
        fit.power_law.plot_ccdf(linewidth=2, linestyle='--', label='Power Law Fit', ax=ax)
        fit.truncated_power_law.plot_ccdf(linewidth=2, linestyle=':', label='Truncated Power Law Fit', ax=ax)

        # Add labels, title, and legend
        plt.xlabel("Cluster Size (log scale)")
        plt.ylabel("Probability Density (log scale)")
        plt.title("Log-Log Plot of Tumor Cluster Sizes with Power Law Fits")
        plt.legend()
        plt.tight_layout()
        plt.show()

    return results

def combine_datasets(num_datasets, bias_factor):
    """
    Combine multiple datasets into a single dataset.
    Input:
    - num_datasets: Number of datasets to combine
    """
    combined_dataset = []
    for i in range(num_datasets):
        _, _, _, cluster_sizes_tumor, cluster_sizes_vessel = simulate_CA(plot=False, bias_factor = bias_factor, tumor_prob=0.3, tumor_clusters=True, vessel_clusters=False)
        combined_dataset.extend(cluster_sizes_tumor[-1])
        print(f"Dataset {i+1} completed.")
    return combined_dataset

def simulations(num_simulations, bias_factor):
    """
    Run multiple simulations and analyze the power law and truncated power law distribution.
    """
    powerlaw_count = 0
    truncated_powerlaw_count = 0
    alphas = []
    x_mins = []
    all_results = []
    
    # Run multiple simulations and analyze the power law distribution
    for i in range(num_simulations):
        dataset = combine_datasets(10, bias_factor)
        print(f"Analyzing simulation {i+1}...")
        results = analyze_power_law(dataset, plot=True)
        
        if results["is_power_law"]:
            powerlaw_count += 1
        if results["is_truncated_power_law"]:
            truncated_powerlaw_count += 1
        
        alphas.append(results["alpha"])
        x_mins.append(results["xmin"])
        all_results.append(results)
        
    # Compute statistics
    proportion_power_law = (powerlaw_count / num_simulations) * 100
    proportion_truncated_power_law = (truncated_powerlaw_count / num_simulations) * 100
    mean_alpha = np.mean(alphas)
    std_alpha = np.std(alphas)
    mean_xmin = np.mean(x_mins)
    std_xmin = np.std(x_mins)

    print(f"Proportion of Power Law: {proportion_power_law:.2f}%")
    print(f"Proportion of Truncated Power Law: {proportion_truncated_power_law:.2f}%")

    # Return results as a dictionary
    return {
        "proportion_power_law": proportion_power_law,
        "proportion_truncated_power_law": proportion_truncated_power_law,
        "mean_alpha": mean_alpha,
        "std_alpha": std_alpha,
        "mean_xmin": mean_xmin,
        "std_xmin": std_xmin,
        "all_results": all_results,
    }

simulations(2, 0.93)

# Testing power-law for different bias values
test = input("Do you want to test the power-law fit for different bias values? (y/n): ")
if test.lower() == "y":
    min_value = float(input("Enter minimum bias factor: "))
    max_value = float(input("Enter maximum bias factor: "))
    num_values = int(input("Enter number of bias factor values: "))
    num_simulations = int(input("Enter number of simulations per parameter set: "))
    
    bias_factors = np.linspace(min_value, max_value, num_values)
    experiment_results = []

    for bias in bias_factors:
        print(f"Running simulation with bias factor: {bias}")
        results = simulations(num_simulations, bias_factor=bias)
        experiment_results.append({
            "bias": bias,
            "proportion_power_law": results["proportion_power_law"],
            "proportion_truncated_power_law": results["proportion_truncated_power_law"],
        })


    # Print results in table format
    print("\nResults Table:")
    print("{:<10} {:<25} {:<25}".format("Bias", "Proportion Power Law (%)", "Proportion Truncated Power Law (%)"))
    for res in experiment_results:
        print("{:<10.2f} {:<25.2f} {:<25.2f}".format(res["bias"], res["proportion_power_law"], res["proportion_truncated_power_law"]))
    
    # Plot results
    plt.figure(figsize=(8, 5))
    biases = [res["bias"] for res in experiment_results]
    proportions_power = [res["proportion_power_law"] for res in experiment_results]
    proportions_truncated = [res["proportion_truncated_power_law"] for res in experiment_results]
    
    plt.plot(biases, proportions_power, marker='o', linestyle='-', label='Power Law')
    plt.plot(biases, proportions_truncated, marker='s', linestyle='--', label='Truncated Power Law')
    
    plt.xlabel("Bias Factor")
    plt.ylabel("Proportion (%)")
    plt.title("Power Law and Truncated Power Law Proportions vs. Bias Factor")
    plt.legend()
    plt.grid()
    plt.show()
else:
    print("Ok.")


def combine_datasets(num_datasets):
    """
    Combine multiple datasets into a single dataset.
    Input:
    - num_datasets: Number of datasets to combine
    """
    combined_dataset_start = []
    combined_dataset_end = []
    for i in range(num_datasets):
        _, _, _, cluster_sizes_tumor, cluster_sizes_vessel = simulate_CA(plot = False, bias_factor=0.93, tumor_clusters=False, vessel_clusters=True)
        combined_dataset_start.extend(cluster_sizes_vessel[70])
        combined_dataset_end.extend(cluster_sizes_vessel[-1])
        print(f"Dataset {i+1} completed.")
    return combined_dataset_start, combined_dataset_end

combine_datasets_start, combine_datasets_end = combine_datasets(20)

analyze_power_law(combine_datasets_start, plot=True)
analyze_power_law(combine_datasets_end, plot=True)
