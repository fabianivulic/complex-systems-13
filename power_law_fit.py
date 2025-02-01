"""
This script analyzes the power-law distribution of tumor and vessel cluster sizes. It includes functions to simulate cluster growth, combine datasets,
and analyze the resulting cluster sizes for power-law and truncated power-law fits.
"""
import powerlaw
import numpy as np
import matplotlib.pyplot as plt
from tumor import simulate_CA

def analyze_power_law(cluster_sizes, plot=False):
    """
    ### Description:
    Analyze whether the cluster sizes follow a power law or a truncated power law.
    ### Input:
    - cluster_sizes: The sizes of the tumor/vessel clusters
    - plot: A boolean to enable plotting
    ### Output:
    - results: A dictionary containing the power-law and truncated power-law test results
    """
    results = {}

    fit = powerlaw.Fit(cluster_sizes, discrete=True)
    results["alpha"] = fit.power_law.alpha
    results["xmin"] = fit.power_law.xmin
    if fit.power_law.xmin <= 0:
        print(f"Warning: Invalid xmin ({fit.power_law.xmin}), setting to minimum valid value.")
        fit.power_law.xmin = max(1, fit.power_law.xmin)  # Ensure xmin is at least 1

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

    # Check if truncated power law is a good fit
    results["is_truncated_power_law"] = all(p < 0.05 for p in [p_tpl_exp, p_tpl_log, p_tpl_pl]) and all(R > 0 for R in [R_tpl_exp, R_tpl_log, R_tpl_pl])

    print(f"Estimated alpha: {results['alpha']}")
    print(f"Xmin: {results['xmin']}")
    print(f"Power law comparisons: {results['power_law_comparisons']}")
    print(f"Is Power Law: {results['is_power_law']}")
    print(f"Truncated Power Law comparisons: {results['truncated_power_law_comparisons']}")
    print(f"Is Truncated Power Law: {results['is_truncated_power_law']}")

    if plot:
        plt.figure(figsize=(6, 4))
        ax = plt.gca()

        # Plot empirical CCDF as green dots
        fit.plot_ccdf(linewidth=0, marker='o', markersize=5, color='green', label='Data', ax=ax)

        # Plot fitted distributions
        fit.power_law.plot_ccdf(linewidth=2, linestyle='--', label='Power Law Fit', ax=ax)
        fit.truncated_power_law.plot_ccdf(linewidth=2, linestyle=':', label='Truncated Power Law Fit', ax=ax)

        plt.xlabel("Cluster Size (log scale)")
        plt.ylabel("Probability Density (log scale)")
        plt.title("Log-Log Plot of Cluster Sizes with Power Law Fits")
        plt.legend()
        plt.tight_layout()
        plt.show()

    return results

def combine_datasets(num_datasets, bias_factor, tumor=True):
    """
    ### Description:
    Combine multiple datasets of cluster sizes at the last time step into one.
    ### Input:
    - num_datasets (int): Number of datasets to combine
    - bias_factor (float): Bias factor for the simulation, influencing cluster growth dynamics.
    - tumor (bool): If True, analyze tumor cluster sizes; if False, analyze vessel cluster sizes.
    """
    combined_dataset = []
    for i in range(num_datasets):
        if tumor:
            _, _, _, cluster_sizes_tumor, _ = simulate_CA(plot=False, bias_factor = bias_factor, tumor_prob=0.3, tumor_clusters=True, vessel_clusters=False)
            combined_dataset.extend(cluster_sizes_tumor[-1])
            print(f"Dataset {i+1} completed.")
        else:
            _, _, _, _, cluster_sizes_vessel = simulate_CA(plot=False, bias_factor = bias_factor, tumor_prob=0.3, tumor_clusters=False, vessel_clusters=True)
            combined_dataset.extend(cluster_sizes_vessel[-1])
            print(f"Dataset {i+1} completed.")
    return combined_dataset

def simulations(num_simulations, bias_factor, tumor = True):
    """
    ### Description:
    Run multiple simulations and analyze whether the resulting cluster sizes follow a power law or truncated power law distribution.
    ### Input:
    - num_simulations (int): Number of simulations to perform.
    - bias_factor (float): Bias factor for the simulation, influencing cluster growth dynamics.
    - tumor (bool): If True, analyze tumor cluster sizes; if False, analyze vessel cluster sizes.
    ### Output:
    - dict: A dictionary containing the proportion of simulations that fit a power law or truncated power law,
            as well as statistical information about estimated parameters (alpha, xmin).
    """
    assert num_simulations > 0, "num_simulations must be greater than zero"
    assert 0 < bias_factor < 1, "bias_factor must be between 0 and 1"

    powerlaw_count = 0
    truncated_powerlaw_count = 0
    alphas = []
    x_mins = []
    all_results = []

    # Run multiple simulations and analyze the power law distribution
    for i in range(num_simulations):
        dataset = combine_datasets(10, bias_factor, tumor = tumor)
        print(f"Analyzing simulation {i+1}/{num_simulations}...")
        results = analyze_power_law(dataset, plot=False)

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

    # Return summarized results in a dictionary
    return {
        "proportion_power_law": proportion_power_law,
        "proportion_truncated_power_law": proportion_truncated_power_law,
        "mean_alpha": mean_alpha,
        "std_alpha": std_alpha,
        "mean_xmin": mean_xmin,
        "std_xmin": std_xmin,
        "all_results": all_results,
    }

def test_bias_values():
    """
    Function to test truncated power-law fit for different bias values.
    """
    min_value = float(input("Enter minimum bias factor: "))
    max_value = float(input("Enter maximum bias factor: "))
    num_values = int(input("Enter number of bias factor values: "))
    tumor = input("Analyze tumor (t) or vessel (v) clusters? ")
    num_simulations = int(input("Enter number of simulations per parameter set: "))

    bias_factors = np.linspace(min_value, max_value, num_values)
    experiment_results = []

    for bias in bias_factors:
        print(f"Running simulation with bias factor: {bias}")
        if tumor == "t":
            results = simulations(num_simulations, bias_factor=bias, tumor = True)
        else:
            results = simulations(num_simulations, bias_factor=bias, tumor = False)
        experiment_results.append({
            "bias": bias,
            "proportion_truncated_power_law": results["proportion_truncated_power_law"],
        })

    print("\nResults Table:")
    print("{:<10} {:<30}".format("Bias", "Proportion Truncated Power Law (%)"))
    for res in experiment_results:
        print("{:<10.2f} {:<30.2f}".format(res["bias"], res["proportion_truncated_power_law"]))

    plt.figure(figsize=(8, 5))
    biases = [res["bias"] for res in experiment_results]
    proportions_truncated = [res["proportion_truncated_power_law"] for res in experiment_results]
    plt.plot(biases, proportions_truncated, marker='s', linestyle='--', label='Truncated Power Law')
    plt.xlabel("Bias Factor")
    plt.ylabel("Proportion (%)")
    plt.title("Truncated Power Law Proportions vs. Bias Factor")
    plt.legend()
    plt.grid()
    plt.show()

def main():
    """Main function to manage user interaction and start simulations."""
    test = input("Do you want to test the power-law fit for different bias values? (y/n): ")
    if test.lower() == "y":
        test_bias_values()
    else:
        print("Ok.")

if __name__ == "__main__":
    main()
