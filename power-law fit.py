import powerlaw
import numpy as np
import matplotlib.pyplot as plt
from growthdeath_jit import simulate_CA
from collections import Counter

def analyze_power_law(cluster_sizes, plot = False):
    """
    Analyze the power law distribution of cluster sizes.
    Input:
    - cluster_sizes: The sizes of the tumor clusters
    - plot: A boolean to enable plotting

    Output:
    - results: A dictionary containing the power law fit results
    """
    results = {}

    # Fit a power law distribution to the data
    fit = powerlaw.Fit(cluster_sizes, discrete=True)
    results["alpha"] = fit.power_law.alpha
    results["xmin"] = fit.power_law.xmin

    R_list, p_value_list = [], []
    # Compute goodness of fit metrics
    R_exp, p_value_exp = fit.distribution_compare('power_law', 'exponential', normalized_ratio=True)
    R_log, p_value_log = fit.distribution_compare('power_law', 'lognormal', normalized_ratio=True)
    R_tpl, p_value_tpl = fit.distribution_compare('power_law', 'truncated_power_law', normalized_ratio=True)
    R_lt, p_value_lt = fit.distribution_compare('lognormal', 'truncated_power_law', normalized_ratio=True)

    R_list.extend([R_exp, R_log, R_tpl, R_lt])
    p_value_list.extend([p_value_exp, p_value_log, p_value_tpl, p_value_lt])    

    #results["R"] = R_list
    # results["p_value"] = p_value_list
    results["is_power_law"] = (i in p_value_list < 0.05 for i in p_value_list) and (i in R_list > 0 for i in R_list) 

    print(f"Estimated alpha: {results['alpha']}")
    print(f"R: {R_list}")
    print(f"p-value: {p_value_list}")
    print(f"Is Power Law: {results['is_power_law']}")

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
        power_law_status = "Power Law" if results["is_power_law"] else "Not Power Law"
        plt.title(f"Log-Log Plot of Tumor Cluster Sizes with Power Law Fit ({power_law_status})")
        plt.legend()
        plt.tight_layout()
        plt.show()
    return results

def combine_datasets(num_datasets):
    """
    Combine multiple datasets into a single dataset.
    Input:
    - num_datasets: Number of datasets to combine
    """
    combined_dataset = []
    for i in range(num_datasets):
        _, _, _, cluster_sizes = simulate_CA(plot=False, bias_factor=0.93, tumor_prob=0.3)
        combined_dataset.extend(cluster_sizes[-1])
        print(f"Dataset {i+1} completed.")
    return combined_dataset

def simulations():
    """
    Run multiple simulations and analyze the power law distribution.
    Input:
    - num_simulations: Number of simulations to run
    - params: Parameters for the simulation

    Output:
    - results: A dictionary containing the simulation results
    """
    powerlaw_count = 0
    alphas = []
    x_mins = []
    all_results = []
    
    choice = input("Combine datasets (C) or run multiple simulations (S)?")

    if choice == "S":
        num_simulations = int(input("Enter number of simulations: "))
        bias_factor = float(input("Enter bias factor: "))
        # Run multiple simulations and analyze the power law distribution
        for i in range(num_simulations):
            _, _, _, cluster_sizes_over_time = simulate_CA(plot = False, bias_factor=bias_factor)
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

        print(f"Proportion of Power Law: {proportion_power_law:.2f}%")

        # Return results as a dictionary
        return {
            "proportion_power_law": proportion_power_law,
            "mean_alpha": mean_alpha,
            "std_alpha": std_alpha,
            "mean_xmin": mean_xmin,
            "std_xmin": std_xmin,
            "all_results": all_results,
        }
    
    else:
        # Combine multiple datasets and analyze the power law distribution
        num_datasets = int(input("Enter number of datasets to combine: "))
        combined_dataset = combine_datasets(num_datasets)
        results = analyze_power_law(combined_dataset, plot=True)
        return results

#simulations()

# Testing power-law for different bias values
test = input("Do you want to test the power-law fit for different bias values? (y/n): ")
if test == "y":
    min_value = input("Enter minimum bias factor: ")
    max_value = input("Enter maximum bias factor: ")
    num_values = input("Enter number of bias factor values: ")
    bias_factor = np.linspace(float(min_value), float(max_value), int(num_values))
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
        _, _, _, cluster_sizes = simulate_CA(plot=False, bias_factor=0.93)
        combined_dataset_start.extend(cluster_sizes[-1])
        combined_dataset_end.extend(cluster_sizes[0])
        print(f"Dataset {i+1} completed.")
    return combined_dataset_start, combined_dataset_end

combine_datasets_start, combine_datasets_end = combine_datasets(20)

analyze_power_law(combine_datasets_start, plot=True)
analyze_power_law(combine_datasets_end, plot=True)
