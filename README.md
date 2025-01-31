# Tumor Angiogenesis Simulation and Analysis

This project implements a cellular automaton (CA) model to simulate tumor angiogenesis, along with tools for analyzing the resulting data. The project includes modules for simulating tumor growth and blood vessel formation, performing network analyses, and visualizing the results. This implementation and project was inspired by the model found in Peña et al. (2024). 

---

## Features

1. **Simulation of Tumor Angiogenesis**  
   - Cellular automaton-based model incorporating blood vessel growth and tumor cell proliferation/death.
   - VEGF-driven stochastic angiogenesis.
   - Tumor entropy and clustering analysis.

2. **Complex Length Analysis**  
   - Measures the lengths of connected tubule complexes in skeletonized images.
   - Fits power-law distributions to tubule lengths for statistical analysis.

3. **Network Analysis**  
   - Converts skeletonized images of vascular structures into graph representations.
   - Computes network metrics such as:
     - Average degree
     - Betweenness centrality
     - Page Rank
     - Clustering coefficient

4. **Visualization Tools**  
   - Animated histograms of tumor cluster sizes.
   - Distribution plots with fitted power-law curves.
   - Network metric visualizations against experimental parameters.

---

## Installation

### Requirements
- Python 3.8+
- Libraries:
  - `numpy`
  - `matplotlib`
  - `scikit-image`
  - `pandas`
  - `networkx`
  - `powerlaw`
  - `scipy`
  - `numba`
  - `skan`

Install dependencies via pip:
```bash
pip install numpy matplotlib scikit-image pandas networkx powerlaw scipy numba skan
```

---

## Usage

### 1. **Simulating Tumor Angiogenesis**
Run the `growthdeath_jit.py` script to simulate angiogenesis:
```bash
python3 growthdeath_jit.py
```
Parameters include:
- `temp`: ADD ALL FINAL PARAMETERS HERE.

---

### 2.  **Network Analysis**
Perform network-based analysis on skeletonized images:
```bash
python3 network_analysis.py
```
This script calculates network metrics for vascular structures, including average degree and clustering coefficient.

---

### 3.  **Network Time**
Perform network analysis on multiple visualizations of the vascular structure at different points in
the simulation
```bash
python3 network_time.py
```
This script calculates network metrics for vascular structures at different points during the simulation, including average degree, clustering coefficient, and average betweenness centrality.

---

### 4. **Plotting Results**
Visualize experimental results using `plot_results.py`:
```bash
python3 plot_results.py
```
This script plots the data gathered from running network_analysis.py. It grants multiple options for plotting the data:
- `temp`: ADD FINAL OPTIONS.
