"""
This file contains thcode for tubule complex length analyses on the simulated images.
Tubule complex length refers to the total number of pixels in a tubule complex. 
"""

import numpy as np
from skimage import io, color, filters, morphology, measure
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import powerlaw

image = io.imread('images/final_grid.png')

# For some reason, the image has 4 channels, so we only keep the first 3 (RGB)
image = image[:, :, :3]
image = color.rgb2gray(image)
threshold = filters.threshold_otsu(image)
binary_image = image > threshold # Image binarization. Not sure why the original image isn't already binary...

skeleton = morphology.skeletonize(binary_image)
plt.figure(figsize=(10, 7.5))
plt.title("Skeletonized Image")
plt.imshow(skeleton, cmap='gray')
plt.axis('off')
plt.show()

labeled_image = measure.label(skeleton, connectivity=2)
num_complexes = labeled_image.max()

# Counts the number of pixels in each tubule complex
lengths = []
for region in measure.regionprops(labeled_image):
    region_length = np.sum(labeled_image == region.label)
    lengths.append(region_length)

complex_colors = ['red', 'green', 'blue', 'yellow', 'orange', 'purple', 'cyan', 'magenta']
num_colors_needed = num_complexes
if num_colors_needed > len(complex_colors): 
    complex_colors = complex_colors * (num_colors_needed // len(complex_colors) + 1)
cmap = ListedColormap(['black'] + complex_colors[:num_colors_needed])
plt.figure(figsize=(10, 7.5))
plt.title("Labeled Tubule Complexes")
plt.imshow(labeled_image, cmap=cmap, vmin=0)
plt.axis('off')
plt.show()

fit = powerlaw.Fit(lengths)
plt.figure(figsize=(8, 6))
fit.plot_pdf(color='blue', linewidth=0, marker='o', label='Simulated Data')
fit.power_law.plot_pdf(color='red', linestyle='--', label=f'Power-Law Fit ($\\alpha={fit.alpha:.2f}$)')
plt.xlabel("Tubule Complex Length")
plt.ylabel("Frequency")
plt.title("Power-Law Fit to Tubule Complex Length Distribution")
plt.legend()
#plt.grid(True, which="both", linestyle="--", linewidth=0.5)
plt.show()

R1, p1 = fit.distribution_compare('power_law', 'exponential')
print(f"Comparing Power Law to Exponential - R1: {R1}, p1: {p1}")
print(f"Estimated scaling exponent (alpha): {fit.alpha:.2f}")
