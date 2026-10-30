from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

from kmeans import kmeans, calculate_inertia

# ============================================================
# ASSIGNMENT 5 - TRIAL 2 COMPARISON
#
# Purpose:
# 1. Recreate the synthetic dataset.
# 2. Re-run the specific trial where random initialization
#    performed poorly (seed = 1).
# 3. Compare Random K-means and K-means++ on that same seed.
# 4. Visualize why the final clustering results differ.
#
# IMPORTANT:
# This script explains the unusual result observed in the
# 20-trial synthetic experiment.
# ============================================================

# ------------------------------------------------------------
# STEP 1: CREATE THE SAME SYNTHETIC DATASET
# ------------------------------------------------------------

# Set the dataset-generation seed so the exact same
# synthetic dataset is recreated every time.
np.random.seed(42)

# Generate three Gaussian groups of 2D points.
group1 = np.random.randn(50, 2) + [2, 2]
group2 = np.random.randn(50, 2) + [8, 8]
group3 = np.random.randn(50, 2) + [2, 8]

# Combine the three groups into one dataset.
X = np.vstack([group1, group2, group3])

# ------------------------------------------------------------
# STEP 2: CHOOSE THE SPECIFIC TRIAL TO REPRODUCE
# ------------------------------------------------------------

# In the 20-trial experiment, the second displayed trial
# used seed = 1 because the loop ran over seeds 0 to 19.
#
# This was the seed where random initialization produced
# the poor solution with much higher inertia.
seed = 1

# ------------------------------------------------------------
# STEP 3: RUN RANDOM K-MEANS
# ------------------------------------------------------------

# Run K-means using random initialization on this seed.
random_centers, random_labels = kmeans(
    X,
    k=3,
    seed=seed,
    init="random"
)

# ------------------------------------------------------------
# STEP 4: RUN K-MEANS++
# ------------------------------------------------------------

# Run K-means++ on the same dataset and same seed.
#
# This makes the comparison fair:
# - same data
# - same number of clusters
# - same convergence settings
# - only the initialization method differs
plus_centers, plus_labels = kmeans(
    X,
    k=3,
    seed=seed,
    init="kmeans++"
)

# ------------------------------------------------------------
# STEP 5: CALCULATE INERTIA FOR BOTH METHODS
# ------------------------------------------------------------

# Inertia measures the sum of squared distances between
# observations and their assigned centroids.
#
# Lower inertia means a more compact clustering solution.
random_inertia = calculate_inertia(
    X,
    random_centers,
    random_labels
)

plus_inertia = calculate_inertia(
    X,
    plus_centers,
    plus_labels
)

# Print the results for this specific trial.
print("\nTRIAL 2 COMPARISON")
print("Random K-means:", round(random_inertia, 2))
print("K-means++:", round(plus_inertia, 2))

# ------------------------------------------------------------
# STEP 6: PREPARE THE COMPARISON PLOT
# ------------------------------------------------------------

# Create a figure with two plots side by side.
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# Store the plotting information for both methods.
results = [
    (
        random_centers,
        random_labels,
        "Random K-means",
        random_inertia
    ),
    (
        plus_centers,
        plus_labels,
        "K-means++",
        plus_inertia
    )
]

# ------------------------------------------------------------
# STEP 7: DRAW BOTH FINAL CLUSTERING RESULTS
# ------------------------------------------------------------

for ax, (centers, labels, title, inertia) in zip(
    axes,
    results
):
    # Plot the observations, colored by cluster assignment.
    ax.scatter(
        X[:, 0],
        X[:, 1],
        c=labels,
        cmap="viridis",
        s=40,
        alpha=0.7
    )

    # Plot the final centroid positions.
    ax.scatter(
        centers[:, 0],
        centers[:, 1],
        color="red",
        marker="X",
        s=250,
        label="Final centroids"
    )

    # Add the title and inertia value to each subplot.
    ax.set_title(
        f"{title}\nInertia = {inertia:.2f}"
    )

    ax.set_xlabel("Feature 1")
    ax.set_ylabel("Feature 2")
    ax.legend()
    ax.grid(True, alpha=0.3)

# ------------------------------------------------------------
# STEP 8: SAVE THE FIGURE
# ------------------------------------------------------------

# Ensure the results folder exists.
Path("results").mkdir(exist_ok=True)

# Add a general title for the whole figure.
plt.suptitle(
    "Trial 2: Random K-means vs K-means++ (Seed = 1)"
)

# Adjust spacing to avoid overlap.
plt.tight_layout()

# Save the figure used in the LaTeX report.
plt.savefig(
    "results/trial2_comparison.png",
    dpi=300,
    bbox_inches="tight"
)

# Display the figure.
plt.show()

print("\nGraph saved successfully!")