from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

from kmeans import (
    initialize_centroids,
    assign_clusters,
    update_centroids,
    kmeans,
    initialize_centroids_plus_plus
)

# ============================================================
# SYNTHETIC DATASET FOR ASSIGNMENT 5
# This file:
# 1. Creates a simple 2D synthetic dataset
# 2. Visualizes how K-means works step by step
# 3. Saves all important graphs into the results folder
# ============================================================

# Create the results folder if it does not already exist.
# This ensures all generated figures can be saved.
Path("results").mkdir(exist_ok=True)

# ============================================================
# STEP 1: GENERATE THE SYNTHETIC DATASET
# ============================================================

# Set a fixed random seed so the same synthetic dataset
# is produced every time the script is run.
np.random.seed(42)

# Create three groups of 2D points.
# np.random.randn(50, 2) creates 50 observations
# with 2 features, sampled from a standard normal distribution.
#
# By adding [2, 2], [8, 8], and [2, 8],
# we shift the groups to different locations.
group1 = np.random.randn(50, 2) + [2, 2]
group2 = np.random.randn(50, 2) + [8, 8]
group3 = np.random.randn(50, 2) + [2, 8]

# Combine the three groups vertically into one dataset.
# The final dataset has shape (150, 2).
X = np.vstack([group1, group2, group3])

# Print basic information about the dataset.
print("Dataset shape:", X.shape)
print("First five observations:")
print(X[:5])

# ============================================================
# GRAPH 1: ORIGINAL SYNTHETIC DATASET
# ============================================================

# Plot the synthetic dataset before clustering.
plt.figure(figsize=(8, 6))
plt.scatter(X[:, 0], X[:, 1])

plt.title("Synthetic Dataset for K-means Clustering")
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.grid(True, alpha=0.3)

# Save the figure to the results folder.
plt.savefig(
    "results/synthetic_dataset.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()

# ============================================================
# GRAPH 2: RANDOM INITIAL CENTROIDS
# ============================================================

# Select 3 random starting centroids from the dataset.
centroids = initialize_centroids(X, k=3)

print("\nInitial centroids:")
print(centroids)

# Plot the dataset and the random starting centroids.
plt.figure(figsize=(8, 6))

plt.scatter(
    X[:, 0],
    X[:, 1],
    color="lightgray",
    label="Data points"
)

plt.scatter(
    centroids[:, 0],
    centroids[:, 1],
    color="red",
    marker="X",
    s=200,
    label="Initial centroids"
)

plt.title("K-means Random Initialization")
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.legend()
plt.grid(True, alpha=0.3)

plt.savefig(
    "results/random_initial_centroids.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()

# ============================================================
# GRAPH 3: INITIAL CLUSTER ASSIGNMENTS
# ============================================================

# Assign each observation to the closest centroid.
labels = assign_clusters(X, centroids)

print("\nFirst 10 cluster assignments:")
print(labels[:10])

# Plot the cluster assignments after the first assignment step.
# Different colors represent different assigned clusters.
plt.figure(figsize=(8, 6))

plt.scatter(
    X[:, 0],
    X[:, 1],
    c=labels,
    cmap="viridis",
    s=40
)

plt.scatter(
    centroids[:, 0],
    centroids[:, 1],
    color="red",
    marker="X",
    s=200,
    label="Initial centroids"
)

plt.title("K-means: Initial Cluster Assignments")
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.legend()
plt.grid(True, alpha=0.3)

plt.savefig(
    "results/initial_cluster_assignments.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()

# ============================================================
# GRAPH 4: UPDATED CENTROIDS
# ============================================================

# Recalculate the centroid positions as the mean of the
# observations assigned to each cluster.
new_centroids = update_centroids(X, labels, k=3)

print("\nOld centroids:")
print(centroids)

print("\nUpdated centroids:")
print(new_centroids)

# Plot both old and new centroid positions.
plt.figure(figsize=(8, 6))

plt.scatter(
    X[:, 0],
    X[:, 1],
    c=labels,
    cmap="viridis",
    alpha=0.5
)

# Old centroids are shown in red.
plt.scatter(
    centroids[:, 0],
    centroids[:, 1],
    color="red",
    marker="X",
    s=200,
    label="Old centroids"
)

# New centroids are shown in black.
plt.scatter(
    new_centroids[:, 0],
    new_centroids[:, 1],
    color="black",
    marker="X",
    s=200,
    label="New centroids"
)

plt.title("K-means: Updating Centroids")
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.legend()
plt.grid(True, alpha=0.3)

plt.savefig(
    "results/centroid_update.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()

# ============================================================
# GRAPH 5: FINAL K-MEANS CLUSTERING
# ============================================================

# Run the full K-means algorithm until convergence.
# We use k=3 because the synthetic dataset has 3 groups.
final_centroids, final_labels = kmeans(
    X,
    k=3,
    seed=42,
    init="random"
)

print("\nFinal centroid positions:")
print(final_centroids)

# Plot the final clustering result.
plt.figure(figsize=(8, 6))

plt.scatter(
    X[:, 0],
    X[:, 1],
    c=final_labels,
    cmap="viridis",
    s=40,
    alpha=0.7
)

plt.scatter(
    final_centroids[:, 0],
    final_centroids[:, 1],
    color="red",
    marker="X",
    s=250,
    label="Final centroids"
)

plt.title("Final K-means Clustering Results")
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.legend()
plt.grid(True, alpha=0.3)

plt.savefig(
    "results/final_kmeans.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()

# ============================================================
# GRAPH 6: RANDOM VS K-MEANS++ INITIALIZATION
# ============================================================

# Compare the starting centroids chosen by the two methods.
random_centers = initialize_centroids(X, k=3, seed=42)
plusplus_centers = initialize_centroids_plus_plus(X, k=3, seed=42)

# Create two plots side by side.
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# -------------------------
# Left plot: random initialization
# -------------------------
axes[0].scatter(
    X[:, 0],
    X[:, 1],
    color="lightgray"
)

axes[0].scatter(
    random_centers[:, 0],
    random_centers[:, 1],
    color="red",
    marker="X",
    s=200
)

axes[0].set_title("Random Initialization")

# -------------------------
# Right plot: K-means++ initialization
# -------------------------
axes[1].scatter(
    X[:, 0],
    X[:, 1],
    color="lightgray"
)

axes[1].scatter(
    plusplus_centers[:, 0],
    plusplus_centers[:, 1],
    color="green",
    marker="X",
    s=200
)

axes[1].set_title("K-means++ Initialization")

# Add common axis labels and grids to both plots.
for ax in axes:
    ax.set_xlabel("Feature 1")
    ax.set_ylabel("Feature 2")
    ax.grid(True, alpha=0.3)

plt.tight_layout()

plt.savefig(
    "results/initialization_comparison.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()