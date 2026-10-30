
import numpy as np
import matplotlib.pyplot as plt

from kmeans import kmeans, calculate_inertia


# ============================================================
# ASSIGNMENT 5 - COMPARING INITIALIZATION METHODS
#
# Purpose:
# 1. Generate the same synthetic dataset.
# 2. Run K-means with random initialization.
# 3. Run K-means with K-means++ initialization.
# 4. Compare their final inertia values.
# 5. Visualize the final clustering results side by side.
#
# IMPORTANT:
# This experiment compares both methods using ONE seed.
# The multiple_trials.py file performs the 20-trial
# comparison used for the main statistical analysis.
# ============================================================


# ------------------------------------------------------------
# STEP 1: GENERATE THE SYNTHETIC DATASET
# ------------------------------------------------------------

# Set a fixed seed so we generate the same dataset
# every time this script runs.
np.random.seed(42)

# Generate three Gaussian groups.
#
# np.random.randn(50, 2) generates 50 observations,
# each containing two features.
#
# Adding the coordinates shifts each Gaussian group
# to a different location in the 2D feature space.
group1 = np.random.randn(50, 2) + [2, 2]
group2 = np.random.randn(50, 2) + [8, 8]
group3 = np.random.randn(50, 2) + [2, 8]

# Combine all three groups into one array.
#
# X.shape = (150, 2)
# 150 observations, 2 numerical features.
X = np.vstack([group1, group2, group3])


# ------------------------------------------------------------
# STEP 2: RUN K-MEANS WITH RANDOM INITIALIZATION
# ------------------------------------------------------------

# Run our implementation of K-means using
# uniform random centroid initialization.
#
# k=3: We want three clusters.
# seed=42: Makes the random process reproducible.
# init="random": Selects initial centers uniformly
# from the observations without replacement.
#
# The function returns:
# random_centers = Final centroid positions
# random_labels = Cluster assignment for every point
random_centers, random_labels = kmeans(
    X,
    k=3,
    seed=42,
    init="random"
)


# ------------------------------------------------------------
# STEP 3: RUN K-MEANS WITH K-MEANS++ INITIALIZATION
# ------------------------------------------------------------

# Run the same K-means algorithm on the same dataset.
#
# The only algorithmic difference is the initialization:
# init="kmeans++" selects starting centers using
# probabilities based on squared distances.
#
# We use the same seed value to make both runs
# reproducible. However, the selected centers
# may differ because the initialization rules differ.
plus_centers, plus_labels = kmeans(
    X,
    k=3,
    seed=42,
    init="kmeans++"
)


# ------------------------------------------------------------
# STEP 4: CALCULATE FINAL INERTIA
# ------------------------------------------------------------

# Inertia is the sum of squared Euclidean distances
# between observations and their assigned centroids.
#
# A LOWER inertia indicates a better result
# according to the K-means clustering objective.

# Calculate inertia for random initialization.
random_inertia = calculate_inertia(
    X,
    random_centers,
    random_labels
)

# Calculate inertia for K-means++ initialization.
plus_inertia = calculate_inertia(
    X,
    plus_centers,
    plus_labels
)


# ------------------------------------------------------------
# STEP 5: DISPLAY THE COMPARISON RESULTS
# ------------------------------------------------------------

# Print the final inertia for both initialization methods.
#
# round(value, 2) displays the result with two decimals.
print("\nCOMPARISON RESULTS")

print(
    "Random K-means inertia:",
    round(random_inertia, 2)
)

print(
    "K-means++ inertia:",
    round(plus_inertia, 2)
)


# ------------------------------------------------------------
# STEP 6: VISUALIZE BOTH CLUSTERING RESULTS
# ------------------------------------------------------------

# Create a figure containing two plots side by side.
#
# axes[0] = Random initialization
# axes[1] = K-means++ initialization
fig, axes = plt.subplots(
    1,
    2,
    figsize=(12, 5)
)

# Store the results for both methods in one list.
#
# This avoids writing the same plotting code twice.
results = [
    (
        random_centers,
        random_labels,
        "Random K-means"
    ),
    (
        plus_centers,
        plus_labels,
        "K-means++"
    )
]

# Go through both sets of results and draw each plot.
#
# zip(axes, results) pairs each subplot with
# the clustering results it should display.
for ax, (centers, labels, title) in zip(axes, results):

    # Plot each observation.
    #
    # X[:, 0] = First feature (horizontal axis)
    # X[:, 1] = Second feature (vertical axis)
    #
    # c=labels colors points according to
    # their assigned clusters.
    ax.scatter(
        X[:, 0],
        X[:, 1],
        c=labels,
        cmap="viridis",
        s=35
    )

    # Plot the final centroid positions.
    #
    # Red X markers distinguish centroids
    # from ordinary observations.
    ax.scatter(
        centers[:, 0],
        centers[:, 1],
        color="red",
        marker="X",
        s=200
    )

    # Add plot titles and axis labels.
    ax.set_title(title)
    ax.set_xlabel("Feature 1")
    ax.set_ylabel("Feature 2")

    # Add a grid to make the plots easier to read.
    ax.grid(True, alpha=0.3)


# Adjust spacing between the plots.
plt.tight_layout()

# Display the completed comparison.
plt.show()
