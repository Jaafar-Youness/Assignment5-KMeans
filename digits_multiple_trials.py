
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_digits

from kmeans import kmeans, calculate_inertia


# ============================================================
# ASSIGNMENT 5 - MULTIPLE TRIALS ON HANDWRITTEN DIGITS
#
# Purpose:
# 1. Load the real-world Handwritten Digits dataset.
# 2. Run Random K-means and K-means++ 20 times each.
# 3. Use different random seeds for initialization.
# 4. Measure final inertia for every trial.
# 5. Calculate mean, standard deviation, and best inertia.
# 6. Save a graph comparing the results across trials.
#
# IMPORTANT:
# This is the experiment used for the Digits results
# in Section 5 of the LaTeX report.
# ============================================================


# ------------------------------------------------------------
# STEP 1: LOAD THE HANDWRITTEN DIGITS DATASET
# ------------------------------------------------------------

# load_digits() loads the Digits dataset from scikit-learn.
#
# The dataset contains:
# - 1,797 observations
# - 64 numerical features per observation
# - 10 digit categories, from 0 to 9
digits = load_digits()

# Extract the numerical feature matrix.
#
# Each row represents one handwritten digit.
# Each column corresponds to a pixel in an 8x8 image.
#
# astype(float) ensures floating-point values are used
# for distance calculations and centroid updates.
X = digits.data.astype(float)

# We do not use digits.target during clustering.
# K-means is unsupervised and only receives the features.

print("Dataset shape:", X.shape)


# ------------------------------------------------------------
# STEP 2: CONFIGURE THE EXPERIMENT
# ------------------------------------------------------------

# Number of repeated trials.
#
# Each trial uses a different random seed.
# range(20) generates seeds from 0 to 19.
n_trials = 20

# Number of clusters.
#
# We use k=10 because the dataset contains ten
# different handwritten digit categories.
#
# The true digit labels are not used by K-means.
k = 10

# Lists used to store the final inertia from each trial.
#
# At the end, each list will contain 20 values.
random_inertias = []
plusplus_inertias = []


# ------------------------------------------------------------
# STEP 3: REPEAT BOTH METHODS ACROSS 20 TRIALS
# ------------------------------------------------------------

for seed in range(n_trials):

    # ----------------------------------------
    # A. RANDOM K-MEANS
    # ----------------------------------------

    # Run our K-means implementation using
    # uniform random centroid initialization.
    #
    # seed changes at each trial, producing
    # different reproducible initializations.
    random_centers, random_labels = kmeans(
        X,
        k=k,
        seed=seed,
        init="random"
    )

    # ----------------------------------------
    # B. K-MEANS++
    # ----------------------------------------

    # Run the same algorithm on the same dataset,
    # using K-means++ for the initial centroids.
    #
    # K-means++ selects centers using probabilities
    # proportional to squared distances from
    # already-selected centroids.
    plus_centers, plus_labels = kmeans(
        X,
        k=k,
        seed=seed,
        init="kmeans++"
    )

    # ----------------------------------------
    # C. CALCULATE INERTIA
    # ----------------------------------------

    # Inertia is the sum of squared Euclidean
    # distances from observations to their
    # assigned final centroids.
    #
    # Lower inertia means a smaller value of
    # the K-means objective on the same dataset.

    # Final inertia for random initialization.
    random_inertia = calculate_inertia(
        X,
        random_centers,
        random_labels
    )

    # Final inertia for K-means++ initialization.
    plus_inertia = calculate_inertia(
        X,
        plus_centers,
        plus_labels
    )

    # ----------------------------------------
    # D. STORE THE RESULTS
    # ----------------------------------------

    # Add each result to its corresponding list.
    # We will use these values for statistics and plotting.
    random_inertias.append(random_inertia)
    plusplus_inertias.append(plus_inertia)

    # Print the final inertia from this trial.
    #
    # seed + 1 converts seeds 0-19 into
    # displayed trial numbers 1-20.
    #
    # :.2f displays values with two decimal places.
    print(
        f"Trial {seed + 1}: "
        f"Random = {random_inertia:.2f}, "
        f"K-means++ = {plus_inertia:.2f}"
    )


# ------------------------------------------------------------
# STEP 4: CALCULATE SUMMARY STATISTICS
# ------------------------------------------------------------

# Mean inertia:
# Average final inertia over the 20 trials.
#
# Standard deviation:
# Measures how much inertia varies between trials.
# np.std() uses population standard deviation by default.
#
# Best inertia:
# The lowest final inertia observed among the 20 trials.
#
# These values are compared separately for each method.

print("\nDIGITS EXPERIMENT SUMMARY")

print("\nRandom K-means:")

print(
    "Mean inertia:",
    np.mean(random_inertias)
)

print(
    "Standard deviation:",
    np.std(random_inertias)
)

print(
    "Best inertia:",
    np.min(random_inertias)
)

print("\nK-means++:")

print(
    "Mean inertia:",
    np.mean(plusplus_inertias)
)

print(
    "Standard deviation:",
    np.std(plusplus_inertias)
)

print(
    "Best inertia:",
    np.min(plusplus_inertias)
)


# ------------------------------------------------------------
# STEP 5: CREATE THE RESULTS FOLDER
# ------------------------------------------------------------

# Ensure that the output folder exists.
#
# exist_ok=True prevents an error if it exists already.
Path("results").mkdir(exist_ok=True)


# ------------------------------------------------------------
# STEP 6: CREATE THE COMPARISON GRAPH
# ------------------------------------------------------------

# This graph displays final inertia for each of the 20 trials.
#
# It helps us compare the variability of both initialization
# methods and see which performed better in each trial.
plt.figure(figsize=(10, 6))

# Create trial numbers from 1 to 20.
trials = range(1, n_trials + 1)

# Plot the random K-means inertia values.
#
# marker="o" adds a circle at every trial.
plt.plot(
    trials,
    random_inertias,
    marker="o",
    label="Random K-means"
)

# Plot the K-means++ inertia values.
#
# marker="s" uses square markers.
# linestyle="--" uses a dashed line.
plt.plot(
    trials,
    plusplus_inertias,
    marker="s",
    linestyle="--",
    label="K-means++"
)

# Add the title and axis descriptions.
plt.title("Handwritten Digits: Inertia Across 20 Trials")
plt.xlabel("Trial Number")
plt.ylabel("Inertia")

# Show each trial number on the horizontal axis.
plt.xticks(range(1, n_trials + 1))

# Display the legend identifying the two methods.
plt.legend()

# Add a grid to make the plot easier to read.
plt.grid(True, alpha=0.3)

# Adjust spacing so graph elements do not overlap.
plt.tight_layout()


# ------------------------------------------------------------
# STEP 7: SAVE THE GRAPH
# ------------------------------------------------------------

# Save the graph at high resolution.
#
# dpi=300 makes it suitable for the report.
# bbox_inches="tight" removes extra whitespace.
#
# This file is included as Figure 5 in the LaTeX report.
plt.savefig(
    "results/digits_inertia_trials.png",
    dpi=300,
    bbox_inches="tight"
)

# Display the completed graph.
plt.show()

print("\nGraph saved to results/digits_inertia_trials.png")
