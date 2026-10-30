
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

from kmeans import kmeans, calculate_inertia


# ============================================================
# ASSIGNMENT 5 - MULTIPLE TRIALS ON SYNTHETIC DATA
#
# Purpose:
# 1. Generate the synthetic Gaussian dataset.
# 2. Run Random K-means and K-means++ 20 times each.
# 3. Use a different random seed for every trial.
# 4. Compare final inertia across the trials.
# 5. Calculate mean, standard deviation, and best inertia.
# 6. Save two graphs for the experimental analysis.
#
# WHY MULTIPLE TRIALS?
# Both initialization strategies involve randomness.
# A single trial may not represent their usual behavior.
# Repeating the experiment helps us measure consistency.
# ============================================================


# ------------------------------------------------------------
# STEP 1: CREATE RESULTS FOLDER
# ------------------------------------------------------------

# Create the folder where the graphs will be saved.
# exist_ok=True avoids an error if it already exists.
Path("results").mkdir(exist_ok=True)


# ------------------------------------------------------------
# STEP 2: GENERATE THE SYNTHETIC DATASET
# ------------------------------------------------------------

# Set a fixed random seed to reproduce the same dataset.
#
# IMPORTANT:
# This seed controls DATA GENERATION.
# It is separate from the seeds used to initialize K-means.
np.random.seed(42)

# Create three Gaussian groups.
#
# Each group contains 50 observations with two features.
# Adding coordinates shifts the groups to different centers.
group1 = np.random.randn(50, 2) + [2, 2]
group2 = np.random.randn(50, 2) + [8, 8]
group3 = np.random.randn(50, 2) + [2, 8]

# Combine all three groups into one dataset.
#
# X.shape = (150, 2)
# 150 observations and 2 features.
X = np.vstack([group1, group2, group3])


# ------------------------------------------------------------
# STEP 3: CONFIGURE THE EXPERIMENT
# ------------------------------------------------------------

# Run 20 trials for each initialization method.
#
# range(20) produces seeds:
# 0, 1, 2, ..., 19.
n_trials = 20

# These lists store the final inertia for each trial.
#
# After the experiment, each list will contain
# exactly 20 inertia values.
random_inertias = []
plusplus_inertias = []


# ------------------------------------------------------------
# STEP 4: RUN BOTH METHODS ACROSS 20 TRIALS
# ------------------------------------------------------------

# Repeat the experiment with 20 different random seeds.
for seed in range(n_trials):

    # ----------------------------------------
    # A. RANDOM K-MEANS
    # ----------------------------------------

    # Run K-means using random initialization.
    #
    # k=3 because the dataset contains 3 Gaussian groups.
    # seed=seed changes the initialization each trial.
    # init="random" selects starting centers uniformly.
    random_centers, random_labels = kmeans(
        X,
        k=3,
        seed=seed,
        init="random"
    )

    # Calculate the final clustering inertia.
    #
    # Inertia = sum of squared distances between
    # observations and their assigned centroids.
    random_inertia = calculate_inertia(
        X,
        random_centers,
        random_labels
    )

    # ----------------------------------------
    # B. K-MEANS++
    # ----------------------------------------

    # Run the same clustering algorithm on the same data.
    #
    # Only the initialization strategy changes.
    # K-means++ uses squared-distance-based sampling
    # to select its initial centroids.
    plus_centers, plus_labels = kmeans(
        X,
        k=3,
        seed=seed,
        init="kmeans++"
    )

    # Calculate the final inertia for K-means++.
    plus_inertia = calculate_inertia(
        X,
        plus_centers,
        plus_labels
    )

    # ----------------------------------------
    # C. STORE RESULTS
    # ----------------------------------------

    # Append each final inertia to its respective list.
    #
    # These lists will later be used to calculate
    # summary statistics and generate graphs.
    random_inertias.append(random_inertia)
    plusplus_inertias.append(plus_inertia)

    # Display the results for the current trial.
    #
    # seed + 1 displays trial numbers 1 through 20,
    # even though the seed values are 0 through 19.
    #
    # :.2f formats the numbers to two decimal places.
    print(
        f"Trial {seed + 1}: "
        f"Random = {random_inertia:.2f}, "
        f"K-means++ = {plus_inertia:.2f}"
    )


# ------------------------------------------------------------
# STEP 5: CALCULATE SUMMARY STATISTICS
# ------------------------------------------------------------

# Mean inertia:
# Measures the average final inertia across the 20 trials.
#
# Lower mean inertia indicates a lower average
# clustering objective value on the same dataset.

# Standard deviation:
# Measures how much the final inertia changes between trials.
#
# Lower standard deviation means the inertia values
# are more consistent across the tested random seeds.

# Best inertia:
# The smallest observed final inertia.
#
# This is the best result among our 20 trials,
# not necessarily the global optimum.

print("\nEXPERIMENT SUMMARY")

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


# ============================================================
# GRAPH 1: INERTIA DISTRIBUTION BOXPLOT
# ============================================================

# A boxplot summarizes how the inertia values
# are distributed across the 20 trials.
#
# It helps us identify variation and unusually
# high inertia values.

plt.figure(figsize=(8, 6))

# Plot both methods side by side.
plt.boxplot(
    [random_inertias, plusplus_inertias],
    tick_labels=["Random K-means", "K-means++"]
)

plt.title("Inertia Distribution Across 20 Trials")
plt.ylabel("Inertia (Lower is Better)")
plt.grid(axis="y", alpha=0.3)

# Adjust spacing to avoid overlapping labels.
plt.tight_layout()

# Save the boxplot at high resolution.
plt.savefig(
    "results/synthetic_inertia_boxplot.png",
    dpi=300,
    bbox_inches="tight"
)

# Display the boxplot.
plt.show()


# ============================================================
# GRAPH 2: INERTIA ACROSS 20 INDIVIDUAL TRIALS
# ============================================================

# This graph shows the actual inertia achieved
# by each method in every individual trial.
#
# It helps identify exactly which initialization
# produced a different or unusually poor result.

plt.figure(figsize=(10, 6))

# Create trial numbers 1 through 20.
trials = range(1, n_trials + 1)

# Plot the inertia values for random initialization.
#
# marker="o" displays circles at each trial.
plt.plot(
    trials,
    random_inertias,
    marker="o",
    label="Random K-means"
)

# Plot K-means++ on the same graph.
#
# marker="s" uses square markers.
# linestyle="--" produces a dashed line.
plt.plot(
    trials,
    plusplus_inertias,
    marker="s",
    linestyle="--",
    label="K-means++"
)

plt.title("K-means Initialization: Inertia Across 20 Trials")
plt.xlabel("Trial Number")
plt.ylabel("Inertia (Lower is Better)")

# Display every trial number on the horizontal axis.
plt.xticks(range(1, n_trials + 1))

# Show the legend identifying the two methods.
plt.legend()

# Add a grid to make values easier to compare.
plt.grid(True, alpha=0.3)

plt.tight_layout()

# Save the graph for the LaTeX report.
plt.savefig(
    "results/synthetic_inertia_trials.png",
    dpi=300,
    bbox_inches="tight"
)

# Display the graph.
plt.show()
