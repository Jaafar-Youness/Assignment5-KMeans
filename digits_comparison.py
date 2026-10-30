
from sklearn.datasets import load_digits
from kmeans import kmeans, calculate_inertia


# ============================================================
# ASSIGNMENT 5 - HANDWRITTEN DIGITS COMPARISON
#
# Purpose:
# 1. Load the real-world Handwritten Digits dataset.
# 2. Run K-means using random initialization.
# 3. Run K-means using K-means++ initialization.
# 4. Calculate and compare the final inertia values.
#
# IMPORTANT:
# This file compares the two methods using ONE seed (42).
#
# The digits_multiple_trials.py file performs the
# 20-trial comparison used in the main report.
# ============================================================


# ------------------------------------------------------------
# STEP 1: LOAD THE HANDWRITTEN DIGITS DATASET
# ------------------------------------------------------------

# load_digits() loads the handwritten digits dataset
# provided by scikit-learn.
#
# It contains:
# - 1,797 observations
# - 64 numerical features per observation
# - Handwritten digit images from 0 to 9
digits = load_digits()

# digits.data contains the numerical feature matrix.
#
# Each row represents one handwritten digit.
# Each column represents one pixel value.
#
# astype(float) ensures the values have a
# floating-point numerical data type.
#
# Floating-point values are appropriate for
# distance calculations and centroid averages.
X = digits.data.astype(float)

# Print the dataset dimensions.
# Expected output: (1797, 64)
print("Dataset shape:", X.shape)


# ------------------------------------------------------------
# STEP 2: RUN RANDOM K-MEANS
# ------------------------------------------------------------

# Run our own K-means implementation.
#
# X: The handwritten digits feature matrix.
# k=10: Create 10 clusters, corresponding to
#       the 10 digit categories (0 through 9).
# seed=42: Reproduce the same initialization.
# init="random": Choose 10 initial centroids
#                uniformly from observations.
#
# The function returns the final centroids
# and the cluster assignment for each digit.
random_centers, random_labels = kmeans(
    X,
    k=10,
    seed=42,
    init="random"
)


# ------------------------------------------------------------
# STEP 3: RUN K-MEANS++
# ------------------------------------------------------------

# Run the same clustering algorithm and parameters
# using K-means++ initialization instead.
#
# K-means++ chooses starting centers with
# probabilities proportional to squared distances
# from previously selected centers.
#
# Only the initialization strategy changes.
plus_centers, plus_labels = kmeans(
    X,
    k=10,
    seed=42,
    init="kmeans++"
)


# ------------------------------------------------------------
# STEP 4: CALCULATE INERTIA FOR BOTH METHODS
# ------------------------------------------------------------

# Inertia is the sum of squared Euclidean distances
# from observations to their assigned final centroids.
#
# Lower inertia means a smaller K-means objective
# for the same dataset.

# Final inertia for random initialization.
random_inertia = calculate_inertia(
    X,
    random_centers,
    random_labels
)

# Final inertia for K-means++.
plus_inertia = calculate_inertia(
    X,
    plus_centers,
    plus_labels
)


# ------------------------------------------------------------
# STEP 5: DISPLAY COMPARISON RESULTS
# ------------------------------------------------------------

print("\nHANDWRITTEN DIGITS RESULTS")

# :.2f displays the inertia with two decimal places.
print(f"Random K-means: {random_inertia:.2f}")
print(f"K-means++: {plus_inertia:.2f}")
