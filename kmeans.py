
import numpy as np


# ============================================================
# 1. RANDOM CENTROID INITIALIZATION
# ============================================================

def initialize_centroids(X, k, seed=42):
    """
    Select k initial centroids randomly from the dataset.

    X: Dataset containing all observations.
    k: Number of clusters we want to create.
    seed: Random seed used to reproduce the same result.

    Returns:
        A NumPy array containing k initial centroids.
    """

    # Create a random number generator with a fixed seed.
    # This ensures that the same seed produces the same
    # initial centroids whenever the code is executed.
    rng = np.random.default_rng(seed)

    # Randomly select k different observations.
    # replace=False prevents selecting the same index twice.
    indices = rng.choice(
        len(X),
        size=k,
        replace=False
    )

    # Use the selected observations as starting centroids.
    # .copy() creates an independent copy of the data.
    centroids = X[indices].copy()

    return centroids


# ============================================================
# 2. K-MEANS++ CENTROID INITIALIZATION
# ============================================================

def initialize_centroids_plus_plus(X, k, seed=42):
    """
    Select initial centroids using K-means++.

    K-means++ chooses the first centroid randomly.
    The remaining centroids are selected using
    probabilities proportional to squared distances
    from the nearest already-selected centroid.

    This encourages initial centers to be spread out.
    """

    # Create a reproducible random number generator.
    rng = np.random.default_rng(seed)

    # Select the first centroid uniformly at random.
    first_index = rng.choice(len(X))

    # Store selected centroids in a list.
    centroids = [X[first_index].copy()]

    # Continue selecting centers until we have k of them.
    for i in range(1, k):

        # Calculate squared Euclidean distances between
        # every observation and every selected centroid.
        #
        # np.newaxis adds dimensions so NumPy can subtract
        # the two arrays using broadcasting.
        #
        # axis=2 sums squared differences across features.
        distances = np.sum(
            (
                X[:, np.newaxis, :]
                - np.array(centroids)[np.newaxis, :, :]
            ) ** 2,
            axis=2
        )

        # For each observation, keep its squared distance
        # to the nearest already-selected centroid.
        #
        # axis=1 means we compare the selected centroids.
        min_distances = np.min(
            distances,
            axis=1
        )

        # Calculate the sum of these squared distances.
        # We use this to convert distances into probabilities.
        total = np.sum(min_distances)

        # If total is zero, all observations are already
        # at positions represented by selected centroids.
        # The current implementation stops in this case.
        if total == 0:
            raise ValueError(
                "Not enough distinct data points"
            )

        # D^2-weighted sampling:
        #
        # Probability = squared distance / total distance.
        #
        # Points farther from existing centroids have
        # a higher probability of being selected.
        probabilities = min_distances / total

        # Randomly choose the next centroid according
        # to the calculated probabilities.
        next_index = rng.choice(
            len(X),
            p=probabilities
        )

        # Add the newly selected centroid to the list.
        centroids.append(X[next_index].copy())

    # Convert the list to a NumPy array.
    return np.array(centroids)


# ============================================================
# 3. ASSIGN OBSERVATIONS TO CLUSTERS
# ============================================================

def assign_clusters(X, centroids):
    """
    Assign each observation to its nearest centroid.

    Returns:
        labels: An array containing the cluster index
                assigned to each observation.
    """

    # Calculate squared Euclidean distances between
    # every observation and every centroid.
    #
    # X[:, np.newaxis, :] has shape (n, 1, d).
    # centroids[np.newaxis, :, :] has shape (1, k, d).
    #
    # Broadcasting produces a distance for every
    # observation-centroid combination.
    #
    # axis=2 sums squared differences across features.
    distances = np.sum(
        (
            X[:, np.newaxis, :]
            - centroids[np.newaxis, :, :]
        ) ** 2,
        axis=2
    )

    # Choose the nearest centroid for each observation.
    #
    # np.argmin returns the index of the smallest value.
    # axis=1 means we compare distances across centroids.
    #
    # Cluster indices start at 0.
    labels = np.argmin(distances, axis=1)

    return labels


# ============================================================
# 4. UPDATE CENTROID POSITIONS
# ============================================================

def update_centroids(X, labels, k):
    """
    Recalculate every centroid as the mean of
    the observations assigned to its cluster.
    """

    # Store the updated centroid positions.
    new_centroids = []

    # Go through every cluster, from 0 to k-1.
    for i in range(k):

        # Select observations assigned to cluster i.
        #
        # labels == i produces a Boolean mask.
        # X[labels == i] keeps only matching observations.
        cluster_points = X[labels == i]

        # If the cluster contains observations,
        # calculate the mean position across all features.
        if len(cluster_points) > 0:
            new_center = np.mean(
                cluster_points,
                axis=0
            )

        else:
            # The current implementation raises an error
            # if a cluster receives no observations.
            #
            # This was the behavior used in our original
            # experiments, so we preserve it here.
            raise ValueError(
                "A cluster has no assigned points"
            )

        # Save the calculated centroid.
        new_centroids.append(new_center)

    # Return all updated centroids as a NumPy array.
    return np.array(new_centroids)


# ============================================================
# 5. COMPLETE K-MEANS ALGORITHM
# ============================================================

def kmeans(
    X,
    k,
    seed=42,
    max_iter=100,
    tol=1e-4,
    init="random"
):
    """
    Run K-means clustering using either initialization.

    Parameters:
        X: Input dataset.
        k: Number of clusters.
        seed: Random seed for reproducibility.
        max_iter: Maximum number of iterations.
        tol: Convergence tolerance.
        init: "random" or "kmeans++".

    Returns:
        centroids: Final cluster centers.
        labels: Final cluster assignment for each point.
    """

    # STEP 1: Select the initialization strategy.
    if init == "random":

        # Randomly select k observations as centroids.
        centroids = initialize_centroids(
            X, k, seed
        )

    elif init == "kmeans++":

        # Select centroids using D^2-weighted sampling.
        centroids = initialize_centroids_plus_plus(
            X, k, seed
        )

    else:
        # Stop if an unsupported initialization is given.
        raise ValueError(
            "Unknown initialization method"
        )

    # STEP 2: Repeat assignment and centroid updating.
    for iteration in range(max_iter):

        # Assign every observation to its nearest centroid.
        labels = assign_clusters(
            X, centroids
        )

        # Calculate new centroids using cluster means.
        new_centroids = update_centroids(
            X, labels, k
        )

        # Calculate how far each centroid moved.
        #
        # new_centroids - centroids gives displacement
        # vectors for each centroid.
        #
        # np.linalg.norm(..., axis=1) calculates the
        # Euclidean length of each displacement vector.
        movement = np.linalg.norm(
            new_centroids - centroids,
            axis=1
        )

        # Replace old centroids with the updated ones.
        centroids = new_centroids

        # STEP 3: Check convergence.
        #
        # np.all returns True only if EVERY centroid
        # moved less than the chosen tolerance.
        if np.all(movement < tol):
            print(
                "Converged after",
                iteration + 1,
                "iterations"
            )
            break

    else:
        # This runs only if the loop finishes without
        # reaching the convergence condition.
        print("Reached maximum iterations")

    # STEP 4: Assign observations using final centroids.
    #
    # This ensures labels correspond to the final
    # centroid positions rather than the previous ones.
    labels = assign_clusters(
        X, centroids
    )

    return centroids, labels


# ============================================================
# 6. CALCULATE CLUSTERING INERTIA
# ============================================================

def calculate_inertia(X, centroids, labels):
    """
    Calculate the final K-means clustering inertia.

    Inertia is the sum of squared distances between
    observations and their assigned centroids.

    A lower inertia indicates a smaller clustering
    objective value for the same dataset.
    """

    # Find the centroid assigned to each observation.
    #
    # labels contains cluster indices.
    # centroids[labels] retrieves the corresponding
    # centroid for every observation.
    assigned_centroids = centroids[labels]

    # Calculate the difference between each observation
    # and its assigned centroid.
    differences = X - assigned_centroids

    # Square all feature differences and sum them.
    #
    # This gives the total within-cluster
    # sum of squared Euclidean distances.
    inertia = np.sum(differences ** 2)

    return inertia
