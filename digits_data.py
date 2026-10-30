
from pathlib import Path
from sklearn.datasets import load_digits
import matplotlib.pyplot as plt


# ============================================================
# ASSIGNMENT 5 - HANDWRITTEN DIGITS DATASET
#
# Purpose:
# 1. Load a real-world dataset using scikit-learn.
# 2. Explore the dataset's dimensions and labels.
# 3. Visualize 10 handwritten digit images.
# 4. Save the visualization for the project report.
#
# IMPORTANT:
# This file only loads and visualizes the dataset.
# The actual clustering experiments are performed
# in digits_comparison.py and digits_multiple_trials.py.
# ============================================================


# ------------------------------------------------------------
# STEP 1: CREATE RESULTS FOLDER
# ------------------------------------------------------------

# Path is used to work with folders and file paths.
# mkdir creates the results folder if it does not exist.
# exist_ok=True prevents an error if the folder exists.
Path("results").mkdir(exist_ok=True)


# ------------------------------------------------------------
# STEP 2: LOAD THE HANDWRITTEN DIGITS DATASET
# ------------------------------------------------------------

# load_digits() loads a dataset included in scikit-learn.
#
# The dataset contains:
# - 1,797 handwritten digit images
# - 10 digit classes (0 through 9)
# - 8 x 8 pixels per image
# - 64 numerical features per observation
digits = load_digits()


# ------------------------------------------------------------
# STEP 3: EXTRACT FEATURES AND LABELS
# ------------------------------------------------------------

# digits.data contains the feature matrix.
#
# Each row represents one handwritten digit.
# Each column represents one pixel.
#
# Shape: (1797, 64)
X_digits = digits.data

# digits.target contains the correct digit labels.
#
# For example, a label of 5 means the image represents
# the handwritten digit 5.
#
# IMPORTANT:
# We use these labels here only to identify images
# in the visualization.
#
# The K-means clustering algorithms do NOT use
# the correct labels during clustering.
y_digits = digits.target


# ------------------------------------------------------------
# STEP 4: DISPLAY DATASET INFORMATION
# ------------------------------------------------------------

# .shape gives the dimensions of the feature matrix.
#
# 1797 = number of observations
# 64 = number of features per observation
print("Dataset shape:", X_digits.shape)

# len() counts how many labels are available.
# There is one label for every observation.
print("Number of labels:", len(y_digits))


# ------------------------------------------------------------
# STEP 5: VISUALIZE HANDWRITTEN DIGITS
# ------------------------------------------------------------

# Create a figure with 2 rows and 5 columns.
# This gives us 10 image positions.
#
# figsize=(10, 5) controls the overall figure size.
fig, axes = plt.subplots(
    2,
    5,
    figsize=(10, 5)
)

# axes.flat allows us to go through every subplot
# as one sequence rather than working row by row.
#
# enumerate() gives us:
# i  = image index
# ax = the subplot where that image will be displayed
for i, ax in enumerate(axes.flat):

    # digits.images contains the original 8 x 8
    # image representation of each digit.
    #
    # imshow() displays the image.
    # cmap="gray" displays the image in grayscale.
    ax.imshow(
        digits.images[i],
        cmap="gray"
    )

    # Display the true digit label above the image.
    ax.set_title(f"Digit: {y_digits[i]}")

    # Hide the axes because pixel coordinates are
    # not needed for this visualization.
    ax.axis("off")


# ------------------------------------------------------------
# STEP 6: ADJUST FIGURE LAYOUT
# ------------------------------------------------------------

# Automatically adjust spacing so subplot titles
# and images do not overlap.
plt.tight_layout()


# ------------------------------------------------------------
# STEP 7: SAVE THE VISUALIZATION
# ------------------------------------------------------------

# Save the figure in the results folder.
#
# dpi=300 produces a high-resolution image.
# bbox_inches="tight" reduces unnecessary whitespace.
#
# This image is included in the LaTeX report
# to illustrate the real-world dataset.
plt.savefig(
    "results/digits_samples.png",
    dpi=300,
    bbox_inches="tight"
)


# ------------------------------------------------------------
# STEP 8: DISPLAY THE VISUALIZATION
# ------------------------------------------------------------

# Open the figure window to display the images.
plt.show()

# Confirm that the graph was saved.
print("Graph saved successfully!")
