
import csv
from pathlib import Path


# ============================================================
# ASSIGNMENT 5 - SAVE EXPERIMENTAL RESULTS
#
# Purpose:
# 1. Collect the summary statistics from our experiments.
# 2. Organize results from both datasets.
# 3. Save them in a CSV file for documentation.
#
# IMPORTANT:
# These are the results previously calculated by
# multiple_trials.py and digits_multiple_trials.py.
#
# This script does NOT rerun K-means.
# It records the existing results in a CSV file.
# ============================================================


# ------------------------------------------------------------
# STEP 1: CREATE THE RESULTS FOLDER
# ------------------------------------------------------------

# pathlib.Path allows us to work with file paths.
#
# mkdir() creates the folder if it does not exist.
# exist_ok=True prevents an error when it already exists.
Path("results").mkdir(exist_ok=True)


# ------------------------------------------------------------
# STEP 2: STORE THE EXPERIMENTAL RESULTS
# ------------------------------------------------------------

# Each row contains five values:
#
# 1. Dataset name
# 2. Initialization method
# 3. Mean inertia across 20 trials
# 4. Standard deviation of inertia
# 5. Best (lowest) observed inertia
#
# These numerical values come from the original
# experiments using seeds 0 through 19.
#
# We keep the full precision of the calculated values,
# rather than rounding them before saving.

results = [
    [
        "Gaussian Blobs",
        "Random K-means",
        329.4255922123372,
        188.4847967300323,
        286.18421424707327
    ],
    [
        "Gaussian Blobs",
        "K-means++",
        286.18421424707327,
        0.0,
        286.18421424707327
    ],
    [
        "Handwritten Digits",
        "Random K-means",
        1178338.220426592,
        19366.72295182389,
        1165160.3788412327
    ],
    [
        "Handwritten Digits",
        "K-means++",
        1179855.9452030575,
        16500.79325416999,
        1165372.0733570256
    ]
]


# ------------------------------------------------------------
# STEP 3: SAVE THE RESULTS AS A CSV FILE
# ------------------------------------------------------------

# A CSV (Comma-Separated Values) file stores
# tabular data in a simple text format.
#
# It can be opened using Excel, Python,
# or other data analysis applications.

with open(
    "results/experiment_summary.csv",
    "w",
    newline="",
    encoding="utf-8"
) as file:

    # Create a CSV writer for the open file.
    writer = csv.writer(file)

    # Write the column names as the first row.
    writer.writerow([
        "Dataset",
        "Method",
        "Mean Inertia",
        "Standard Deviation",
        "Best Inertia"
    ])

    # Write all four experimental result rows.
    #
    # writerows() saves multiple rows at once.
    writer.writerows(results)


# ------------------------------------------------------------
# STEP 4: CONFIRM SUCCESS
# ------------------------------------------------------------

print("Results saved successfully!")
print("File: results/experiment_summary.csv")
