
# Assignment 5 — Clustering and Initialization

**Course:** Statistical Methods for Machine Learning  
**Student:** Jaafar Youness  
**Academic Year:** 2025/2026

## Project Description

This project compares two initialization methods for K-means clustering:

- **Random initialization:** Selects initial centroids randomly.
- **K-means++:** Selects initial centroids using squared-distance probabilities.

Both methods were implemented from scratch in Python using NumPy.

The objective is to compare clustering performance and stability.

## Datasets

Two datasets were used:

1. **Synthetic Gaussian dataset:** 150 observations, 2 features, and 3 clusters.
2. **Handwritten Digits dataset:** 1,797 observations, 64 features, and 10 clusters.

## Experiments

Both methods were tested using 20 different random seeds (0–19).

We compared:
- Mean inertia
- Standard deviation
- Best observed inertia

### Results

| Dataset | Method | Mean Inertia | Std. Deviation |
|---|---|---:|---:|
| Synthetic | Random K-means | 329.43 | 188.48 |
| Synthetic | K-means++ | 286.18 | 0.00 |
| Digits | Random K-means | 1,178,338.22 | 19,366.72 |
| Digits | K-means++ | 1,179,855.95 | 16,500.79 |

K-means++ was more consistent on the synthetic dataset. On the Digits dataset, random initialization achieved slightly lower average inertia, while K-means++ showed lower variability.

## How to Run

Install the required libraries:

```bash
pip install numpy matplotlib scikit-learn
```

Run the main experiments:

```bash
python multiple_trials.py
python digits_multiple_trials.py
```

The experiment graphs are saved in the `results` folder.

## Project Structure

- `kmeans.py` — K-means and K-means++ implementation
- `synthetic_data.py` — Synthetic dataset visualizations
- `digits_data.py` — Digits dataset visualization
- `multiple_trials.py` — Synthetic dataset experiments
- `digits_multiple_trials.py` — Digits dataset experiments
- `trial2_comparison.py` — Analysis of an unsuccessful random initialization
- `results/` — Graphs and experimental results
- `Assignment5_Report.pdf` — Final written report

## Reference

Arthur, D. & Vassilvitskii, S. (2007).
*k-means++: The Advantages of Careful Seeding.*
```
