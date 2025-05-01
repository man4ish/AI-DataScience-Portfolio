# Customer Segmentation via Clustering

This project performs unsupervised customer segmentation using K-Means and DBSCAN on a small customer dataset. Principal Component Analysis (PCA) is used to reduce feature dimensions for visualization.

## Dataset

A simple CSV file `customers.csv` with the following columns:
- `CustomerID`
- `Age`
- `AnnualIncome`
- `SpendingScore`

## Techniques Used

- **K-Means Clustering** for customer grouping
- **DBSCAN** for density-based clustering
- **PCA** for dimensionality reduction
- **StandardScaler** for feature normalization

## Requirements

- Python 3.8+
- pandas
- seaborn
- matplotlib
- scikit-learn

Install dependencies:
```bash
pip install pandas matplotlib seaborn scikit-learn
```

## How to Run

```bash
python customer_segmentation.py
```

## Output

- Displays 2 scatter plots for K-Means and DBSCAN clusters using PCA components.
- Saves a plot as `cluster_results.png`.

## Focus

- Unsupervised learning
- Customer behavior analysis
- Dimensionality reduction
- Clustering evaluation
