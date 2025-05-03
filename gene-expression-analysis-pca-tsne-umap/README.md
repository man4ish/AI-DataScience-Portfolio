# Gene Expression Analysis - Dimensionality Reduction (PCA, t-SNE, UMAP)

This project performs dimensionality reduction on gene expression data using Principal Component Analysis (PCA), t-SNE, and UMAP, and visualizes the results in 2D plots.

## Project Overview

The goal of this project is to preprocess gene expression data, perform dimensionality reduction techniques (PCA, t-SNE, and UMAP), and visualize the results. The code is structured to:

1. **Load the gene expression data** from a tab-separated file.
2. **Normalize the data** by converting raw counts into normalized values.
3. **Perform PCA, t-SNE, and UMAP** on the normalized data.
4. **Visualize** the results of the dimensionality reduction in 2D plots.
5. **Optionally save the resulting plots** as image files (PNG, JPEG, etc.).

## Requirements

- Python 3.x
- Required Python packages:
  - pandas
  - numpy
  - scikit-learn
  - umap-learn
  - matplotlib

You can install the necessary dependencies using `pip`:

```bash
pip install pandas numpy scikit-learn umap-learn matplotlib
```

## Project Structure
```
.
├── gene-expression-analysis.py   # Main script for data processing and visualization
└── GSE268408_FullTable_RawCounts_DU145-SOX11overexpression_dataset.txt  # Sample gene expression data file
```
## How to Run

Run the Python script using the following command:

```bash
python gene-expression-analysis.py
```

## Output
The script will generate 2D scatter plots showing the results of PCA, t-SNE, and UMAP. If a save_path is specified, the plot will be saved as an image file in the given location. Otherwise, the plot will be displayed using matplotlib.