"""
This module performs gene expression analysis using PCA, t-SNE, and UMAP for dimensionality reduction and visualization.
The script processes a raw count dataset with gene identifiers as rows and sample names as columns, performing the following steps:

1. Loading the dataset
2. Normalizing the data
3. Performing PCA for dimensionality reduction
4. Using t-SNE for non-linear dimensionality reduction
5. Using UMAP for further dimensionality reduction
6. Visualizing the results using matplotlib

Requirements:
- pandas
- sklearn
- umap-learn
- matplotlib
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import umap

def load_data(file_path):
    """
    Loads the gene expression dataset from a given file path.
    
    Args:
    - file_path (str): Path to the dataset file.
    
    Returns:
    - pandas.DataFrame: DataFrame containing the gene expression data.
    """
    data = pd.read_csv(file_path, sep='\t', index_col=0)
    return data

def normalize_data(data):
    """
    Normalizes the gene expression data using log transformation and Z-score normalization.
    
    Args:
    - data (pandas.DataFrame): Raw gene expression data.
    
    Returns:
    - pandas.DataFrame: Normalized gene expression data.
    """
    # Log-transform and Z-score normalization
    data_log = np.log1p(data)  # Log transformation
    data_normalized = (data_log - data_log.mean(axis=0)) / data_log.std(axis=0)  # Z-score normalization
    return data_normalized

def perform_pca(data, n_components=2):
    """
    Performs PCA on the normalized data and returns the reduced dimensions.
    
    Args:
    - data (pandas.DataFrame): Normalized gene expression data.
    - n_components (int): Number of principal components to return.
    
    Returns:
    - pandas.DataFrame: Reduced data with principal components.
    """
    pca = PCA(n_components=n_components)
    pca_result = pca.fit_transform(data)
    pca_df = pd.DataFrame(pca_result, columns=[f'PC{i+1}' for i in range(n_components)])
    return pca_df

def perform_tsne(data, n_components=2, perplexity=30, learning_rate=200):
    """
    Performs t-SNE on the normalized data and returns the reduced dimensions.
    
    Args:
    - data (pandas.DataFrame): Normalized gene expression data.
    - n_components (int): Number of t-SNE components to return.
    - perplexity (float): t-SNE perplexity parameter.
    - learning_rate (float): t-SNE learning rate parameter.
    
    Returns:
    - pandas.DataFrame: Reduced data with t-SNE components.
    """
    tsne = TSNE(n_components=n_components, perplexity=perplexity, learning_rate=learning_rate)
    tsne_result = tsne.fit_transform(data)
    tsne_df = pd.DataFrame(tsne_result, columns=[f'tSNE{i+1}' for i in range(n_components)])
    return tsne_df

def perform_umap(data, n_components=2):
    """
    Performs UMAP on the normalized data and returns the reduced dimensions.
    
    Args:
    - data (pandas.DataFrame): Normalized gene expression data.
    - n_components (int): Number of UMAP components to return.
    
    Returns:
    - pandas.DataFrame: Reduced data with UMAP components.
    """
    umap_model = umap.UMAP(n_components=n_components)
    umap_result = umap_model.fit_transform(data)
    umap_df = pd.DataFrame(umap_result, columns=[f'UMAP{i+1}' for i in range(n_components)])
    return umap_df

def plot_results(pca_df, tsne_df, umap_df, title="Dimensionality Reduction Results", save_path=None):
    """
    Generates 2D scatter plots for PCA, t-SNE, and UMAP results.
    
    Args:
    pca_df (pandas.DataFrame): PCA results (2D).
    tsne_df (pandas.DataFrame): t-SNE results (2D).
    umap_df (pandas.DataFrame): UMAP results (2D).
    title (str): Title for the plot (default is "Dimensionality Reduction Results").
    save_path (str or None): Path to save the image. If `None`, the plot will be displayed instead of saved.
    """
    fig, axs = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle(title)

    # PCA plot
    axs[0].scatter(pca_df.iloc[:, 0], pca_df.iloc[:, 1], c='blue', label="PCA")
    axs[0].set_title("PCA")
    axs[0].set_xlabel("PC1")
    axs[0].set_ylabel("PC2")

    # t-SNE plot
    axs[1].scatter(tsne_df.iloc[:, 0], tsne_df.iloc[:, 1], c='green', label="t-SNE")
    axs[1].set_title("t-SNE")
    axs[1].set_xlabel("t-SNE 1")
    axs[1].set_ylabel("t-SNE 2")

    # UMAP plot
    axs[2].scatter(umap_df.iloc[:, 0], umap_df.iloc[:, 1], c='red', label="UMAP")
    axs[2].set_title("UMAP")
    axs[2].set_xlabel("UMAP 1")
    axs[2].set_ylabel("UMAP 2")

    # Adjust layout and show the plot
    plt.tight_layout()
    plt.subplots_adjust(top=0.85)

    # Save the plot if a save path is provided, otherwise show the plot
    if save_path:
        plt.savefig(save_path, format="png")
    else:
        plt.show()

    plt.close()

def main():
    """
    Main function to execute the workflow of loading the dataset, performing dimensionality reduction, and visualizing the results.
    """
    # Load the data
    file_path = "data/GSE268408_FullTable_RawCounts_DU145-SOX11overexpression_dataset.txt"
    data = load_data(file_path)
    
    # Normalize the data
    normalized_data = normalize_data(data)
    
    # Perform dimensionality reductions
    pca_df = perform_pca(normalized_data)
    tsne_df = perform_tsne(normalized_data)
    umap_df = perform_umap(normalized_data)
    
    # Plot the results
    plot_results(pca_df, tsne_df, umap_df, save_path="dimensionality_reduction_results.png")

if __name__ == "__main__":
    main()
