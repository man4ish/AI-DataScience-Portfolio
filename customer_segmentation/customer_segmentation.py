import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans, DBSCAN

# Load dataset
df = pd.read_csv("customers.csv")

# Feature selection
X = df[['Age', 'AnnualIncome', 'SpendingScore']]

# Standardize features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# PCA for visualization
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)
df['PC1'] = X_pca[:, 0]
df['PC2'] = X_pca[:, 1]

# KMeans Clustering
kmeans = KMeans(n_clusters=3, random_state=42)
df['KMeans_Cluster'] = kmeans.fit_predict(X_scaled)

# DBSCAN Clustering
dbscan = DBSCAN(eps=0.8, min_samples=2)
df['DBSCAN_Cluster'] = dbscan.fit_predict(X_scaled)

# Plotting results
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
sns.scatterplot(data=df, x='PC1', y='PC2', hue='KMeans_Cluster', palette='Set2')
plt.title("K-Means Clustering")

plt.subplot(1, 2, 2)
sns.scatterplot(data=df, x='PC1', y='PC2', hue='DBSCAN_Cluster', palette='Set1')
plt.title("DBSCAN Clustering")

plt.tight_layout()
plt.savefig("cluster_results.png")
plt.show()
