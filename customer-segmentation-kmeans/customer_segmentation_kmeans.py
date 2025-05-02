import kaggle
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

# Step 1: Download the dataset using Kaggle API
# (Assuming kaggle.json is set up correctly in the ~/.kaggle/ folder)

dataset = 'jessemostipak/customer-segmentation'
kaggle.api.dataset_download_files(dataset, path='.', unzip=True)

# Step 2: Load the dataset into a pandas dataframe
df = pd.read_csv("customer-segmentation.csv")

# Step 3: Preprocess the data
df = df.drop(columns=['CustomerID'])
df['Genre'] = df['Genre'].map({'Female': 0, 'Male': 1})

# Scale the features
scaler = StandardScaler()
df_scaled = scaler.fit_transform(df)

# Step 4: Use the Elbow Method to determine the optimal number of clusters
inertia = []
for k in range(1, 11):
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(df_scaled)
    inertia.append(kmeans.inertia_)

# Plot the Elbow Method graph
plt.figure(figsize=(8, 6))
plt.plot(range(1, 11), inertia, marker='o')
plt.title('Elbow Method For Optimal k')
plt.xlabel('Number of clusters')
plt.ylabel('Inertia')
plt.show()

# Step 5: Apply KMeans clustering with the chosen number of clusters
kmeans = KMeans(n_clusters=5, random_state=42)
df['Cluster'] = kmeans.fit_predict(df_scaled)

# Step 6: Visualize the clusters
plt.figure(figsize=(8, 6))
plt.scatter(df['Annual Income (k$)'], df['Spending Score (1-100)'], c=df['Cluster'], cmap='viridis')
plt.title('Customer Segments')
plt.xlabel('Annual Income (k$)')
plt.ylabel('Spending Score (1-100)')
plt.show()
