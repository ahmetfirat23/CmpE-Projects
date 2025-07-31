import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from kneed import KneeLocator

# Load the data
df = pd.read_csv('new_dataset.csv')

# Prepare the data for clustering
X = df['count'].values.reshape(-1, 1)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Find optimal number of clusters using elbow method
inertias = []
K = range(1, 10)

for k in K:
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(X_scaled)
    inertias.append(kmeans.inertia_)

# Use knee locator to find the optimal number of clusters
kn = KneeLocator(K, inertias, curve='convex', direction='decreasing')
optimal_clusters = kn.knee

# Perform final clustering with optimal number of clusters
kmeans = KMeans(n_clusters=optimal_clusters, random_state=42)
df['cluster'] = kmeans.fit_predict(X_scaled)

# Get cluster centers and convert back to original scale
centers = scaler.inverse_transform(kmeans.cluster_centers_)
centers_sorted = np.sort(centers.flatten())

# Calculate boundaries between clusters
boundaries = []
for i in range(len(centers_sorted)-1):
    boundary = (centers_sorted[i] + centers_sorted[i+1]) / 2
    boundaries.append(boundary)

# Print results
print(f"Optimal number of clusters: {optimal_clusters}")
print("\nCluster centers:")
for i, center in enumerate(centers_sorted):
    print(f"Cluster {i}: {center:.2f}")

print("\nRecommended boundaries:")
for i, boundary in enumerate(boundaries):
    print(f"Between clusters {i} and {i+1}: {boundary:.2f}")

# Visualize results
plt.figure(figsize=(12, 6))

# Plot 1: Elbow curve
plt.subplot(1, 2, 1)
plt.plot(K, inertias, 'bx-')
plt.xlabel('k')
plt.ylabel('Inertia')
plt.title('Elbow Method for Optimal k')
plt.vlines(optimal_clusters, plt.ylim()[0], plt.ylim()[1], colors='r', linestyles='dashed')

# Plot 2: Distribution with clusters
plt.subplot(1, 2, 2)
plt.hist(df['count'], bins=30)
for boundary in boundaries:
    plt.axvline(x=boundary, color='r', linestyle='--')
plt.xlabel('Count')
plt.ylabel('Frequency')
plt.title('Distribution with Cluster Boundaries')

plt.tight_layout()
plt.show()