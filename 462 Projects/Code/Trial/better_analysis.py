import pandas as pd
import numpy as np
from sklearn.cluster import KMeans, DBSCAN
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
from scipy.stats import gaussian_kde
import matplotlib.pyplot as plt
from sklearn.metrics import silhouette_score

def analyze_distribution(df, count_column='count'):
    # Prepare data
    X = df[count_column].values.reshape(-1, 1)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # 1. Gaussian Mixture Model
    n_components_range = range(2, 6)
    bic = []
    for n_components in n_components_range:
        gmm = GaussianMixture(n_components=n_components, random_state=42)
        gmm.fit(X_scaled)
        bic.append(gmm.bic(X_scaled))
    
    optimal_components = n_components_range[np.argmin(bic)]
    gmm = GaussianMixture(n_components=optimal_components, random_state=42)
    gmm_labels = gmm.fit_predict(X_scaled)
    
    # 2. DBSCAN for outlier detection
    dbscan = DBSCAN(eps=0.5, min_samples=5)
    dbscan_labels = dbscan.fit_predict(X_scaled)
    outliers = X[dbscan_labels == -1]
    
    # 3. Kernel Density Estimation
    kde = gaussian_kde(X.flatten())
    x_range = np.linspace(X.min(), X.max(), 200)
    density = kde(x_range)
    
    # Visualization
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # Plot 1: Distribution with GMM components
    axes[0, 0].hist(X, bins=30, density=True, alpha=0.6)
    axes[0, 0].plot(x_range, density, 'r-', label='KDE')
    axes[0, 0].set_title('Distribution with KDE')
    axes[0, 0].legend()
    
    # Plot 2: GMM BIC scores
    axes[0, 1].plot(n_components_range, bic, 'bo-')
    axes[0, 1].set_xlabel('Number of components')
    axes[0, 1].set_ylabel('BIC score')
    axes[0, 1].set_title('Model Selection (BIC)')
    
    # Plot 3: DBSCAN results
    axes[1, 0].scatter(range(len(X)), X, c=dbscan_labels, cmap='viridis')
    axes[1, 0].set_title('DBSCAN Clustering')
    
    # Plot 4: Box plot
    axes[1, 1].boxplot(X)
    axes[1, 1].set_title('Box Plot with Outliers')
    
    plt.tight_layout()
    plt.show()
    
    # Calculate optimal boundaries
    gmm_centers = scaler.inverse_transform(gmm.means_).flatten()
    gmm_centers.sort()
    
    # Calculate boundaries using weighted approach
    boundaries = []
    for i in range(len(gmm_centers)-1):
        # Weight the boundary by the covariances of adjacent components
        cov1 = gmm.covariances_[i][0][0]
        cov2 = gmm.covariances_[i+1][0][0]
        w1 = 1/cov1
        w2 = 1/cov2
        boundary = (w1*gmm_centers[i] + w2*gmm_centers[i+1])/(w1 + w2)
        boundaries.append(boundary)
    
    # Calculate statistics for each segment
    segments = pd.cut(df[count_column], 
                     bins=[-np.inf] + list(boundaries) + [np.inf],
                     labels=['y'+str(i) for i in range(len(boundaries)+1)])
    
    segment_stats = df.groupby(segments)[count_column].agg(['count', 'mean', 'std'])
    
    return {
        'optimal_components': optimal_components,
        'centers': gmm_centers,
        'boundaries': boundaries,
        'outliers': outliers,
        'segment_stats': segment_stats
    }

# Load and analyze data
df = pd.read_csv('new_dataset.csv')
results = analyze_distribution(df)

print("Analysis Results:")
print(f"Optimal number of components: {results['optimal_components']}")
print("\nComponent centers:")
for i, center in enumerate(results['centers']):
    print(f"Component {i}: {center:.2f}")

print("\nRecommended boundaries:")
for i, boundary in enumerate(results['boundaries']):
    print(f"Between components {i} and {i+1}: {boundary:.2f}")

print("\nSegment Statistics:")
print(results['segment_stats'])

print("\nNumber of detected outliers:", len(results['outliers']))