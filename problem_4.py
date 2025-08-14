from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

# === 1. Generate random 2D dataset ===
X, y_true = make_blobs(
    n_samples=300,
    centers=3,
    cluster_std=0.60,
    random_state=42
)

# === 2. Apply KMeans with 3 clusters ===
kmeans = KMeans(n_clusters=3, random_state=42)
kmeans.fit(X)
y_kmeans = kmeans.predict(X)

# === 3. Plot results ===
plt.figure(figsize=(8, 6))
plt.scatter(
    X[:, 0], X[:, 1],
    c=y_kmeans, cmap="viridis",
    s=50
)
# Plot cluster centers
centers = kmeans.cluster_centers_
plt.scatter(
    centers[:, 0], centers[:, 1],
    c="red", s=200, alpha=0.75, marker="X", label="Centroids"
)
plt.title("K-Means Clustering with 3 Clusters")
plt.legend()
plt.savefig("outputs/kmeans_clusters_problem_4.png")
plt.close()

print("K-means clustering plot saved as 'kmeans_clusters_problem_4.png' in the outputs folder")