import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import warnings

warnings.filterwarnings('ignore')

df = pd.read_csv("dataset.csv")
X = df[['X', 'Y']].values

inertias = []
for k in range(1, 6):
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    kmeans.fit(X)
    inertias.append(kmeans.inertia_)

plt.figure(figsize=(8, 5))
plt.plot(range(1, 6), inertias, marker='o')
plt.title("Elbow Curve")
plt.xlabel("Number of Clusters (K)")
plt.ylabel("Inertia")
plt.show()

optimal_k = 3
np.random.seed(42)
centroids = X[np.random.choice(len(X), optimal_k, replace=False)].astype(float)

plt.figure(figsize=(8, 5))
plt.scatter(X[:, 0], X[:, 1], c='gray', label='Data Points')
plt.scatter(centroids[:, 0], centroids[:, 1], c='red', marker='X', s=200, label='Initial Centroids')
plt.title("Initial Centroids and Data Points")
plt.legend()
plt.show()

iteration = 1
while True:
    distances = np.linalg.norm(X[:, np.newaxis] - centroids, axis=2)
    labels = np.argmin(distances, axis=1)
    new_centroids = np.array([X[labels == i].mean(axis=0) for i in range(optimal_k)])
    
    plt.figure(figsize=(8, 5))
    plt.scatter(X[:, 0], X[:, 1], c=labels, cmap='viridis', label='Data Points')
    plt.scatter(new_centroids[:, 0], new_centroids[:, 1], c='red', marker='X', s=200, label='Updated Centroids')
    plt.title(f"Iteration {iteration}: Updated Centroids and Assignments")
    plt.legend()
    plt.show()
    
    if np.allclose(centroids, new_centroids):
        break
        
    centroids = new_centroids
    iteration += 1

final_inertia = sum(np.sum((X[labels == i] - centroids[i]) ** 2) for i in range(optimal_k))

print("Final cluster labels:\n", labels)
print("\nFinal centroids:\n", centroids)
print("\nFinal inertia:\n", final_inertia)
