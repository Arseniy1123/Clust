import numpy as np
import matplotlib.pyplot as plt

# Координаты точек
points = {
    'a': np.array([0.6, 1.9]),
    'b': np.array([1.8, 1.6]),
    'c': np.array([2.7, 2.0]),
    'd': np.array([3.0, 2.1]),
    'e': np.array([3.0, 2.6]),
    'f': np.array([3.1, 4.5]),
    'g': np.array([3.8, 0.6]),
    'h': np.array([4.2, 2.7])
}

# Исходные точки для K-means
initial_centers = np.array([points['f'], points['g']])

data_matrix = np.array(list(points.values()))

def cosine_similarity(x, y):
    return np.dot(x, y) / (np.linalg.norm(x) * np.linalg.norm(y))

def assign_to_clusters(data, centers):
    clusters = {}
    for point in data:
        distances = [cosine_similarity(point, center) for center in centers]
        cluster_index = np.argmin(distances)
        if cluster_index not in clusters:
            clusters[cluster_index] = []
        clusters[cluster_index].append(point)
    return clusters

def update_centers(clusters):
    new_centers = []
    for cluster_points in clusters.values():
        new_center = np.mean(cluster_points, axis=0)
        new_centers.append(new_center)
    return np.array(new_centers)

# K-means
centers = initial_centers.copy()
for _ in range(10):  # Количество итераций
    clusters = assign_to_clusters(data_matrix, centers)
    centers = update_centers(clusters)

# Визуализация
fig, axes = plt.subplots(1, 1, figsize=(8, 6))
colors = ['r', 'g', 'b']

# K-means
for i, (cluster_index, cluster_points) in enumerate(clusters.items()):
    cluster_points = np.array(cluster_points)
    axes.scatter(cluster_points[:, 0], cluster_points[:, 1], color=colors[i], label=f'Cluster {i + 1}')

axes.scatter(centers[:, 0], centers[:, 1], color='black', marker='x', label='Centroids')
axes.set_title('K-means Clustering')
axes.legend()
plt.show()
