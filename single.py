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

# Преобразование векторов координат в матрицу
data_matrix = np.array(list(points.values()))

def max_similarity(data, k):
    clusters = [[i] for i in range(len(data))]
    while len(clusters) > k:
        max_similarity = 0
        merge_i, merge_j = 0, 0
        for i, cluster1 in enumerate(clusters):
            for j, cluster2 in enumerate(clusters):
                if i < j:
                    similarity = max([np.dot(data[p1], data[p2]) / (np.linalg.norm(data[p1]) * np.linalg.norm(data[p2])) for p1 in cluster1 for p2 in cluster2])
                    if similarity > max_similarity:
                        max_similarity = similarity
                        merge_i, merge_j = i, j
        clusters[merge_i].extend(clusters[merge_j])
        del clusters[merge_j]
    return clusters

# Single-linkage
single_link_clusters = max_similarity(data_matrix, 2)

# Визуализация
fig, axes = plt.subplots(1, 1, figsize=(8, 6))
colors = ['r', 'g']

# Single-linkage
for i, cluster_indices in enumerate(single_link_clusters):
    cluster_points = data_matrix[cluster_indices]
    axes.scatter(cluster_points[:, 0], cluster_points[:, 1], color=colors[i], label=f'Cluster {i + 1}')

axes.set_title('Single-linkage Clustering')
axes.legend()
plt.show()
