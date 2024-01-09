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

def complete_linkage(data, k):
    clusters = [[i] for i in range(len(data))]
    while len(clusters) > k:
        min_distance = float('inf')
        merge_i, merge_j = 0, 0
        for i, cluster1 in enumerate(clusters):
            for j, cluster2 in enumerate(clusters):
                if i < j:
                    max_distance = 0
                    for p1 in cluster1:
                        for p2 in cluster2:
                            cosine_similarity = np.dot(data[p1], data[p2]) / (np.linalg.norm(data[p1]) * np.linalg.norm(data[p2]))
                            distance = 1 - cosine_similarity
                            max_distance = max(max_distance, distance)
                    if max_distance < min_distance:
                        min_distance = max_distance
                        merge_i, merge_j = i, j
        clusters[merge_i].extend(clusters[merge_j])
        del clusters[merge_j]
    return clusters



# Complete-linkage
complete_link_clusters = complete_linkage(data_matrix, 2)

# Визуализация
fig, axes = plt.subplots(1, 1, figsize=(8, 6))
colors = ['r', 'g']

# Complete-linkage
for i, cluster_indices in enumerate(complete_link_clusters):
    cluster_points = data_matrix[cluster_indices]
    axes.scatter(cluster_points[:, 0], cluster_points[:, 1], color=colors[i], label=f'Cluster {i + 1}')

axes.set_title('Complete-linkage Clustering')
axes.legend()
plt.show()
