import numpy as np
import random
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

def init_board_gauss(N, k):
    n = float(N)/k
    X = []
    for i in range(k):
        c = (random.uniform(-1, 1), random.uniform(-1, 1))
        s = random.uniform(0.05,0.5)
        x = []
        while len(x) < n:
            a, b = np.array([np.random.normal(c[0], s), np.random.normal(c[1], s)])
            # Continue drawing points from the distribution in the range [-1,1]
            if abs(a) < 1 and abs(b) < 1:
                x.append([a,b])
        X.extend(x)
    X = np.array(X)[:N]
    return X

X = init_board_gauss(200,3)

kmeans = KMeans(n_clusters=3)
kmeans = kmeans.fit(X)
labels = kmeans.predict(X)
centroids = kmeans.cluster_centers_
tran = kmeans.transform(X)

def get_top_points_idx(cluster_distance_matrix, labels, top_n):
    orig_dataset_idx = []
    k = cluster_distance_matrix.shape[1]
    for i in range(k):
        cluster_dataset_idx = np.where(labels==i)[0]
        cluster_distance = cluster_distance_matrix[cluster_dataset_idx, i]
        cluster_distance_sort_top_n = np.argsort(cluster_distance)[:top_n]
        orig_dataset_idx.append(cluster_dataset_idx[cluster_distance_sort_top_n])
    return orig_dataset_idx

def get_top_data_points(orig_dataset, get_top_points_idx):
    n_col = orig_dataset.shape[1]
    top_data_points = np.empty(shape=(0,n_col))
    for i in range(len(get_top_points_idx)):
        top_data_points = np.vstack((top_data_points, orig_dataset[get_top_points_idx[i], :]))
    return top_data_points
        
get_top_points_idx = get_top_points_idx(tran, labels, 5)
top_data_points = get_top_data_points(X, get_top_points_idx)


plt.scatter(X[:,0], X[:,1], c=labels)
plt.scatter(centroids[:,0],centroids[:,1], marker='*',c='#050505', s=500)
plt.scatter(top_data_points[:,0], top_data_points[:,1], marker='+',c='#ff0000')
plt.show()
