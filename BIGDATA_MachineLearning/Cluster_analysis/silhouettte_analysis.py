from sklearn.datasets import make_blobs
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

import numpy as np
from matplotlib import cm
from sklearn.metrics import silhouette_samples

X,y=make_blobs(n_samples=150,
               n_features=2,
               centers=3,
               cluster_std=0.5,
               shuffle=True,
               random_state=0)

km = KMeans(n_clusters=3,
            init='k-means++',
            n_init=10,
            max_iter=300,
            tol=1e-4,
            random_state=0)

y_km = km.fit_predict(X)
cluster_labels = np.unique(y_km)
n_clusters = cluster_labels.shape[0]
silhouette_vals = silhouette_samples(X, y_km, metric='euclidean')


y_ax_lower, y_ax_upper = 0, 0
yticks = []

plt.figure(figsize=(8, 6))

for i, c in enumerate(cluster_labels):
    c_silhouette_vals = silhouette_vals[y_km == c]
    c_silhouette_vals.sort()
    
    y_ax_upper += len(c_silhouette_vals)
    color = cm.jet(float(i) / n_clusters)
    
    plt.barh(range(y_ax_lower, y_ax_upper),
             c_silhouette_vals,
             height=1.0,
             edgecolor='none',
             color=color)
    
    yticks.append((y_ax_lower + y_ax_upper) / 2)
    y_ax_lower += len(c_silhouette_vals)

plt.axvline(x=np.mean(silhouette_vals), color="red", linestyle="--")
plt.yticks(yticks, cluster_labels)
plt.xlabel("Silhouette coefficient values")
plt.ylabel("Cluster")
plt.title("Silhouette plot for K-Means clustering")
plt.show()


