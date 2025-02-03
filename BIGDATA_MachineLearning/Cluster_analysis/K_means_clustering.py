from sklearn.datasets import make_blobs
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
X,y=make_blobs(n_samples=150,
               n_features=2,
               centers=3,
               cluster_std=0.5,
               shuffle=True,
               random_state=0)
plt.scatter(X[:,0],X[:,1],c='white',marker='o',edgecolors='black',s=50)

plt.grid()
plt.tight_layout()

plt.show()

km=KMeans(n_clusters=3,
          init='random',
          n_init=10,
          max_iter=300,
          tol=1e-04,
          random_state=0)
y_km=km.fit_predict(X)


