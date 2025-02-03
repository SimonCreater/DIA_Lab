from sklearn.datasets import make_moons
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN
X,y=make_moons(n_samples=200,
               noise=0.05,
               random_state=0)
plt.scatter(X[:,0],X[:,1])
plt.tight_layout()
plt.show()

db=DBSCAN(eps=0.2,
          min_samples=5,
          metric='euclidean')
y_db=db.fit_predict(X)

plt.scatter(X[y_db==0, 0],#클러스터 0에 속하는 샘플의 0번째 특징
            X[y_db==0, 1],
            c='lightblue',
            edgecolors='black',
            marker='o',
            s=40,
            label='Cluster 1')
plt.scatter(X[y_db==1,0],
            X[y_db==1,1],
            c='red',
            edgecolors='black',
            marker='s',
            s=40,
            label='Cluster 2')

plt.legend()
plt.tight_layout()
plt.show()