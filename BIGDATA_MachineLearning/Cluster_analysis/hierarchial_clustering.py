from sklearn.datasets import make_blobs
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import pandas as pd
import numpy as np
from scipy.spatial.distance import pdist,squareform
from scipy.cluster.hierarchy import linkage,dendrogram

np.random.seed(123)

variables=['X','Y','Z']
labels=['ID_0','ID_1','ID_2','ID_3','ID_4']

X=np.random.random_sample([5,3])*10
df=pd.DataFrame(X,columns=variables,index=labels)

row_dist=pd.DataFrame(squareform(
  pdist(df,metric='euclidean')),columns=labels,index=labels)


row_clusters=linkage(pdist(df,
                           metric='euclidean',
                           ),method='complete')


pd.DataFrame(row_clusters,
             columns=['row label 1',
                      'row label 2',
                      'distance',
                      'no. of items in clust. '],
                      index=['cluster %d %(i+1)' for i in range(row_clusters.shape[0])])
row_dendr=dendrogram(row_clusters,
                     labels=labels,
                     )

plt.tight_layout()
plt.ylabel('Euclidean distance')
plt.show()