import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt 
from matplotlib.colors import ListedColormap
from sklearn.decomposition import PCA
from sklearn.datasets import make_moons,make_circles
from scipy.spatial.distance import pdist,squareform
from numpy import exp
from scipy.linalg import eigh
X,y=make_moons(n_samples=100,random_state=123)
scikit_pca=PCA(n_components=2)
X_spca=scikit_pca.fit_transform(X)
fig,ax=plt.subplots(nrows=1,ncols=2,figsize=(7,3))

print(X_spca)
# fig,ax=plt.subplots(nrows=1,ncols=2,figsize=(7,3))

# ax[0].scatter(X_spca[y == 0, 0], X_spca[y == 0, 1],
#               color='red', marker='^', alpha=0.5)
# ax[0].scatter(X_spca[y == 1, 0], X_spca[y == 1, 1],
#               color='blue', marker='o', alpha=0.5)

# ax[1].scatter(X_spca[y == 0, 0], np.zeros((50, 1)) + 0.02,
#               color='red', marker='^', alpha=0.5)
# ax[1].scatter(X_spca[y == 1, 0], np.zeros((50, 1)) - 0.02,
#               color='blue', marker='o', alpha=0.5)

# ax[0].set_xlabel('PC1')
# ax[0].set_ylabel('PC2')
# ax[1].set_ylim([-1, 1])
# ax[1].set_yticks([])
# ax[1].set_xlabel('PC1')

# plt.tight_layout()
# plt.show()
def rbf_kernel_pca(X,gamma,n_components):
  sq_dists=pdist(X,'sqeuclidean')

  mat_sq_dists=squareform(sq_dists)
  K=exp(-gamma*mat_sq_dists)
  N=K.shape[0]

  one_n=np.ones((N,N))
  K=K-one_n.dot(K)-K.dot(one_n)+one_n.dot(K).dot(one_n)

  eigvals,eigvecs=eigh(K)
  eigvals,eigvecs=eigvals[::-1],eigvecs[::-1]

  X_pc=np.column_stack([eigvecs[:, i]for i in range(n_components)])

X,y=make_circles(n_samples=100,random_state=123)
scikit_pca=PCA(n_components=2)
X_spca=scikit_pca.fit_transform(X)
fig,ax=plt.subplots(nrows=1,ncols=2,figsize=(7,3))
ax[0].scattter(X_spca[y==0, 0],X_spca[y==0, 1],
               color='red',marker='^',alpha=0.5)
sc=StandardScaler()
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, stratify=y, random_state=0)

X_train_std=sc.fit_transform(X_train) 
X_test_std=sc.transform(X_test)
cov_mat=np.cov(X_test_std.T)
eigen_vals,eigen_vecs=np.linalg.eig(cov_mat)


mean_vecs=[]
for label in range(1,4):
  mean_vecs.append(np.mean(X_train_std[y_train==label],axis=0))

d=13
S_W=np.zeros((d,d))
for label,mv in zip(range(1,4),mean_vecs):
  class_scatter=np.zeros((d,d))
  for row in X_test_std[y_train==label]:
   row,mv=row.reshape(d,1),mv.reshape(d,1)
   class_scatter+=(row-mv).dot((row-mv).T)
  S_W+=class_scatter 


mean_overall=np.mean(X_train_std,axis=0)
mean_overall=mean_overall.reshape(d,1)

S_B=np.zeros((d,d))
for i,mean_vec in enumerate(mean_vecs):
  n=X_train[y_train==i+1, :].shape[0]
  mean_vec=mean_vec.reshape(d,1)
  S_B+=n*(mean_vec-mean_overall).dot((mean_vec-mean_overall).T)