import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt 
from matplotlib.colors import ListedColormap
from scipy.spatial.distance import pdist,squareform
from numpy import exp
from scipy.linalg import eigh
df_wine = pd.read_csv('https://archive.ics.uci.edu/'
                      'ml/machine-learning-databases/wine/wine.data',
                      header=None)


X, y = df_wine.iloc[:, 1:].values, df_wine.iloc[:, 0].values

np.set_printoptions(precision=4)
mean_vecs=[]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, stratify=y, random_state=0)
sc = StandardScaler()
X_train_std = sc.fit_transform(X_train)
X_test_std = sc.transform(X_test)
print(X_train_std)
for label in range(1,4):
  mean_vecs.append(np.mean(X_train_std[y_train==label],axis=0))
print(y_train)

d=13
S_W=np.zeros((d,d))
for label,mv in zip(range(1,4),mean_vecs):
  class_scatter=np.zeros((d,d))
  for row in X_train_std[y_train==label]:
    row,mv=row.reshape(d,1),mv.reshape(d,1)
    class_scatter+=(row-mv).dot((row-mv).T)
  S_W+=class_scatter
mean_overall=np.mean(X_train_std)
print(mean_overall)
mean_overall=mean_overall.reshape(d,1)



S_B=np.zeros((d,d))
for i,mean_vec in enumerate(mean_vecs):
  n=X_train[y_train==i+1, :].shape[0]
  mean_vec=mean_vec.reshape(d,1)
  S_B+=n*(mean_vec-mean_overall).dot((mean_vec-mean_overall).T)

def rbf_kernel_pca(X,gamma,n_components):
  sq_dists=pdist(X,'sqeuclidean')
  mat_sq_dists=squareform(sq_dists)

  K=exp(-gamma*mat_sq_dists)

  N=K.shape[0]
  one_n=np.ones((N,N))/N
  K = K - one_n.dot(K) - K.dot(one_n) + one_n.dot(K).dot(one_n)
  eigvals, eigvecs = eigh(K)
  eigvals, eigvecs = eigvals[::-1], eigvecs[:, ::-1]

  X_pc = np.column_stack([eigvecs[:, i]
                            for i in range(n_components)])

  return X_pc




