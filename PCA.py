import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt 
from matplotlib.colors import ListedColormap
from sklearn.decomposition import PCA
def plot_decision_regions(X,y,classifier,resolution=0.02):
    markers=('s','x','o','^','v')
    colors=('red','blue','lightgreen','gray','cyan')
    cmap = ListedColormap(colors[:len(np.unique(y))])
    x1_min, x1_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    x2_min, x2_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, resolution),
                           np.arange(x2_min, x2_max, resolution))
    Z = classifier.predict(np.array([xx1.ravel(), xx2.ravel()]).T)
    Z = Z.reshape(xx1.shape)
    plt.contourf(xx1, xx2, Z, alpha=0.4, cmap=cmap)
    plt.xlim(xx1.min(), xx1.max())
    plt.ylim(xx2.min(), xx2.max())

    # 클래스별로 샘플을 그립니다
    for idx, cl in enumerate(np.unique(y)):
        plt.scatter(x=X[y == cl, 0], 
                    y=X[y == cl, 1],
                    alpha=0.6, 
                    color=cmap(idx),
                    edgecolor='black',
                    marker=markers[idx], 
                    label=cl)

df_wine = pd.read_csv('https://archive.ics.uci.edu/'
                      'ml/machine-learning-databases/wine/wine.data',
                      header=None)


X, y = df_wine.iloc[:, 1:].values, df_wine.iloc[:, 0].values


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, stratify=y, random_state=0)


sc = StandardScaler()
X_train_std = sc.fit_transform(X_train)
X_test_std = sc.transform(X_test)


cov_mat = np.cov(X_train_std.T)

# 고유값과 고유벡터 계산
eigen_vals, eigen_vecs = np.linalg.eig(cov_mat)
print("고유값:\n", eigen_vals)
print("고유벡터:\n",eigen_vecs)


# plt.bar(range(1, len(eigen_vals) + 1), eigen_vals, alpha=0.7, align='center')
# plt.title("고유값 분포")
# plt.xlabel("고유값 인덱스")
# plt.ylabel("고유값 크기")
# plt.show()

# 주석 처리된 시각화 코드 보완
# 시각화를 위한 설정
colors = ['r', 'b', 'g']
markers = ['s', 'x', 'o']
# for l, c, m in zip(np.unique(y_train), colors, markers):
#     plt.scatter(X_train_std[y_train == l, 0], 
#                 X_train_std[y_train == l, 1], 
#                 c=c, label=f"Class {l}", marker=m)
# plt.title("Wine Dataset Visualization (First Two Features)")
# plt.xlabel("Standardized Feature 1")
# plt.ylabel("Standardized Feature 2")
# plt.legend()
# plt.show()

tot = sum(eigen_vals)
var_exp=[(i/tot) for i in sorted(eigen_vals,reverse=True)]
cum_var_exp = np.cumsum(var_exp)
eigen_pairs=[(np.abs(eigen_vals[i]),eigen_vecs[:, i])
             for i in range(len(eigen_vals))]
eigen_pairs.sort(key=lambda k:k[0],reverse=True)

w = np.hstack((eigen_pairs[0][1][:, np.newaxis],
               eigen_pairs[1][1][:, np.newaxis]))
print('투영 행렬 W:\n', w)
X_train_pca = X_train_std.dot(w)
colors = ['r', 'b', 'g']
markers = ['s', 'x', 'o']

for l, c, m in zip(np.unique(y_train), colors, markers):
    plt.scatter(X_train_pca[y_train == l, 0], 
                X_train_pca[y_train == l, 1], 
                c=c, label=l, marker=m)

plt.xlabel('PC 1')
plt.ylabel('PC 2')
plt.legend(loc='lower left')
plt.tight_layout()
# plt.savefig('images/05_03.png', dpi=300)
plt.show()
np.set_printoptions(precision=4)
mean_vecs=[]
for label in range(1,4):
    mean_vecs.append(np.mean(X_train_std[y_train==label]))

d=13
S_W=np.zeros((d,d))
for label, mv in zip(range(1,4), mean_vecs):
    class_scatter=np.zeros((d,d))
    for row in X_train_std[y_train==label]:
        row,mv=row.reshape(d,1),mv.rehape(d,1)
        class_scatter+=(row-mv).dot((row-mv).T)
    S_W+=class_scatter

mean_overall=np.mean(X_train_std,axis=0)
mean_overall=mean_overall.reshape(d,1)
S_B=np.zeros((d,d))
for i,mean_vec in enumerate(mean_vecs):
    n=X_train[y_train==i+1, :].shape[0]
    mean_vec=mean_vec.reshape(d,1)
    S_B+=n*(mean_vec-mean_overall).dot((mean_vec-mean_overall).T)
