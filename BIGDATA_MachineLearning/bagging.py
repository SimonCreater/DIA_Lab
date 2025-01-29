import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler,LabelEncoder
import matplotlib.pyplot as plt 
from matplotlib.colors import ListedColormap
from scipy.spatial.distance import pdist,squareform
from numpy import exp
from sklearn.ensemble import BaggingClassifier
from sklearn.tree import DecisionTreeClassifier
from scipy.linalg import eigh
df_wine = pd.read_csv('https://archive.ics.uci.edu/'
                      'ml/machine-learning-databases/wine/wine.data',
                      header=None)

df_wine.columns = ['Class label', 'Alcohol', 'Malic acid', 'Ash',
                   'Alcalinity of ash', 'Magnesium', 'Total phenols',
                   'Flavanoids', 'Nonflavanoid phenols', 'Proanthocyanins',
                   'Color intensity', 'Hue', 'OD280/OD315 of diluted wines',
                   'Proline']
df_wine=df_wine[df_wine['Class label']!=1]
y=df_wine['Class label'].values
X = df_wine[['Alcohol', 'OD280/OD315 of diluted wines']].values
print(y)
le=LabelEncoder()
y=le.fit_transform(y)

X_train, X_test, y_train, y_test =\
            train_test_split(X, y, 
                             test_size=0.2, 
                             random_state=1,
                             stratify=y)
tree = DecisionTreeClassifier(criterion='entropy', 
                              max_depth=None,
                              random_state=1)

bag = BaggingClassifier(base_estimator=tree,#결정트리
                        n_estimators=500, #생성할 모델 수
                        max_samples=1.0, #각 기본 모델이 학습에 사용할 비율
                        max_features=1.0, 
                        bootstrap=True, 
                        bootstrap_features=False, #특성은 샘플링 안함
                        n_jobs=1, #cpu개수
                        random_state=1)
