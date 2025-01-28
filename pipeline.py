import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.svm import SVC
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import train_test_split
from sklearn.metrics import ConfusionMatrixDisplay
df = pd.read_csv('https://archive.ics.uci.edu/ml/'
                 'machine-learning-databases'
                 '/breast-cancer-wisconsin/wdbc.data', header=None)
X=df.loc[:, 2:].values
y=df.loc[:, 1].values
le=LabelEncoder()
y=le.fit_transform(y)

pipe_lr=make_pipeline(StandardScaler(),PCA(n_components=2),LogisticRegression(random_state=1))

pipe_lr.fit()

pipe_svc=make_pipeline(StandardScaler(),SVC(random_state=1))

param_range=[0.0001,0.001,0.01,0.1,1.0,10.0,100.0,1000.0]
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.20,stratify=y,random_state=1)

kfold=StratifiedKFold(n_splits=10).split(X_train,y_train)
scores=[]
for k,(train,test) in  enumerate(X_train,y_train):
  pipe_lr.fit(X_train[train],y_train[train])
  score=pipe_lr.score(X_train[test],y_train[test])

