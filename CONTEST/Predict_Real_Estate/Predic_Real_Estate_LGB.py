import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split,StratifiedKFold
from sklearn.metrics import f1_score, classification_report, confusion_matrix

from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import OneHotEncoder,LabelEncoder
from sklearn.impute import KNNImputer
import lightgbm as lgb
train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')

columns_fill_knn = ['해당층', '총층', '전용면적', '방수', '욕실수', '총주차대수']
imputer=KNNImputer(n_neighbors=5)

train[columns_fill_knn]=imputer.fit_transform(train[columns_fill_knn])
test[columns_fill_knn]=imputer.transform(test[columns_fill_knn])


label_encode_cols = ['중개사무소', '게재일', '제공플랫폼', '방향']


for col in label_encode_cols:
  le=LabelEncoder()
  combined_data=pd.concat([train[col],test[col]],axis=0).astype(str)
  le.fit(combined_data)
  train[col]=le.transform(train[col].astype(str))
  test[col]=le.transform(test[col].astype(str))

one_hot_cols=['매물확인방식','주차가능여부']
one_hot_encoder=OneHotEncoder(sparse_output=False,handle_unknown='ignore')

train_encoded=OneHotEncoder.fit_transform(train[one_hot_cols])
test_encoded=OneHotEncoder.transform(test[one_hot_cols])

train=pd.concat([train.drop(columns=one_hot_cols),
                 pd.DataFrame(train_encoded,index=train.index)],axis=1)


test=pd.concat([test.drop(columns=one_hot_cols),
                 pd.DataFrame(train_encoded,index=train.index)],axis=1)

train = train.drop(columns=['ID'])
test_id = test['ID']
test = test.drop(columns=['ID'])

X=train.drop(columns=['허위매물여부'])
y=train['허위매물여부']

