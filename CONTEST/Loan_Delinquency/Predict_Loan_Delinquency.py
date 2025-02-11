import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
from sklearn.preprocessing import OneHotEncoder
from xgboost import XGBClassifier
import warnings
warnings.filterwarnings(action='ignore')

train_df=pd.read_csv('./train.csv').drop(columns=['UID'])
test_df=pd.read_csv('./test.csv').drop(columns=['UID'])

categorical_col = [
    '주거 형태',
    '현재 직장 근속 연수',
    '대출 목적',
    '대출 상환 기간'
]

encoder=OneHotEncoder(sparse_output=False,handle_unknown='ignore')

encoder.fit(train_df[categorical_col])

train_encoded=encoder.transform(train_df[categorical_col])
test_encoded=encoder.transform(test_df[categorical_col])

train_encoded_df=pd.DataFrame(train_encoded,
                              columns=encoder.get_feature_names_out(categorical_col),
                              )
test_encoded_df=pd.DataFrame(test_encoded,
                              columns=encoder.get_feature_names_out(categorical_col),
                              )

train_df=pd.concat([train_df.drop(columns=categorical_col).reset_index(drop=True),train_encoded_df],axis=1)
test_df=pd.concat([test_df.drop(columns=categorical_col).reset_index(drop=True),test_encoded_df],axis=1)


X_train,X_val,y_train,y_val=train_test_split(
  train_df.drop(columns=["채무 불이행 여부"]),
  train_df['채무 불이행 여부'],
  test_size=0.2,
  random_state=42
)
model=XGBClassifier(
  n_estimators=100,
  max_depth=5,
  learning_rate=0.15,
  random_state=42,
  use_label_encoder=False,
  eval_metric="auc"
)
eval_set=[(X_train,y_train),(X_val,y_val)]


model.fit(
  X_train,y_train,
  eval_set=eval_set,
  verbose=True,
)

preds = model.predict_proba(test_df)[:,1]

submit = pd.read_csv('./sample_submission.csv')


submit['채무 불이행 확률'] = preds
submit.to_csv('./submission.csv', encoding='UTF-8-sig', index=False)