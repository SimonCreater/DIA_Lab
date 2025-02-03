import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier  
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder,OneHotEncoder
train=pd.read_csv('./train.csv')
train.head()

x=train.drop(['ID','허위매물여부'],axis=1)
y=train['허위매물여부']

mean_imputer=SimpleImputer(strategy='mean')

columns_fill_mean=['해당층','총층','전용면적','욕실수','총주차대수']

x[columns_fill_mean]=mean_imputer.fit_transform(x[columns_fill_mean])

label_encode_cols=['중개사무소','게재일','제공플랫폼','방향']

label_encoders={}
for col in label_encode_cols:
  le=LabelEncoder()
  x[col]=le.fit_transform(x[col].astype(str))
  label_encoders[col]=le

one_hot_cols=['매물확인방식','주차가능여부']

one_hot_encoder=OneHotEncoder(sparse_output=False,handle_unknown='ignore')

x_encoded=one_hot_encoder.fit_transform(x[one_hot_cols])
x_encoded_df=pd.DataFrame(x_encoded,##NUMpy 배열
                          columns=one_hot_encoder.get_feature_names_out(one_hot_cols),##원-핫 인코딩된 컬럼 이름
                          index=x.index)

x=pd.concat([x.drop(columns=one_hot_cols),x_encoded_df],axis=1)
print(x_encoded_df)

model=RandomForestClassifier(n_estimators=100,
                             criterion='gini',
                             max_depth=None,
                             random_state=42)
model.fit(x,y)

test=pd.read_csv('./test.csv')
test[columns_fill_mean]=mean_imputer.transform(test[columns_fill_mean])

for col in label_encode_cols:
  if col in test.columns:
    le=label_encoders[col]
    test[col]=test[col].astype(str)
    unseen=set(test[col].unique())-set(le.classes_)
    if unseen:
      le.classes_=np.concatenate([le.classes_,np.array(list(unseen))])
    test[col]=le.transform(test[col])

test_encoded=one_hot_encoder.transform(test[one_hot_cols])
test_encoded_df=pd.DataFrame(test_encoded,
                             columns=one_hot_encoder.get_feature_names_out(one_hot_cols),
                             index=test.index)
test=pd.concat([test.drop(columns=one_hot_cols),test_encoded_df],axis=1)

test.drop(columns=['ID'],inplace=True)

pred = pd.Series(model.predict(test))
submission_template_path = "./sample_submission.csv" 
submission_df = pd.read_csv(submission_template_path)


submission_df['ID'] = submission_df['ID']



submission_df['허위매물여부'] = pred.values


submission_df.to_csv("submission.csv", index=False)


print(submission_df.head())

