
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import f1_score, classification_report, confusion_matrix


import seaborn as sns
import matplotlib.pyplot as plt


from imblearn.over_sampling import SMOTE


import lightgbm as lgb

train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')

columns_fill_knn = ['해당층', '총층', '전용면적', '방수', '욕실수', '총주차대수']
from sklearn.impute import KNNImputer
imputer = KNNImputer(n_neighbors=5)
train[columns_fill_knn] = imputer.fit_transform(train[columns_fill_knn])
test[columns_fill_knn] = imputer.transform(test[columns_fill_knn])


from sklearn.preprocessing import LabelEncoder
label_encode_cols = ['중개사무소', '게재일', '제공플랫폼', '방향']
for col in label_encode_cols:
    le = LabelEncoder()
    combined_data = pd.concat([train[col], test[col]], axis=0).astype(str)
    le.fit(combined_data)
    train[col] = le.transform(train[col].astype(str))
    test[col] = le.transform(test[col].astype(str))


from sklearn.preprocessing import OneHotEncoder
one_hot_cols = ['매물확인방식', '주차가능여부']
one_hot_encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
train_encoded = one_hot_encoder.fit_transform(train[one_hot_cols])
test_encoded = one_hot_encoder.transform(test[one_hot_cols])
train = pd.concat([train.drop(columns=one_hot_cols), pd.DataFrame(train_encoded, index=train.index)], axis=1)
test = pd.concat([test.drop(columns=one_hot_cols), pd.DataFrame(test_encoded, index=test.index)], axis=1)

train = train.drop(columns=['ID'])
test_id = test['ID']
test = test.drop(columns=['ID'])


X = train.drop(columns=['허위매물여부'])
y = train['허위매물여부']

X.columns = X.columns.astype(str)


smote = SMOTE(random_state=42)
X_sm, y_sm = smote.fit_resample(X, y)


print("After SMOTE, X_sm shape:", X_sm.shape)
print("After SMOTE, y_sm distribution:\n", pd.Series(y_sm).value_counts())


def f1_metric(y_pred, data):
    y_true = data.get_label()
    y_pred_binary = (y_pred > 0.5).astype(int)
    f1 = f1_score(y_true, y_pred_binary)
    return 'f1', f1, True

params = {
    'objective': 'binary',
    'boosting_type': 'gbdt',
    'num_leaves': 31,
    'learning_rate': 0.05,
    'class_weight': 'balanced',
    'seed': 42
}

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# oof_preds와 test_preds 초기화 (SMOTE 이후 데이터 크기로 조정)
oof_preds = np.zeros(len(X_sm))
test_preds = np.zeros(len(test))


for fold, (train_idx, val_idx) in enumerate(skf.split(X_sm, y_sm)):
    print(f"Fold {fold + 1}")
    X_train, X_val = X_sm.iloc[train_idx], X_sm.iloc[val_idx]
    y_train, y_val = y_sm.iloc[train_idx], y_sm.iloc[val_idx]

    train_data = lgb.Dataset(X_train, label=y_train)
    val_data = lgb.Dataset(X_val, label=y_val)

    model = lgb.train(
        params,
        train_data,
        valid_sets=[val_data],
        feval=f1_metric, 
        num_boost_round=1000,
        early_stopping_rounds=50,
        verbose_eval=100
    )

    oof_preds[val_idx] = model.predict(X_val)  # val_idx는 SMOTE 이후 인덱스를 사용
    test_preds += model.predict(test) / skf.n_splits


oof_preds_binary = (oof_preds > 0.5).astype(int)
print("OOF F1 Score:", f1_score(y_sm, oof_preds_binary, average='macro'))

cm = confusion_matrix(y_sm, oof_preds_binary)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()
test_preds_binary = (test_preds > 0.5).astype(int)
submission = pd.DataFrame({'ID': test_id, '허위매물여부': test_preds_binary})
submission.to_csv('submission_2.csv', index=False)
print("Submission file saved to 'submission_2.csv'")