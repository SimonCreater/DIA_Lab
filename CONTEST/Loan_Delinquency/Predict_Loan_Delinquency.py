import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
from sklearn.preprocessing import OneHotEncoder
from xgboost import XGBClassifier
import warnings
warnings.filterwarnings(action='ignore')

train_df=pd.read_csv('./train.csv').drop(columns=['UID'])
test_Df=pd.read_csv('./test.csv').drop(columns=['UID'])

categorical_col = [
    '주거 형태',
    '현재 직장 근속 연수',
    '대출 목적',
    '대출 상환 기간'
]
