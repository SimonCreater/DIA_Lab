import pandas as pd
from io import StringIO
from sklearn.impute import SimpleImputer
import numpy as np
from sklearn.preprocessing import FunctionTransformer,MinMaxScaler
from sklearn.model_selection import train_test_split

csv_data = """
A,B,C,D
1.0,5.0,9.0,
2.0,,10.0,14.0
,,11.0,15.0
4.0,8.0,,16.0
"""

df = pd.read_csv(StringIO(csv_data))
print(df)
imr=SimpleImputer(missing_values=np.nan,strategy='mean')
imr=imr.fit(df.values)

ftr_imr=FunctionTransformer(lambda X:imr.fit_transform(X.T).T,validate=False)
imputed_data=ftr_imr.fit_transform(df.values)
print(imputed_data)
df=pd.DataFrame([
  ['green','M',10.1,'class2'],
  ['red','L',13.5,'class1'],
  ['blue','XL',15.3,'class2']
])
df.columns=['color','size','price','classlabel']
size_mapping={
  'XL':3,
  'L':2,
  'M':1
}
df['size']=df['size'].map(size_mapping)
print(df)

class_mapping={label:idx for idx,label in enumerate(np.unique(df['classlabel']))}


df_wine = pd.read_csv('https://archive.ics.uci.edu/'
                      'ml/machine-learning-databases/wine/wine.data',
                      header=None)

df_wine.columns = ['Class label', 'Alcohol', 'Malic acid', 'Ash',
                   'Alcalinity of ash', 'Magnesium', 'Total phenols',
                   'Flavanoids', 'Nonflavanoid phenols', 'Proanthocyanins',
                   'Color intensity', 'Hue', 'OD280/OD315 of diluted wines',
                   'Proline']
X,y=df_wine.iloc[:, 1:].values,df_wine.iloc[:,0].values

X_train,X_test,y_train,y_test=train_test_split(X, y, 
                     test_size=0.3, 
                     random_state=0, 
                     stratify=y)
#특성 스케일 맞추기 결정트리,랜덤 포레스트
#최소-최대 스케일 변환
np.random.seed(0)
# 픽셀 값 정규화
image = np.random.randint(0, 256, (50, 50), dtype=np.uint8)
normalized_image=image/255.0
#좌표값 스케일링
bounding_boxes = np.array([
    [5, 10, 40, 35],  # 첫 번째 경계 상자
    [0, 0, 50, 50]    # 두 번째 경계 상자
])

scaler = MinMaxScaler()
scaled_boxes = scaler.fit_transform(bounding_boxes)
print("Scaled Bounding Boxes (0~1):", scaled_boxes)

# 원본 크기로 복원
original_boxes = scaler.inverse_transform(scaled_boxes)
print("Restored Bounding Boxes (Original Scale):", original_boxes)
#