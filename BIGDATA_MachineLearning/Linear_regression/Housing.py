import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mlxtend.plotting import heatmap
from mlxtend.plotting import scatterplotmatrix
from LinearRegressionGD import LinearRegressionGD
from sklearn.preprocessing import StandardScaler,PolynomialFeatures
from sklearn.linear_model import RANSACRegressor,LinearRegression
df = pd.read_csv('https://raw.githubusercontent.com/rasbt/'
                 'python-machine-learning-book-3rd-edition/'
                 'master/ch10/housing.data.txt',
                 header=None,
                 sep='\s+')
df.columns=['CRIM', 'ZN', 'INDUS', 'CHAS', 
              'NOX', 'RM', 'AGE', 'DIS', 'RAD', 
              'TAX', 'PTRATIO', 'B', 'LSTAT', 'MEDV']

# 1. CRIM      1인당 범죄 발생률 (해당 지역의 범죄율)
# 2. ZN        25,000 평방피트(약 700평) 이상의 주거용 토지가 차지하는 비율
# 3. INDUS     비소매업(공업, 사무실 등) 지역이 차지하는 비율
# 4. CHAS      찰스강 접경 여부 (1 = 강을 접하고 있음, 0 = 접하지 않음)
# 5. NOX       대기 중 질소산화물 농도 (1,000만분의 1 단위)
# 6. RM        주택 1가구당 평균 방 개수
# 7. AGE       1940년 이전에 건설된 주택의 비율
# 8. DIS       보스턴 주요 고용센터까지의 가중 거리
# 9. RAD       방사형 고속도로 접근성 지수
# 10. TAX      재산세율 (10,000달러당 세금)
# 11. PTRATIO  지역별 학생-교사 비율
# 12. B        흑인 거주 비율 지표 (1000 * (Bk - 0.63)^2, Bk는 해당 지역의 흑인 비율)
# 13. LSTAT    저소득층 인구 비율 (%)
# 14. MEDV     자가주택의 중앙값 (단위: 1,000달러)
# print(df.head())
cols = ['LSTAT', 'INDUS', 'NOX', 'RM', 'MEDV']

scatterplotmatrix(df[cols].values,figsize=(10,8),names=cols,alpha=0.5)



print(df[cols].values)#샘플개수X특징개수
cm=np.corrcoef(df[cols].values.T)#특징개수 X 샘플개수
hm=heatmap(cm,row_names=cols,column_names=cols)

X=df[['RM']].values#2차원 배열로 변환해야 dot함수가 계산됨
y=df['MEDV'].values
sc_x=StandardScaler()
sc_y=StandardScaler()
X_xtd=sc_x.fit_transform(X)
y_std=sc_y.fit_transform(y[:,np.newaxis]).flatten()
lr=LinearRegressionGD()
lr.fit(X_xtd,y_std)
plt.plot(range(1, lr.n_iter+1), lr.cost_)
plt.ylabel('SSE')
plt.xlabel('Epoch')
plt.tight_layout()
plt.show()

#-----------------------QR 분해 해결하기,그람슈미트 과정 



ransac=RANSACRegressor(LinearRegression(),
                       max_trials=100,
                       min_samples=50,
                       loss='absolute_error',
                       residual_threshold=5.0,
                       random_state=0)
ransac.fit(X,y)

inlier_mask=ransac.inlier_mask_#True값 저장
outlier_mask=np.logical_not(inlier_mask)#False 값을 True로 저장장
line_X=np.arange(3,10,1)
line_y_ransac=ransac.predict(line_X[:, np.newaxis])



