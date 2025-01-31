import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mlxtend.plotting import heatmap
from mlxtend.plotting import scatterplotmatrix
from LinearRegressionGD import LinearRegressionGD
from sklearn.preprocessing import StandardScaler,PolynomialFeatures
from sklearn.linear_model import RANSACRegressor,LinearRegression
import Housing
lr=LinearRegression()
pr=LinearRegression()
quadratic=PolynomialFeatures(degree=2)
X_quad=quadratic.fit_transform(Housing.X)
X = np.array([258.0, 270.0, 294.0, 
              320.0, 342.0, 368.0, 
              396.0, 446.0, 480.0, 586.0])\
             [:, np.newaxis]

y = np.array([236.4, 234.4, 252.8, 
              298.6, 314.2, 342.2, 
              360.8, 368.0, 391.2,
              390.8])
lr.fit(Housing.X,Housing.y)
X_fit=np.arange(250,600,10)[:, np.newaxis]

y_lin_fit=lr.predict(X_fit)
pr.fit(X_quad,y)
y_quad_fit=pr.predic(quadratic.fit_transform(X_fit))
