import numpy as np
class LinearRegressionGD(object):
  def __init__(self,eta=0.001,n_iter=20):
    self.eta=eta#학습률
    self.n_iter=n_iter#반복
  def fit(self,X,y):
    self.w_=np.zeros(1+X.shape[1])
    self.cost_=[]

    for i in range(self.n_iter):
      output=self.net_input(X)#예측
      errors=(y-output)
      self.w_[1:]+=self.eta*X.T.dot(errors)# w1,s2... 가중치 업데이트(특징의 대해서 업데이트 하므로 전취 즉 가로로 만들어준 후 곱한다 이거 좀 어렵네...)
      self.w_[0]+=self.eta*errors.sum()
      cost=(errors**2).sum()/2.0#오차(errors)를 제곱한 후 더한 값(SSE)을 2로 나눠 비용 함수 값을 계산
      self.cost_.append(cost)
    return self
  def net_input(self,X):
    return np.dot(X,self.w_[1:])+self.w_[0]#dot에서 ,X는 가로 Self.W_는 세로로 곱함

  def predict(self,X):
    return self.net_input(X)
    