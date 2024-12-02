import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('C:/Users/r2com/Documents/.vscode/selenium/boston.csv')#절대 경로
# df = pd.read_csv('./boston.csv')
X = df.drop(columns = 'MEDV')
y = df[['MEDV']]
#pip install statsmodels
# import statsmodels.api as sm
# X_constant = sm.add_constant(X)
# model_1 = sm.OLS(y,X_constant)
# lin_reg = model_1.fit()
# print(lin_reg.summary())

from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_score

# lr = LinearRegression()
# neg_mse_scores = cross_val_score(lr,X,y,scoring='neg_mean_squared_error',cv=5)#n
# rmse_scores = np.sqrt(-1*neg_mse_scores)
# avr_rmse = np.mean(rmse_scores)

# print('5 folds의 개별 Negative MSE scires: ',  np.round(neg_mse_scores,2)) 
# print('5 folds의 개별 RMSE scores :',np.round(rmse_scores,2))
# print('5 folds의 평균 RMSE:{0:.3f}'.format(avr_rmse))
from sklearn.model_selection import KFold

#예시 데이터
# X_data = np.arange(10).reshape(10,1)
# y_target = np.arange(10)
# kf = KFold(n_splits=5,shuffle=True,random_state=42)#K_fold 객체 생성(5개의 객체로 나눔)

# fold_idx = 1 #Fold 정보확인
# for train_index, test_index in kf.split(X_data):
#     print(f'Fold {fold_idx}:')
#     print(f'Train indices: {train_index}')
#     print(f'Test indices: {test_index}')
#     print(f'Train data: {X_data[train_index].flatten()}')
#     print(f'Test data: {X_data[test_index].flatten()}')
#     fold_idx +=1
    
# from sklearn.preprocessing import PolynomialFeatures #다항회귀 패키지

# X = np.arange(4).reshape(2,2)
# print('일차 단항식 계수 feature:\n',X)

# #defree=2인 2차 다항식으로 변환하기 위해 Polynomial Features를 이용하여 변환.
# ploy = PolynomialFeatures(degree=3).fit_transform(X)
# print('3차 다항식 계수 feature: \n', ploy)

# def polynomail_func(X):
#     y = 1+2*X+X**2 +X**3
#     return y
# y = polynomail_func(X)
# #linear regression에 3차 다항식 계수 feature와 3차 다항식 결정값으로 학습 후 회귀 계수 확인
# from sklearn.linear_model import LinearRegression
# model = LinearRegression()
# model.fit(ploy, y)
# print('Polynomial 회귀 계수 : \n', np.round(model.coef_,2))
# print('Polynomial 회귀 shape : ',model.coef_.shape)

from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline

def polynomail_func(X):
    y = 1 + 2 * X + X**2+X**3
    return y
#Pipeline 객체로 streamline하게 Polynomial Feature 변환과 Linear regression을 연결
model = Pipeline([('Poly', PolynomialFeatures(degree=3)),
                  ('linear', LinearRegression())])
X = np.arange(4).reshape(2,2)
y = polynomail_func(X)

model = model.fit(X,y)
print('Polynomial 회귀 계수 : \n', np.round(model.named_steps['linear'].coef_,2))                  