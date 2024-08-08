from sklearn.datasets import make_regression
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
X,y=make_regression(n_samples=100,n_features=1,n_informative=1,n_targets=1,noise=20,random_state=13)

plt.scatter(X,y)
plt.show()

X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=2)
lr=LinearRegression()
lr.fit(X_train,y_train)
print(lr.coef_)
print(lr.intercept_)

y_pred=lr.predict(X_test)
print(r2_score(y_test,y_pred))
print(np.mean(cross_val_score(lr,X,y,scoring='r2',cv=10)))
m=28.46

class GDRegressor:
    def __init__(self,learning_rate,epochs):
        self.m=100
        self.b=-120
        self.lr=learning_rate
        self.epochs=epochs
    def fit(self,X,y):
        # Calculate the b using GD
        for i in range(self.epochs):
            loss_slope_b=-2*np.sum(y-self.m*X.ravel()-self.b)
            loss_slope_m=-2*np.sum((y-self.m*X.ravel()-self.b)*X.ravel())

            self.b=self.b - (self.lr * loss_slope_b)
            self.m=self.m - (self.lr * loss_slope_m)
            # print(loss_slope,self.b)
        print(self.m,self.b)
    def predict(self,X):
        return self.m*X + self.b 

gd=GDRegressor(0.001,100)

print(gd.fit(X_train,y_train))
print(gd.predict(X))

y_pred=gd.predict(X_test)
print(r2_score(y_test,y_pred))